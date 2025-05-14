
import argparse
import json
import re
import os
from collections import defaultdict
from metrics import calc

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--response_file', help='Path to the ground truth file containing question.', required=True)
    return parser.parse_args()


def fetch_predicted_order(text):
    # return a list of integers indicating the correct step
    pred_order=[]
    try:
        for part in str2int(text):
            int_n=None
            try:
                int_n=int(re.search(r'\d+', part).group())
            except:
                pass

            if int_n is not None:
                pred_order.append(int_n)
    except Exception as e:
        # print('[Error]: ', text)
        # raise ValueError(e)
        pass
    
    return pred_order

def str2int(text):
    parts = text.split(',')
    parts = [part.strip() for part in parts]
    return parts

def compare_lists(gt, pred):
    # Ensure both lists are of the same length
    if len(gt) != len(pred):
        # print("Lists must be of the same length")
        return [0]*len(gt)

    # Compare the lists and create the result list
    result = [1 if a == b else 0 for a, b in zip(gt, pred)]
    return result


def extract_following_text(text, phrase):
    # Split text into sentences
    # sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    
    # Compile regex for case-insensitive phrase matching
    phrase_regex = re.compile(re.escape(phrase), re.IGNORECASE)
    
    # Extract parts following the matched phrase
    results = []
    for sentence in sentences:
        match = phrase_regex.search(sentence)
        if match:
            results.append(sentence[match.end():].strip())  # Get text after the phrase
    
    return results

def is_consecutive_in_range(seq, a, b):
    """Check if seq contains all consecutive numbers within [a, b]."""
    return set(seq) == set(range(a, b + 1))

def calc_scores(args):

    if args.response_file.endswith('.jsonl'):
        results=[json.loads(q) for q in open(args.response_file, 'r')]
        result_file=args.response_file[:-6]+'_result.json'
        modified_result_file=args.response_file[:-6]+'_updated.json'
    elif args.response_file.endswith('.json'):
        results=json.load(open(args.response_file))
        result_file=args.response_file[:-5]+'_result.json'
        modified_result_file=args.response_file[:-5]+'_updated.json'
    else:
        raise ValueError(args.response_file)

    # process here
    phrases=['', 
             'correct order is:', 
             'correct order:', 
             'correct order', 
             '**Correct order:**', 
             '*Correct order:*',
             'follow these steps in order', 
             'the correct order is',
             'the correct order is', 
             'the final output should be',
             ] # FIXME: add more such catching phrase based on your model's behavior
    for res in results:
        pred_order_found=None
        if isinstance(res['pred'], list):
            pred=res['pred'][0]
        else:
            pred=res['pred']
        for phrase in phrases:
            pred=pred.replace('*', '') # for gemini
            pred_orders=extract_following_text(pred, phrase=phrase)
            if len(pred_orders)==0:
                continue
            for pred_order in pred_orders:
                try:
                    pred_order=fetch_predicted_order(pred_order)
                except Exception as e:
                    pass
                    # print(e)
                
                # if their lengths are matched we can expect to find the right text
                # other wise keep checking - and if not foudn, we will fill zeros
                if len(pred_order)==len(res['answer']) and is_consecutive_in_range(pred_order, 1, len(res['answer'])): 
                    pred_order_found=pred_order
                    break
            
            break

        if pred_order_found is None:
            print(f"[Invalid Response Found]: {pred}\nCorrect answer: {res['answer']}")
            # there are invalid seq of numbers; invild if the total seq length does not match with the predicted
            # or they just did not answer in proper way
            # print('############ MISSED ############')
            # print(pred)
            # print('GT', res['answer'])
            # print('############')
            pred_order_found=[0]*len(res['answer']) # if can not fetch ans just feed 0s
            
        res['pred_order']=pred_order_found

    scores = calc(results, gtfile="./HF_DATA/data.json")
    json.dump(scores, open(result_file, 'w'), indent=2)
    json.dump(results, open(modified_result_file, 'w'), indent=4)

    return 


if __name__=='__main__':
    args=parse_args()
    calc_scores(args)
    

