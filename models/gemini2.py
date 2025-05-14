


import math
import os
import argparse
import json
import os
import json
import torch
import time
from tqdm import tqdm
from itertools import chain
from transformers.trainer_pt_utils import IterableDatasetShard
from data.dataset import VCRBench
from utils import set_deterministic
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
set_deterministic(seed=42)

def generate_response(model, question, video_path):

    if video_path is not None:
        video_file = genai.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            print('[ERROR: Failed to upload the file]')
            return 
    
    response=None
    question=question.strip()
    generation_config={'max_output_tokens':args.model_max_length}
    safety_settings={
                    'HATE': 'BLOCK_NONE',
                    'HARASSMENT': 'BLOCK_NONE',
                    'SEXUAL' : 'BLOCK_NONE',
                    'DANGEROUS' : 'BLOCK_NONE'
                    }
    while response is None:
        try:
            if video_path is not None:
                response = model.generate_content([video_file, question],
                                        # request_options={"timeout": 600}, 
                                            # safety_settings=safety_settings,
                                            # generation_config=generation_config,
                                        )
            else:
                response = model.generate_content([question],
                                        # request_options={"timeout": 600}, 
                                            # safety_settings=safety_settings,
                                            # generation_config=generation_config,
                                        )
        except Exception as e:
            print(e)
            print('retrying...')
            time.sleep(10)
            continue

    print(response)

    try:
        outputs = response.to_dict()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print("[ERROR]", e)
        # outputs=None
        outputs="FAILED"

    if video_path is not None:
        genai.delete_file(video_file.name)

    return outputs
    

def run_inference(args):

    model = genai.GenerativeModel(model_name=args.model_path)
    dataset = VCRBench(args.question_file, 
                        video_root=args.video_folder,
                        mode=args.mode,
                        )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.path.isfile(args.output_file):
        seen=[json.loads(s)['qid'] for s in open(args.output_file)]
        ans_file=open(args.output_file, "a",)
    else:
        seen=[]
        ans_file=open(args.output_file, "w",)
    # output = []

    # iterate over each sample in the ground truth file
    for idx, sample in enumerate(tqdm(dataset)):
        if sample['qid'] in seen:
            print(f"[skipped] {sample['qid']}")
            continue
        video = sample['video'] 
        # raise FileNotFoundError
        question = sample['question']
        question=question.strip()

        sample_set = {'qid': sample['qid'], 
                      'question': question, 
                      'answer': sample['answer'], 
                      'video_file': sample['video_file']}
        
        # inference
        pred=generate_response(model, question, video)    
        if pred is None:   
            continue    
        sample_set['pred'] = pred

        # print(pred)
        # output.append(sample_set)

        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    # with open(os.path.join(args.output_file), "w",) as f:
    #     json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--base_model_name", type=str, default="gemini")
    parser.add_argument('--model-path', help='pass the name of the Gemini model', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=False, default='HF_DATA')
    parser.add_argument("--mode", type=str, default="default")
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=False, default='./data/data.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    
    args = parser.parse_args()
    run_inference(args)
