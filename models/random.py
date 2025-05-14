


import math
import os
import argparse
import json
import os
import json
import torch
import copy
import random
from tqdm import tqdm
from itertools import chain
from utils import set_deterministic
from data import VCRBench, valid_modes
# set_deterministic(seed=42)

def run_inference(args):

    # load model and model related confis
    model=None

    # load data
    dataset = VCRBench(args.question_file, 
                        video_root=args.video_folder,
                        mode=args.mode,
                        )
    
    output = []
    # iterate over each sample in the ground truth file
    for idx, sample in enumerate(tqdm(dataset)):
        # print(sample)
        video = sample['video']
        question = sample['question']
        question=question.strip()
        sample_set = {'qid': sample['qid'], 
                      'question': question, 
                      'answer': sample['answer'], 
                      'video_file': sample['video_file']}

        # inference
        pred=None
        pred_order=copy.copy(sample['answer'])
        random.shuffle(pred_order)
        pred=['Clip '+str(o) for o in pred_order]
        pred=', '.join(pred)
        pred=pred.strip()

        sample_set['pred'] = pred

        # print(sample_set)
        output.append(sample_set)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(os.path.join(args.output_file), "w",) as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--base_model_name", type=str, default="random")
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=False, default='HF_DATA')
    parser.add_argument("--mode", type=str, default='default', choices=valid_modes)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=False, default='./HF_DATA/data.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=False, default='./output/random/response.json')
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model_max_length", type=int, required=False, default=128)

    args = parser.parse_args()
    run_inference(args)

     
