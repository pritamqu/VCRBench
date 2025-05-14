
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
from decord import VideoReader, cpu
from PIL import Image
from io import BytesIO
import base64
from utils import set_deterministic
set_deterministic(seed=42)
from openai import OpenAI
import numpy as np

def load_video_base64(video_path, max_frames_num=32):
    vr = VideoReader(video_path, ctx=cpu(0))
    # total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    if len(frame_idx)>max_frames_num and max_frames_num!=-1:
        uniform_sampled_frames = np.linspace(0, len(vr) - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    print('video shape', spare_frames.shape)
    base64_frames = []

    for idx in range(spare_frames.shape[0]):
        frame=spare_frames[idx]
        # print('***', frame.shape)
        img = Image.fromarray(frame)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        base64_frames.append(img_base64)

    return base64_frames

def get_response_video(model_name, input_text, base64Frames, 
                       ):

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                input_text,
                *map(lambda x: {"image": x, "resize": 512}, base64Frames), # low: 256, medium: 512, high: 768
            ],
        },
    ]
    params = {
        "model": model_name,
        "messages": PROMPT_MESSAGES,
        "max_tokens": args.model_max_length,
    }

    response = None
    while response is None:
        try:
            response = client.chat.completions.create(**params)
        except Exception as e:
            print(e)
            print('retrying...')
            time.sleep(10)
            continue

    response = response.choices[0].message.content
    return response

def run_inference(args):

    
    dataset = VCRBench(args.question_file, 
                        video_root=args.video_folder,
                        mode=args.mode,
                        )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file=open(args.output_file, "w",)

    for idx, sample in enumerate(tqdm(dataset)):
        video = sample['video']
        # raise FileNotFoundError
        question = sample['question']
        question=question.strip()

        sample_set = {'qid': sample['qid'], 
                      'question': question, 
                      'answer': sample['answer'], 
                      'video_file': sample['video_file']}

        if video is None:
            raise NotImplementedError()
        
        # max_frames_num=-1 would allow taking all frames at 1fps
        base64Frames = load_video_base64(video, max_frames_num=32) 
        # pred=None
        pred=get_response_video(args.model_path, question, base64Frames)            
        sample_set['pred'] = pred

        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

        # break #TODO:

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--base_model_name", type=str, default="gpt")
    parser.add_argument('--model-path', help='pass the name of the GPT model', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=False, default='HF_DATA')
    parser.add_argument("--mode", type=str, default="default")
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=False, default='./data/data.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)
    run_inference(args)