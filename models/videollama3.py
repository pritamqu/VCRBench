
# ref https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B

import json
import torch
import copy
import random
import argparse
import os
from tqdm import tqdm
from utils import set_deterministic
from data import VCRBench, valid_modes
set_deterministic(seed=42)
import math
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor


def generate_response(
        model, 
        processor,
        prompt,
        video_path, 
        max_frames=128,
        max_new_tokens=128):

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": max_frames}},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(response)

    return response



def run_inference(args):

    # load model and model related confis
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)



    # load data
    dataset = VCRBench(args.question_file, 
                        video_root=args.video_folder,
                        mode=args.mode,
                        )
    
    # output = []
    generation_config = dict(max_new_tokens=args.model_max_length, do_sample=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file=open(args.output_file, "w",)
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
        if video is None:
            raise NotImplementedError()
        else:
            pred= generate_response(
                        model, 
                        processor,
                        question,
                        video, 
                        max_frames=args.max_frames,
                        max_new_tokens=args.model_max_length
            )

        sample_set['pred'] = pred
        # print(pred)

        # print(sample_set)
        # output.append(sample_set)
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # with open(os.path.join(args.output_file), "w",) as f:
    #     json.dump(output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--base_model_name", type=str, default="random")
    parser.add_argument('--model-path', help='base model weights', required=True)
    parser.add_argument('--model-path2', help='additional weights such as LORA', default=None, type=str, required=False)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=False, default='HF_DATA')
    parser.add_argument("--mode", type=str, default='default', choices=valid_modes)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=False, default='./data/data.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=False, default='./output/random/response.json')
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    parser.add_argument("--max_frames", type=int, required=False, default=64,)

    args = parser.parse_args()
    run_inference(args)

