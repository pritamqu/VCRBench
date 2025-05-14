
# ref https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
# Qwen/Qwen2.5-VL-7B-Instruct
# Qwen/Qwen2.5-VL-72B-Instruct


import json
import torch
import copy
import random
import argparse
import os
import time
from tqdm import tqdm
from collections import defaultdict
from utils import set_deterministic
from data import VCRBench, valid_modes
set_deterministic(seed=42)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def generate_response(
        model, 
        processor,
        prompt,
        video_path, 
        max_new_tokens=128, 
        fps=1.0):

    # fps=1.0
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    end=time.time() # to measure time
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt",
        # **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # print(output_text)
    elapsed_time[number_of_events].append(time.time()-end) # to measure time

    return output_text


def generate_response_language_only(
        model, 
        processor,
        prompt,
        max_new_tokens=128, 
        fps=1.0):

    
    # Messages containing a local video path and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        fps=fps,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text



def run_inference(args):

    # load model and model related confis

    if args.base_model_name=='qwen2_5_vl_7b':
        MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto", 
        )
        
    elif args.base_model_name=='qwen2_5_vl_72b':
        MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct"
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto", 
        )

    else:
        raise NotImplementedError(args.base_model_name)

    # load data
    dataset = VCRBench(args.question_file, 
                        video_root=args.video_folder,
                        mode=args.mode,
                        )
    
    output = []
    global elapsed_time
    global number_of_events
    elapsed_time=defaultdict(list)
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

        number_of_events=len(sample['answer'])
        
        # inference
        if video is None:
            pred=generate_response_language_only(model, 
                                processor, 
                                prompt=question, 
                                # video_path=video, 
                                max_new_tokens=args.model_max_length)
        else:
            pred=generate_response(model, 
                                processor, 
                                prompt=question, 
                                video_path=video, 
                                max_new_tokens=args.model_max_length, 
                                fps=args.fps)
        

        sample_set['pred'] = pred
        # print(pred)

        # print(sample_set)
        output.append(sample_set)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(os.path.join(args.output_file), "w",) as f:
        json.dump(output, f, indent=4)

    time_file='.'.join(args.output_file.split('.')[:-1])+'_time.json'
    with open(os.path.join(time_file), "w",) as f:
        json.dump(elapsed_time, f, indent=1)

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
    parser.add_argument("--fps", type=float, required=False, default=1, help="1fps")

    args = parser.parse_args()
    run_inference(args)

