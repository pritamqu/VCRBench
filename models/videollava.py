
# ref https://huggingface.co/docs/transformers/en/model_doc/video_llava

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
import numpy as np
import av
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def generate_response(
        model, 
        processor,
        prompt,
        video_path, 
        max_frames=8,
        max_new_tokens=128):
    
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / max_frames).astype(int)
    video = read_video_pyav(container, indices)

    prompt = f"USER: <video>\n{prompt} ASSISTANT:"
    inputs = processor(text=prompt, videos=video, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(response)

    return response



def run_inference(args):

    # load model and model related confis
    
    model = VideoLlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16, 
                                                               attn_implementation="flash_attention_2",
                                                               device_map="auto")
    processor = VideoLlavaProcessor.from_pretrained(args.model_path)


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
    parser.add_argument("--max_frames", type=int, required=False, default=8,)

    args = parser.parse_args()
    run_inference(args)

