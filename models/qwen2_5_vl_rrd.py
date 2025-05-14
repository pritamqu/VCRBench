import re
import json
import torch
import copy
import random
import argparse
import os
import time
from collections import defaultdict
from tqdm import tqdm
from utils import set_deterministic
from data import VCRBench, valid_modes
set_deterministic(seed=42)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, fetch_video
VERBOSE=False

def str2int(text):
    try:
        integer=int(text)
    except Exception as e:
        integer=int(re.search(r'\d+', text).group())
    
    return integer


class RRD(object):
    def __init__(self, 
                 model, 
                 processor, 
                 max_new_tokens=1024, 
                 fps=1.0) -> None:
        self.model=model
        self.processor=processor
        self.max_new_tokens=max_new_tokens
        self.fps=fps
        self.full_conversation=[]

    def load_video(self, video_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": ""},
                ],
            }
        ]
        _, video_inputs = process_vision_info(messages)


        return video_inputs

    def inference_video(self, prompt, 
                        video_inputs=None, 
                        video_path=None, 
                        max_new_tokens=None):

        if max_new_tokens is None:
            max_new_tokens=self.max_new_tokens

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": self.fps, 
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if video_inputs is None:
            _, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            fps=self.fps,
            padding=True,
            return_tensors="pt",
            # **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        if isinstance(output_text, list):
            assert(len(output_text))==1
            output_text=output_text[0]

        self.full_conversation.append({
            "user":'<video>\n'+prompt, 'assistant': output_text
        })

        return output_text

    def inference_language(self, prompt, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens=self.max_new_tokens
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if isinstance(output_text, list):
            assert(len(output_text))==1
            output_text=output_text[0]

        self.full_conversation.append({
            "user":prompt, 'assistant': output_text
        })

        return output_text

    def video_recognition(self, video_inputs, video_path, 
                            ):
        
        prompt='''The video contains multiple short clips.
        The clip numbers are mentioned at the beginning of each clip as Clip 1, Clip 2, and so on.
        Watch each clip carefully, paying attention to its fine-grained actions and events. 
        Note the unique events in each clip compared to the rest of the video.
        Respond with a one sentence description indicating the key and fine-grained actions or events for each clip.
        Please respond in this format:
        Clip 1: <Write one sentence description>
        Clip 2: <Write one sentence description>
        ...

        Your response must not contain anything else.
        '''        
        response=self.inference_video(prompt, video_inputs, video_path)

        return response

    def identify_number_of_events(self, video_inputs, video_path
                            ):
        
        prompt='''The given video consists of multiple short clips.
        The clip numbers are mentioned at the beginning of each clip as Clip 1, Clip 2, and so on.
        Watch the video carefully and count the total number of clips.
        Respond with only a number, nothing else.'''

        response=self.inference_video(prompt, video_inputs, video_path)

        return response

    def video_recognition_iterative(self, video_inputs, video_path, 
                            ):
        
        number_of_steps = self.identify_number_of_events(video_inputs, video_path)
        number_of_steps = str2int(number_of_steps)
        
        prompt='''The video contains multiple short clips.
        The clip numbers are mentioned at the beginning of each clip as Clip 1, Clip 2, and so on.
        Watch Clip {step} carefully, paying attention to its fine-grained actions and events. 
        Note the unique events in Clip {step} compared to the rest of the video.
        Respond with a one sentence description indicating the key and fine-grained actions or events.
        Your response must not contain anything else.
        DO NOT MENTION ANY CLIP NUMBERS IN YOUR RESPONSE.'''

        description_dict={}
        for step in range(number_of_steps):
            response=self.inference_video(prompt.format(step=step+1), video_inputs, video_path)
            description_dict[f'Clip {step+1}'] = response

        response=""
        for k in description_dict:
            response+=f"Clip {k}: {description_dict[k]}\n"
            
        return response


    def causal_reasoning(self, event_details, goal):
        
        prompt='''The following steps are needed to complete the task: {goal}. 
        However, these steps are randomly shuffled, and your job is to arrange them in the correct order to complete the task: {goal}. 
        Use your reasoning and common sense to arrange these steps to successfully complete the task.

        {clip_details}

        The final output should be in this format:

        Correct order: <mention the step numbers separated by a comma>
        '''
        
        response=self.inference_language(prompt.format(clip_details=event_details, 
                                                    goal=goal))

        return response

    def causal_reasoning_with_videos(self, event_details, goal, video_inputs, video_path):
        
        prompt='''The given video consists of multiple short clips, each showing a different segment needed to complete the task: {goal}. 
        These clips are randomly shuffled, and your job is to arrange them in the correct order to complete the task: {goal}. 
        The clip numbers are mentioned at the beginning of each clip as Clip 1, Clip 2, and so on. 
        Additionally, to assist you with the task, a brief description of the activities performed in each clip is provided below.
        Use your reasoning and common sense to arrange these clips to successfully complete the task.

        A brief description of each clip:
        {clip_details}

        The final output should be in this format:

        Correct order: <mention the clip numbers separated by a comma>
        '''

        response=self.inference_video(prompt.format(clip_details=event_details, 
                                                    goal=goal), video_inputs, video_path)

        return response

    def __call__(self, video_path, goal, 
                sequential_recognition=True,
                reason_with_videos=False, 
                ):

        video_inputs=self.load_video(video_path)
        if reason_with_videos:
            end=time.time()
            if sequential_recognition:
                event_details = self.video_recognition_iterative(video_inputs, video_path)
            else:
                event_details = self.video_recognition(video_inputs, video_path)
            response = self.causal_reasoning_with_videos(event_details, goal, video_inputs, video_path)
            elapsed_time.append(time.time()-end) # to measure time
            return response
        else:
            end=time.time()
            if sequential_recognition:
                event_details = self.video_recognition_iterative(video_inputs, video_path)
            else:
                event_details = self.video_recognition(video_inputs, video_path)
            response = self.causal_reasoning(event_details, goal)
            elapsed_time.append(time.time()-end) # to measure time
            return response
        
        


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
                        mode='rrd',
                        )
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file=open(args.output_file, "w",)
    # iterate over each sample in the ground truth file
    global elapsed_time
    elapsed_time=[]
    for idx, sample in enumerate(tqdm(dataset)):
        
        # print(sample)
        video = sample['video']
        goal = sample['question']
        goal = goal.strip()
        sample_set = {'qid': sample['qid'], 
                      'question': '', 
                      'answer': sample['answer'], 
                      'video_file': sample['video_file']}
        
        rrd_module=RRD(model=model, processor=processor, max_new_tokens=args.model_max_length, 
                fps=args.fps)
        pred = rrd_module(video, goal, 
                        reason_with_videos=args.reason_with_videos,
                        sequential_recognition=args.sequential_recognition
                        )
        
        sample_set['pred'] = pred
        sample_set['RRD'] = rrd_module.full_conversation
        # print(pred)
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()

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
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=False, default='./data/data.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=False, default='./output/random/response.json')
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    # parser.add_argument("--max_frames", type=int, required=False, default=-1, help="when -1 take all frames at 1fps")
    parser.add_argument("--fps", type=float, required=False, default=1, help="1fps")
    parser.add_argument('--sequential_recognition', action='store_true')
    parser.add_argument('--reason_with_videos', action='store_true')

    args = parser.parse_args()
    run_inference(args)

