
import torch
import os
import json

PROMPT='''The given video consists of multiple short clips, each showing a different segment needed to complete the task: {goal}. 
These clips are randomly shuffled, and your job is to arrange them in the correct order to complete the task: {goal}. 
The clip numbers are mentioned at the beginning of each clip as Clip 1, Clip 2, and so on.
In order to solve this task, first, you should identify the activity that is performed in each clip, and then use your reasoning and common sense to arrange these clips to successfully complete the task.

The final output should be in this format:

Correct order: <mention the Clip numbers separated by a comma>
'''

pp_prompts=dict(
    default=PROMPT,
    rrd="",
)
valid_modes=list(pp_prompts.keys())

class VCRBench(torch.utils.data.Dataset):
    """basic dataset for inference"""

    def __init__(
        self,
        question_file,
        video_root,
        mode='default',
        video_load_fn=None,
        load_video_kwargs={}
    ) -> None:
        super(VCRBench, self).__init__()
        self.data = json.load(open(question_file))
        self.video_load_fn = video_load_fn
        self.load_video_kwargs = load_video_kwargs
        assert mode in valid_modes, print(f"choose from: {valid_modes}")
        self.mode = mode
        self.prompt = pp_prompts[mode]
        self.video_root = os.path.join(video_root, 'videos')

        print(f"video root: {self.video_root}")

    def __len__(self) -> int:
        return len(self.data)
    
    def prepare_question(self, sample):
        if self.mode in ['default']:
            prompt=self.prompt.format(goal=sample['goal'])
        elif self.mode in ['rrd']:
            prompt=sample['goal']
        else:
            raise ValueError()
        return prompt
    
    def prepare_answer(self, sample):
        gt = [o+1 for o in sample['ground_truth']]
        return gt

    def prepare_sample(self, sample):
        video_file = os.path.join(self.video_root, sample['video_file'])
        if self.video_load_fn:
            video = self.video_load_fn(video_file, **self.load_video_kwargs)
        else:
            video = video_file

        return {'video': video, 
                'video_file': sample['video_file'],
                'qid': sample['qid'],
                'question': self.prepare_question(sample), 
                'answer': self.prepare_answer(sample)}

    def __getitem__(self, i):
        sample = self.data[i]
        return self.prepare_sample(sample)
    

if __name__=='__main__':

    dataset=VCRBench(question_file="data/data.json", 
                    video_root="./HF_DATA",
                    mode='default', 
                    )
        
    for sample in dataset:
        print(f"question: {sample['question']}")
        print(f"video file: {sample['video_file']}")
        print(f"answer: {sample['answer']}")
        print('*'*10)

        break

        