
#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate vcr # FIXME: got error in version of protobuf

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

base_model_name="videollava_7b" 
model_weights="/datasets/video_llm/model_weights/Video-LLaVA-7B-hf" 
response_file="./output/${base_model_name}/response.jsonl"

python -m models.videollava \
    --base_model_name ${base_model_name} \
    --model-path ${model_weights} \
    --output-file ${response_file} \
    --max_frames 8

# eval
python -m process --response_file $response_file



