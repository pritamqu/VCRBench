
#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate vcr

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

mode="rrd" 
base_model_name="qwen2_5_vl_72b"
model_weights="/datasets/video_llm/model_weights/Qwen2.5-VL-72B-Instruct"
response_file="./output/${base_model_name}/${mode}.jsonl"

python -m models.qwen2_5_vl_rrd \
    --base_model_name ${base_model_name} \
    --model-path ${model_weights} \
    --output-file ${response_file} \
    --fps 1
    
# eval
python -m process --response_file $response_file



