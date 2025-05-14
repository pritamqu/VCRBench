
#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate vcr

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

base_model_name="qwen2_5_vl_7b"
model_weights="/model-weights/Qwen2.5-VL-7B-Instruct"
response_file="./output/${base_model_name}/response.json"

python -m models.qwen2_5_vl \
    --base_model_name ${base_model_name} \
    --model-path ${model_weights} \
    --output-file ${response_file} \
    --fps 1 \

# eval
python -m process --response_file $response_file



