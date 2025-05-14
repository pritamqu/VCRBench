
#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate vcr

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

base_model_name="gemini"
model_weights="gemini-2.0-flash-thinking-exp"
response_file="./output/${model_weights}/response.jsonl"

python -m models.gemini2 \
    --base_model_name ${base_model_name} \
    --model-path ${model_weights} \
    --output-file ${response_file} \

# eval
python -m process --response_file $response_file

