
#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate pp

TRIAL=$1
base_model_name="random"
response_file="./output/random/response_"${TRIAL}".json"

python -m models.random \
    --base_model_name ${base_model_name} \
    --model-path './dummy-path' \
    --output-file $response_file \

python -m process --response_file ${response_file}
