#!/bin/bash
#SBATCH --job-name=caching
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=2                   # number of nodes
#SBATCH --gres=gpu:2               # number of GPUs per node
#SBATCH --time=30:00             # maximum execution time (HH:MM:SS)
#SBATCH --qos=big_qos

conda init
conda activate kohya
unset LD_LIBRARY_PATH

WORK_DIR=sd-scripts
MODEL_NAME_OR_PATH=MODEL.safetensors
JSON_RESULT_PATH=$WORK_DIR/finetune/json_results
IN_JSON=INFO.json
cd $WORK_DIR

# assert files are present
if [ ! -f $IN_JSON ]; then
    echo "File $IN_JSON not found!"
    exit 1
fi
if [ ! -d $JSON_RESULT_PATH ]; then
    mkdir -p $JSON_RESULT_PATH
fi
if [ ! -d $MODEL_NAME_OR_PATH ]; then
    echo "Model $MODEL_NAME_OR_PATH not found!"
    exit 1
fi

CUDA_DEVICES_NUM=4 # number of GPUs

for i in $(seq 0 $((CUDA_DEVICES_NUM-1)))
do
    CUDA_VISIBLE_DEVICES=$i python finetune/prepare_buckets_latents.py --in_json $IN_JSON --out_json ${JSON_RESULT_PATH}_${i}.json --split_dataset --n_split $CUDA_DEVICES_NUM --current_index $i --model_name_or_path $MODEL_NAME_OR_PATH --max_resolution "1024,1024" --max_bucket_reso 4096 --full_path --recursive &
done

wait

# merge jsons
python finetune/merge_jsons.py --jsons "${JSON_RESULT_PATH}_*.json" --out_json ${JSON_RESULT_PATH}.json
