#!/bin/bash
USE_SLURM=1

MODEL_NAME="mamba"
MODEL_SIZES=('130m' '370m' '790m' '1.4b' '2.8b')
MAX_LAYER=70

for size in ${MODEL_SIZES[@]}; do
    for layer in $(seq 0 $MAX_LAYER); do
        if [ $USE_SLURM -eq 1 ]; then
            sbatch slurm_submit.sh $MODEL_NAME $size $layer
        else
            echo "Running evaluation for $MODEL_NAME $size layer $layer"
            python mteb-harness.py --model_family $MODEL_NAME --model_size $size --revision main --evaluation_layer $layer
        fi
    done
done
