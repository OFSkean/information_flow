#!/bin/bash
USE_SLURM=1

MODEL_NAME="bert"
MODEL_SIZES=('base', 'large')
MAX_LAYER=30

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
