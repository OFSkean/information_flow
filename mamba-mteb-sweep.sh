#!/bin/bash
USE_SLURM=1

MODEL_NAME="mamba"
MODEL_SIZES=('130m' '370m' '790m' '1.4b' '2.8b')
MAX_LAYER=70
REVISION="main"

for size in ${MODEL_SIZES[@]}; do
    for layer in $(seq 0 $MAX_LAYER); do
        if [ $USE_SLURM -eq 1 ]; then
            sbatch slurm_submit.sh $MODEL_NAME $size $REVISION $layer
        else
            echo "Running evaluation for $MODEL_NAME $size layer $layer"
            python experiments/mteb-harness.py --model_family $MODEL_NAME --model_size $size --revision $REVISION --evaluation_layer $layer
        fi
    done
done
