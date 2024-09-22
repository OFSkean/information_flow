#!/bin/bash
USE_SLURM=1
MODEL_NAME="mamba"
MODEL_SIZES=('130m' '370m' '790m')
REVISION="main"
PURPOSE="run_entropy_metrics"

for size in ${MODEL_SIZES[@]}; do
    if [ $USE_SLURM -eq 1 ]; then
        sbatch slurm_submit.sh \
            --model_family $MODEL_NAME \
            --model_size $size \
            --revision $REVISION \
            --evaluation_layer -1 \
            --purpose $PURPOSE
    else
        python experiments/mteb-harness.py \
            --model_family $MODEL_NAME \
            --model_size $size \
            --revision $REVISION \
            --evaluation_layer -1 \
            --base_results_path "experiments/results" \
            --purpose $PURPOSE
    fi
done
