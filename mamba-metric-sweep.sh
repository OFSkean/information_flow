#!/bin/bash
USE_SLURM=0

MODEL_NAME="mamba"
MODEL_SIZES=('130m' '370m')
REVISION="main"

for size in ${MODEL_SIZES[@]}; do
    echo "Running evaluation for $MODEL_NAME $size layer $layer"
    python experiments/mteb-harness.py --model_family $MODEL_NAME --model_size $size --revision $REVISION --base_results_path "experiments/results" --purpose run_entropy_metrics
done
