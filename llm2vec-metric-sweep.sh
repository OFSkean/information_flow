#!/bin/bash
USE_SLURM=0

MODEL_NAME="LLM2Vec-mntp-unsup-simcse"
MODEL_SIZES=('8B')
REVISION="main"

for size in ${MODEL_SIZES[@]}; do
    echo "Running evaluation for $MODEL_NAME $size layer $layer"
    python experiments/mteb-harness.py --model_family $MODEL_NAME --model_size $size --revision $REVISION  --base_results_path "experiments/results" --purpose 'run_entropy_metrics'
done
