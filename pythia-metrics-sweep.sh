#!/bin/bash
USE_SLURM=0

MODEL_NAME="Pythia"
MODEL_SIZES=('14m' '70m' '160m' '410m')
REVISION="main"

for size in ${MODEL_SIZES[@]}; do
    echo "Running evaluation for $MODEL_NAME $size layer $layer"
    python experiments/mteb-harness.py --model_family $MODEL_NAME --model_size $size --revision $REVISION --base_results_path "experiments/results" --purpose run_entropy_metrics
done
