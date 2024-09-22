#!/bin/bash
USE_SLURM=0

#MODEL_NAME="LLM2Vec-mntp-unsup-simcse"
MODEL_NAME="LLM2Vec-mntp"
MODEL_SIZE='8B'
REVISION="main"
PURPOSE="run_entropy_metrics"

if [ $USE_SLURM -eq 1 ]; then
    sbatch -J $JOBNAME slurm_submit.sh \
        --model_family $MODEL_NAME \
        --model_size $MODEL_SIZE \
        --revision $REVISION \
        --evaluation_layer -1 \
        --purpose $PURPOSE
else
    python experiments/mteb-harness.py \
        --model_family $MODEL_NAME \
        --model_size $MODEL_SIZE \
        --revision $REVISION \
        --evaluation_layer -1 \
        --base_results_path "experiments/results" \
        --purpose $PURPOSE
fi
