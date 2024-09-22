#!/bin/bash
USE_SLURM=0

#MODEL_NAME="LLM2Vec-mntp-unsup-simcse"
MODEL_NAME="LLM2Vec-mntp"
MODEL_SIZES=('8B')
MAX_LAYER=32
REVISION="main"

for size in ${MODEL_SIZES[@]}; do
    for layer in $(seq 0 $MAX_LAYER); do
        if [ $USE_SLURM -eq 1 ]; then
            JOBNAME="llm2vec-$layer"
            sbatch -J $JOBNAME slurm_submit.sh \
                --model_family $MODEL_NAME \
                --model_size $size \
                --revision $REVISION \
                --evaluation_layer $layer \
                --purpose run_tasks
        else
            python experiments/mteb-harness.py \
                --model_family $MODEL_NAME \
                --model_size $size \
                --revision $REVISION \
                --evaluation_layer $layer \
                --base_results_path "experiments/results" \
                --purpose run_tasks
        fi
    done
done
