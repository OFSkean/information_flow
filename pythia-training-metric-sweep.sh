#!/bin/bash

USE_SLURM=1
MODEL_NAME="Pythia"
MODEL_SIZES=('410m')
PURPOSE="run_entropy_metrics"

pythia_revision_steps=(64 128 256 512 1000 2000 4000 8000 16000 32000 64000 128000)
REVISIONS=()
for step in "${pythia_revision_steps[@]}"; do
    REVISIONS+=("step$step")
done

for size in ${MODEL_SIZES[@]}; do
    for revision in "${REVISIONS[@]}"; do
        if [ $USE_SLURM -eq 1 ]; then
            JOBNAME="step$revision"
            sbatch -J $JOBNAME slurm_submit.sh \
                --model_family $MODEL_NAME \
                --model_size $size \
                --revision $revision \
                --evaluation_layer -1 \
                --purpose $PURPOSE
        else
            python experiments/mteb-harness.py \
                --model_family $MODEL_NAME \
                --model_size $size \
                --revision $revision \
                --evaluation_layer -1 \
                --base_results_path "experiments/results" \
                --purpose $PURPOSE
        fi
    done
done
