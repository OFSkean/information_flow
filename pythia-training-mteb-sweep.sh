#!/bin/bash
USE_SLURM=1

MODEL_NAME="Pythia"
MODEL_SIZES=('410m')
MAX_LAYER=30

pythia_revision_steps=(1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 16000 32000 64000 128000)
REVISIONS=("main")
for step in "${pythia_revision_steps[@]}"; do
    REVISIONS+=("step$step")
done

for size in ${MODEL_SIZES[@]}; do
    for layer in $(seq 0 $MAX_LAYER); do
        for revision in "${REVISIONS[@]}"; do
            if [ $USE_SLURM -eq 1 ]; then
                sbatch slurm_submit.sh $MODEL_NAME $size $layer $revision
            else
                echo "Running evaluation for $MODEL_NAME $size layer $layer"
                python mteb-harness.py --model_family $MODEL_NAME --model_size $size --revision main --evaluation_layer $layer
            fi
        done
    done
done
