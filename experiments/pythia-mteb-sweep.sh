#!/bin/bash
MODEL_NAME="Pythia"
MODEL_SIZES=('14m' '70m' '160m' '410m' '1b' '1.4b' '2.8b')
MAX_LAYER=50
for size in ${MODEL_SIZES[@]}; do
    for layer in $(seq 0 $MAX_LAYER); do
        echo "Running evaluation for $MODEL_NAME $size layer $layer"
        python mteb-harness.py --model_family $MODEL_NAME --model_size $size --revision main --evaluation_layer $layer
    done
done
