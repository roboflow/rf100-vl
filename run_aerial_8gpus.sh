#!/bin/bash

# Run benchmark on aerial datasets across 8 GPUs

DATASETS=(
    "aerial-airport"
    "aerial-cows"
    "aerial-pool"
    "aerial-sheep"
    "apoce-aerial-photographs-for-object-detection-of-construction-equipment"
)

# Distribute datasets across 8 GPUs
# GPU 0: aerial-airport
# GPU 1: aerial-cows
# GPU 2: aerial-pool
# GPU 3: aerial-sheep
# GPU 4: apoce-aerial-photographs-for-object-detection-of-construction-equipment
# GPUs 5-7: idle (only 5 aerial datasets)

python3 benchmark_owlv2.py --datasets aerial-airport --gpu_id 0 &
python3 benchmark_owlv2.py --datasets aerial-cows --gpu_id 1 &
python3 benchmark_owlv2.py --datasets aerial-pool --gpu_id 2 &
python3 benchmark_owlv2.py --datasets aerial-sheep --gpu_id 3 &
python3 benchmark_owlv2.py --datasets apoce-aerial-photographs-for-object-detection-of-construction-equipment --gpu_id 4 &

# Wait for all background processes to complete
wait

echo "All aerial datasets processed!"
