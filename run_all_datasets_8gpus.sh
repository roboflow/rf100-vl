#!/bin/bash

# Run benchmark on all RF100-VL datasets across 8 GPUs
# This distributes 100 datasets evenly: 12-13 datasets per GPU

# Get all dataset names
ALL_DATASETS="-grccs 13-lkc01 2024-frc actions activity-diagrams aerial-airport aerial-cows aerial-pool aerial-sheep aircraft-turnaround-dataset all-elements apoce-aerial-photographs-for-object-detection-of-construction-equipment aquarium-combined asphaltdistressdetection ball bees bibdetection buoy-onboarding cable-damage canalstenosis car-logo-detection circuit-voltages clashroyalechardetector cod-mw-warzone conveyor-t-shirts countingpills crystal-clean-brain-tumors-mri-dataset dataconvert deepfruits deeppcb defect-detection dentalai electric-pylon-detection-in-rsi everdaynew exploratorium-daphnia flir-camera-objects floating-waste football-player-detection fruitjes grapes-5 grass-weeds gwhd2021 halo-infinite-angel-videogame human-detection-in-floods inbreast infraredimageofpowerequipment into-the-vale invoice-processing ism-band-packet-detection jellyfish l10ul502 label-printing-defect-version-2 lacrosse-object-detection liver-disease macro-segmentation mahjong marine-sharks needle-base-tip-min-max new-defects-in-wood nih-xray orgharvest orionproducts paper-parts peixos-fish penguin-finder-seg pig-detection pill recode-waste roboflow-trained-dataset screwdetectclassification sea-cucumbers-new-tiles signatures smd-components soda-bottles speech-bubbles-detection spinefrxnormalvindr sssod stomata-cells taco-trash-annotations-in-context the-dreidel-project thermal-cheetah tomatoes-2 trail-camera train truck-movement tube uavdet-small ufba-425 underwater-objects urine-analysis1 varroa-mites-detection--test-set water-meter wb-prova weeds4 wheel-defect-detection wildfire-smoke wine-labels x-ray-id xray zebrasatasturias"

# Convert to array
read -ra DATASETS <<< "$ALL_DATASETS"

# Split datasets across 8 GPUs
GPU0_DATASETS="${DATASETS[@]:0:13}"
GPU1_DATASETS="${DATASETS[@]:13:12}"
GPU2_DATASETS="${DATASETS[@]:25:12}"
GPU3_DATASETS="${DATASETS[@]:37:13}"
GPU4_DATASETS="${DATASETS[@]:50:12}"
GPU5_DATASETS="${DATASETS[@]:62:13}"
GPU6_DATASETS="${DATASETS[@]:75:12}"
GPU7_DATASETS="${DATASETS[@]:87:13}"

echo "Starting benchmark on all 100 RF100-VL datasets across 8 GPUs..."
echo "This will take several hours. Progress will be logged to gpu_*.log files."

# Launch processes in background
python3 benchmark_owlv2.py --datasets $GPU0_DATASETS --gpu_id 0 > gpu_0.log 2>&1 &
python3 benchmark_owlv2.py --datasets $GPU1_DATASETS --gpu_id 1 > gpu_1.log 2>&1 &
python3 benchmark_owlv2.py --datasets $GPU2_DATASETS --gpu_id 2 > gpu_2.log 2>&1 &
python3 benchmark_owlv2.py --datasets $GPU3_DATASETS --gpu_id 3 > gpu_3.log 2>&1 &
python3 benchmark_owlv2.py --datasets $GPU4_DATASETS --gpu_id 4 > gpu_4.log 2>&1 &
python3 benchmark_owlv2.py --datasets $GPU5_DATASETS --gpu_id 5 > gpu_5.log 2>&1 &
python3 benchmark_owlv2.py --datasets $GPU6_DATASETS --gpu_id 6 > gpu_6.log 2>&1 &
python3 benchmark_owlv2.py --datasets $GPU7_DATASETS --gpu_id 7 > gpu_7.log 2>&1 &

echo "All 8 GPU processes launched in background!"
echo "Monitor progress with: tail -f gpu_*.log"
echo "Check running processes with: ps aux | grep benchmark_owlv2"
echo "Wait for completion with: wait"
