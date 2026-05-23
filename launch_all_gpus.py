#!/usr/bin/env python3
"""Launch OWLv2 benchmark across all 8 GPUs for all RF100-VL datasets."""

import subprocess
import os

# All 100 datasets
ALL_DATASETS = [
    "-grccs", "13-lkc01", "2024-frc", "actions", "activity-diagrams",
    "aerial-airport", "aerial-cows", "aerial-pool", "aerial-sheep",
    "aircraft-turnaround-dataset", "all-elements",
    "apoce-aerial-photographs-for-object-detection-of-construction-equipment",
    "aquarium-combined", "asphaltdistressdetection", "ball", "bees",
    "bibdetection", "buoy-onboarding", "cable-damage", "canalstenosis",
    "car-logo-detection", "circuit-voltages", "clashroyalechardetector",
    "cod-mw-warzone", "conveyor-t-shirts", "countingpills",
    "crystal-clean-brain-tumors-mri-dataset", "dataconvert", "deepfruits",
    "deeppcb", "defect-detection", "dentalai",
    "electric-pylon-detection-in-rsi", "everdaynew",
    "exploratorium-daphnia", "flir-camera-objects", "floating-waste",
    "football-player-detection", "fruitjes", "grapes-5", "grass-weeds",
    "gwhd2021", "halo-infinite-angel-videogame", "human-detection-in-floods",
    "inbreast", "infraredimageofpowerequipment", "into-the-vale",
    "invoice-processing", "ism-band-packet-detection", "jellyfish",
    "l10ul502", "label-printing-defect-version-2", "lacrosse-object-detection",
    "liver-disease", "macro-segmentation", "mahjong", "marine-sharks",
    "needle-base-tip-min-max", "new-defects-in-wood", "nih-xray",
    "orgharvest", "orionproducts", "paper-parts", "peixos-fish",
    "penguin-finder-seg", "pig-detection", "pill", "recode-waste",
    "roboflow-trained-dataset", "screwdetectclassification",
    "sea-cucumbers-new-tiles", "signatures", "smd-components",
    "soda-bottles", "speech-bubbles-detection", "spinefrxnormalvindr",
    "sssod", "stomata-cells", "taco-trash-annotations-in-context",
    "the-dreidel-project", "thermal-cheetah", "tomatoes-2", "trail-camera",
    "train", "truck-movement", "tube", "uavdet-small", "ufba-425",
    "underwater-objects", "urine-analysis1",
    "varroa-mites-detection--test-set", "water-meter", "wb-prova",
    "weeds4", "wheel-defect-detection", "wildfire-smoke", "wine-labels",
    "x-ray-id", "xray", "zebrasatasturias"
]

# Distribute across 8 GPUs
NUM_GPUS = 8
datasets_per_gpu = len(ALL_DATASETS) // NUM_GPUS
remainder = len(ALL_DATASETS) % NUM_GPUS

print(f"Starting benchmark on {len(ALL_DATASETS)} RF100-VL datasets across {NUM_GPUS} GPUs...")
print("This will take several hours. Progress will be logged to gpu_*.log files.")

processes = []
start_idx = 0

for gpu_id in range(NUM_GPUS):
    # Calculate how many datasets for this GPU
    num_datasets = datasets_per_gpu + (1 if gpu_id < remainder else 0)
    end_idx = start_idx + num_datasets

    gpu_datasets = ALL_DATASETS[start_idx:end_idx]
    start_idx = end_idx

    print(f"GPU {gpu_id}: {len(gpu_datasets)} datasets")

    # Build command (use -- to separate datasets from flags)
    cmd = [
        "python3", "benchmark_owlv2.py",
        "--gpu_id", str(gpu_id),
        "--datasets",
        "--"
    ] + gpu_datasets

    # Launch process
    log_file = open(f"gpu_{gpu_id}.log", "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd="/root"
    )
    processes.append((proc, log_file))
    print(f"  Launched PID {proc.pid}")

print(f"\nAll {NUM_GPUS} GPU processes launched in background!")
print("Monitor progress with: tail -f gpu_*.log")
print("Check running processes with: ps aux | grep benchmark_owlv2")
