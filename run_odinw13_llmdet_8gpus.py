#!/usr/bin/env python3
import subprocess
import time
import os

# ODinW-13 dataset list
odinw13_datasets = [
    "AerialMaritimeDrone",
    "Aquarium",
    "CottontailRabbits",
    "EgoHands",
    "NorthAmericaMushrooms",
    "Packages",
    "PascalVOC",
    "Raccoon",
    "ShellfishOpenImages",
    "VehiclesOpenImages",
    "pistols",
    "pothole",
    "thermalDogsAndPeople"
]

# Configuration
odinw_dir = "/root/odinw"
output_dir = "/root/predictions/odinw13_llmdet_zeroshot"
model_name = "iSEE-Laboratory/llmdet_large"
threshold = 0.01
num_gpus = 8

# Distribute datasets across GPUs
# Try to balance by dataset size (estimated)
gpu_assignments = [
    ["AerialMaritimeDrone", "ShellfishOpenImages"],  # GPU 0
    ["Aquarium", "VehiclesOpenImages"],               # GPU 1
    ["CottontailRabbits", "pistols"],                 # GPU 2 - pistols is large!
    ["EgoHands", "pothole"],                          # GPU 3 - EgoHands is large
    ["NorthAmericaMushrooms", "thermalDogsAndPeople"], # GPU 4
    ["Packages"],                                      # GPU 5
    ["PascalVOC"],                                     # GPU 6 - largest!
    ["Raccoon"]                                        # GPU 7
]

print("="*80)
print("ODinW-13 LLMDet Zero-Shot Evaluation on 8 H100 GPUs")
print("="*80)
print(f"Total datasets: {len(odinw13_datasets)}")
print(f"Model: {model_name}")
print(f"Threshold: {threshold}")
print(f"Output directory: {output_dir}")
print("="*80)
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Launch processes
processes = []
log_files = []

for gpu_id, datasets in enumerate(gpu_assignments):
    if not datasets:
        continue

    log_file = f"/root/odinw13_llmdet_gpu{gpu_id}.log"
    log_files.append(log_file)

    print(f"GPU {gpu_id}: Processing {len(datasets)} datasets: {datasets}")
    print(f"  Log file: {log_file}")

    # Prepare command
    cmd = [
        "python", "/root/benchmark_odinw_llmdet.py",
        "--odinw_dir", odinw_dir,
        "--output_dir", output_dir,
        "--model_name", model_name,
        "--threshold", str(threshold),
        "--gpu_id", str(gpu_id),
        "--datasets"
    ] + datasets

    # Set CUDA_VISIBLE_DEVICES to isolate GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Update gpu_id in command to 0 since CUDA_VISIBLE_DEVICES remaps it
    cmd[cmd.index("--gpu_id") + 1] = "0"

    # Open log file
    log_f = open(log_file, 'w')

    # Launch process
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env
    )

    processes.append((proc, log_f, gpu_id))

print()
print(f"Launched {len(processes)} processes")
print(f"Monitor progress with: tail -f /root/odinw13_llmdet_gpu*.log")
print()

# Monitor processes
start_time = time.time()
completed = 0

while processes:
    time.sleep(5.0)

    # Check which processes have completed
    remaining = []
    for proc, log_f, gpu_id in processes:
        if proc.poll() is not None:
            # Process completed
            log_f.close()
            completed += 1
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] GPU {gpu_id} completed ({completed}/{len(gpu_assignments)})")
        else:
            remaining.append((proc, log_f, gpu_id))

    processes = remaining

# All done
elapsed = time.time() - start_time
minutes = elapsed / 60

print()
print("="*80)
print(f"All processes completed in {elapsed:.1f} seconds ({minutes:.1f} minutes)")
print("="*80)
print(f"Results saved to: {output_dir}")
print()

# Check for errors in log files
print("Checking for errors...")
for log_file in log_files:
    with open(log_file, 'r') as f:
        content = f.read()
        if "Error" in content or "Traceback" in content:
            print(f"  ⚠️ {log_file} contains errors - check the log")
        elif "Saved" in content:
            print(f"  ✓ {log_file} completed successfully")
        else:
            print(f"  ? {log_file} status unclear")
print()
