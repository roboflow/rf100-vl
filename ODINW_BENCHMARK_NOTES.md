# ODinW Benchmark Notes

## Dataset Structure Quirks

When benchmarking models on ODinW datasets, note the following special cases:

### Directory Structure Special Cases

1. **PascalVOC**
   - Uses `valid` folder instead of `test` folder for evaluation
   - Script should automatically use `split="valid"` when dataset is PascalVOC

2. **pistols**
   - Uses `export` folder as the root directory (not standard train/test/valid)
   - Contains test split in `/root/odinw/pistols/export/`

### ODinW-13 Datasets

The 13 datasets used in the original GLIP ODinW benchmark:
1. AerialMaritimeDrone (large variant)
2. Aquarium
3. CottontailRabbits
4. EgoHands (generic variant)
5. NorthAmericaMushrooms
6. Packages
7. PascalVOC ⚠️ uses `valid` split
8. Raccoon
9. ShellfishOpenImages
10. VehiclesOpenImages
11. pistols ⚠️ uses `export` directory
12. pothole
13. thermalDogsAndPeople

### ODinW-35 Datasets

The full 35 dataset benchmark includes:
- All 13 datasets from ODinW-13
- 22 additional datasets with various configurations
- Some datasets have multiple variants (e.g., AerialMaritimeDrone has both `large` and `tiled`)

## Implementation Notes

### Handling Special Cases in Code

```python
def find_dataset_root(dataset_path):
    dataset_name = os.path.basename(dataset_path)

    # pistols: use 'export' directory
    if dataset_name == "pistols":
        export_dir = os.path.join(dataset_path, "export")
        if os.path.exists(export_dir):
            return export_dir

    # PascalVOC: uses 'valid' instead of 'test'
    if dataset_name == "PascalVOC":
        if split == "test":
            split = "valid"
```

### Zero-Shot Evaluation

For zero-shot evaluation on ODinW:
- Use class names directly (no "a photo of a" prefix)
- Threshold: 0.01
- Model: OWLv2-Large recommended

## Directory Locations

- **ODinW data**: `/root/odinw/`
- **Predictions**: `/root/predictions/odinw13_owlv2_zeroshot/`
- **Benchmark script**: `/root/benchmark_odinw_owlv2.py`
- **Multi-GPU launcher**: `/root/run_odinw13_8gpus.py`

## References

- [GLIP ODinW Paper](https://github.com/microsoft/GLIP)
- [ODinW Challenge](https://eval.ai/web/challenges/challenge-page/1839/overview)
