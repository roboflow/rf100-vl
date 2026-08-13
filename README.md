<div align="center">
<h2>Roboflow 100-VL:<br>A Multi-Domain Object Detection
Benchmark <br>for Vision-Language Models</h2>

Peter Robicheaux <sup>1†</sup>
Matvei Popov<sup>1†</sup>
Anish Madan <sup>2</sup>
Isaac Robinson <sup>1</sup>

Joseph Nelson <sup>1</sup>
Deva Ramanan <sup>2</sup>
Neehar Peri <sup>2</sup>

<a target="_blank" href="https://roboflow.com">Roboflow</a>&nbsp;&nbsp;&nbsp;
<a target="_blank" href="https://www.cmu.edu/">Carnegie Mellon University</a>

<p class="first-authors">† Equal Contribution</p>

<div>
<!-- <a href="https://www.arxiv.org/pdf/2502.13130" target="_blank">
  <img src="https://img.shields.io/badge/📄_Paper-arXiv-red?style=for-the-badge" alt="Paper" />
</a>&nbsp; -->
<a href="https://universe.roboflow.com/rf100-vl/" target="_blank">
  <img src="https://img.shields.io/badge/🌐_Datasets-Roboflow_Universe-blue?style=for-the-badge" alt="Datasets" />
</a>&nbsp;
<a href="https://rf100-vl.org" target="_blank">
  <img src="https://img.shields.io/badge/🔗_Website-rf100--vl.org-green?style=for-the-badge" alt="Website" />
</a>
</div>
</div>

Introduced in the paper "[Roboflow 100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models](https://arxiv.org/pdf/2505.20612)", RF100-VL is a large-scale collection of 100 multi-modal datasets with diverse concepts not commonly found in VLM pre-training.

The benchmark includes images, with corresponding annotations, from seven domains: flora and fauna, sport, industry, document processing, laboratory imaging, aerial imagery, and miscellaneous datasets related to various use cases for which detection models are commonly used.

You can use RF100-VL to benchmark fully supervised, semi-supervised and few-shot object detection models, and Vision Language Models (VLMs) with localization capabilities.

## Download RF100-VL

To download RF100-VL, first install the `rf100vl` pip package:

```
pip install rf100vl
```

RF100-VL is hosted on Roboflow Universe, the world's largest repository of annotated computer vision dataset. You will need a free Roboflow Universe API key to download the dataset. [Learn how to find your API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key)

Export your API key into an environment variable called `ROBOFLOW_API_KEY`:

```
export ROBOFLOW_API_KEY=YOUR_KEY
```

Several helper functions are available to download RF100-VL and its subsets. These are split up into two categories: functions that retrieve Dataset objects with the name of each project and its category. (that start with `get_`), and data downloaders (that start with `download_`).

| Data Loader Name | Dataset Name |
|--------------------------------|------------------------|
| `get_rf100vl_fsod_projects` | RF100-VL-FSOD |
| `get_rf100vl_projects` | RF100-VL |
| `get_rf20vl_fsod_projects` | RF20-VL-FSOD |
| `get_rf20vl_full_projects` | RF20-VL |
| `download_rf100vl_fsod` | RF100-VL-FSOD |
| `download_rf100vl` | RF100-VL |
| `download_rf20vl_fsod` | RF20-VL-FSOD |
| `download_rf20vl_full` | RF20-VL |

Each dataset object has its own `download` method.

Here is an example showing how to download the full dataset:

```python
from rf100vl import download_rf100vl

download_rf100vl(path="./rf100-vl/")
```

The datasets will be downloaded in COCO JSON format to a directory called `rf100-vl`. Every dataset will be in its own sub-folder.

### CLI

A command-line downloader is available via the optional `cli` extra:

```
pip install "rf100vl[cli]"
```

```
rf100vl download rf100vl ./rf100-vl/
```

Flags:

| Flag | Description |
|------|-------------|
| `dataset` | `rf100vl` or `rf20vl` (positional) |
| `path` | download destination (positional) |
| `--fsod` | use the few-shot object detection variant |
| `--index N` | download only dataset index N in the variant (zero-based; 0 = first dataset), instead of all |
| `--model_format` | annotation format, default `coco` |
| `--overwrite` | overwrite existing files, default `True` |
| `--api_key` | Roboflow API key, defaults to `ROBOFLOW_API_KEY` env var |

Examples:

```
rf100vl download rf100vl ./data                 # RF100-VL, full
rf100vl download rf100vl ./data --fsod           # RF100-VL-FSOD
rf100vl download rf20vl ./data --fsod            # RF20-VL-FSOD
rf100vl download rf100vl ./data --index 0        # first dataset by index
```

## Combine Datasets

Fold two or more RF100-VL sub-datasets into a single COCO dataset — one
unified label space, one set of `train/valid/test` folders — instead of
training against each dataset separately.

Category labels are namespaced per source dataset (`dataset:label`, e.g.
`deeppcb:open` vs `stomata-cells:open`) to avoid false-friend collisions
between datasets that happen to use the same word for different concepts.
Image filenames are prefixed with their source dataset (`bees_<file>.jpg`)
so they never collide once combined.

**Two ways to combine, matching two use cases:**

1. **Download + combine in one step** — `download(..., combine=True)` /
   `rf100vl download ... --combine`. Downloads the selected datasets into
   `<path>/.cache/` (kept, reused on future calls with an overlapping
   selection) and writes the combined dataset directly at
   `<path>/{train,valid,test}`.
2. **Combine datasets you already downloaded** — `combine_downloaded(...)`
   / `rf100vl combine`. No network access; operates in place on a
   directory that already contains per-dataset folders (e.g. from several
   plain `download` calls). **Destructive by default**: each per-dataset
   folder (`path/bees`, `path/deeppcb`, ...) is *moved* into the combined
   tree and removed once empty, so no disk space is duplicated. Pass
   `keep_originals=True` to copy instead and leave the source folders
   intact. Interrupted runs are resumable — progress is tracked in
   `path/.combine_manifest.json`, so re-running skips already-merged
   datasets instead of redoing them.

Python API:

```python
from rf100vl import download_and_combine, combine_downloaded

# download 3 datasets by global index and combine them
download_and_combine("./combined", indices=[0, 15, 29])

# or combine datasets you already downloaded under ./data
# (./data/bees, ./data/deeppcb, ... get folded into ./data/{train,valid,test})
combine_downloaded("./data")
```

CLI:

```
rf100vl download rf100vl ./combined --combine --indices 0,15,29
rf100vl combine ./data
rf100vl combine ./data --names bees,deeppcb --keep-originals
```

| Flag (`combine`) | Description |
|------|-------------|
| `path` | directory of already-downloaded per-dataset folders to combine (positional) |
| `--indices` | comma-separated global dataset indices to include, e.g. `0,3,7`; mutually exclusive with `--names` |
| `--names` | comma-separated canonical dataset basenames to include, e.g. `bees,deeppcb`; mutually exclusive with `--indices` |
| `--keep_originals` | copy instead of move — leaves source per-dataset folders untouched (default: `False`) |

`download`'s `--combine` flag adds `--indices`/`--names` (same meaning as
above) and requires one of `--index`, `--indices`, or `--names` to pick a
finite selection; not yet supported together with `--fsod`.

## CVPR 2025 Workshop Challenge: Few-Shot Object Detection from Annotator Instructions

**Organized by:** Anish Madan, Neehar Peri, Deva Ramanan

### Introduction

This challenge focuses on few-shot object detection (FSOD) with 10 examples of each class provided by a human annotator. Existing FSOD benchmarks repurpose well-established datasets like COCO by partitioning categories into base and novel classes for pre-training and fine-tuning respectively. However, these benchmarks do not reflect how FSOD is deployed in practice.

Rather than pre-training on only a small number of base categories, we argue that it is more practical to download a foundational model (e.g., a vision-language model (VLM) pretrained on web-scale data) and fine-tune it for specific applications. We propose a new FSOD benchmark protocol that evaluates detectors pre-trained on any external dataset (not including the target dataset), and fine-tuned on K-shot annotations per C target classes.

We evaluate a subset of 20 datasets from Roboflow-VL. Each dataset is independently evaluated using AP. Roboflow-VL includes datasets that are out-of-distribution from typical internet-scale pre-training data, making it a particularly challenging (even for VLMs) for Foundational FSOD.

:rotating_light: Top performing teams can win cash prizes! :rotating_light:

:1st_place_medal: 1st Place: $750

:2nd_place_medal: 2nd Place: $500

:3rd_place_medal: 3rd Place: $250

To be eligible for prizes, teams must submit a technical report, open source their code, and provide instructions on how to reproduce their results. Teams must also beat our best performing official baseline to be eligible for prizes. Many thanks to Roboflow for sponsoring prizes!

### Benchmarking Protocols

**Goal:** Developing robust object detectors using few annotations provided by annotator instructions. The detector should detect object instances of interest in real-world testing images.

**Environment for model development:**
- **Pretraining:** Models are allowed to pre-train on any existing datasets.
- **Fine-Tuning:** Models can fine-tune on 10 shots from each of RF20-VL-FSOD's datasets
- **Evaluation:** Models are evaluated on RF20-VL-FSOD's test set. Each dataset is evaluated independently.

**Evaluation metrics:**
- **AP:** The average precision of IoU thresholds from 0.5 to 0.95 with the step size 0.05.

### Submission Details

Submit a zip file with pickle files for each dataset. The name of each pickle file should match the name of each dataset. Each pickle file should use the following COCO format.

```json
[ 
"image_id": int,
"instances":
  [{ "image_id": int,
  "category_id": int,
  "bbox": [x,y,width,height],
  "score": float
  },
  {"image_id": int,
  "category_id": int,
  "bbox": [x,y,width,height],
  "score": float }, ... ,],
  ...,
]
```

We've provided a [sample submission](https://drive.google.com/file/d/1Pp8oAYMMnCxTzFa078NS3CuzdNlIEVzp/view) for your reference. Submissions should be uploaded to our [EvalAI server](https://eval.ai/web/challenges/challenge-page/2459/overview).

### Official Baseline

We pre-train Detic on ImageNet21-K, COCO Captions, and LVIS. We evaluate this pre-trained model zero-shot on the datasets in RF20-VL.

Our baseline code is available [here](https://github.com/anishmadan23/foundational_fsod/tree/fsod_rf20vl?tab=readme-ov-file).

### Timeline

- Submission opens: March 15th, 2025
- Submission closes: June 8th, 2025, 11:59 pm Pacific Time
- The top 3 participants on the leaderboard will be invited to give a talk at the workshop

### References

1. Madan et. al. "Revisiting Few-Shot Object Detection with Vision-Langugage Models". Proceedings of the Conference on Neural Information Processing Systems. 2024
2. Zhou et. al. "Detecting Twenty-Thousand Classes Using Image-Level Supervision". Proceedings of the IEEE European Conference on Computer Vision. 2022

## Acknowledgements

This work was supported in part by compute provided by NVIDIA, and the NSF GRFP (Grant No. DGE2140739).

## License

The datasets that comprise RF100-VL are licensed under an [Apache 2.0 license](LICENSE).

## Citation
If you find our paper and code repository useful, please cite us:
```bib
@article{robicheaux2025roboflow100vl,
  title={Roboflow100-VL: A multi-domain object detection benchmark for vision-language models},
  author={Robicheaux, Peter and Popov, Matvei and Madan, Anish and Robinson, Isaac and Nelson, Joseph and Ramanan, Deva and Peri, Neehar},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```
