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

<p class="first-authors">† First authors</p>

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


Introduced in the paper "[Roboflow 100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models](https://media.roboflow.com/rf100vl/rf100vl.pdf)", RF100-VL is a large-scale collection of 100 multi-modal datasets with diverse concepts not commonly found in VLM pre-training.

The benchmark includes images, with corresponding annotations, from seven domains: flora and fauna, sport, industry, document processing, laboratory imaging, aerial imagery, and miscellaneous datasets related to various use cases for which detection models are commonly used.

You can use RF100-VL to benchmark fully supervised, semi-supervised and few-shot object detection models, and Vision Language Models (VLMs) with localization capabilities.

![](https://media.roboflow.com/rf100vl/results.png)

## Download RF100-VL

To download RF100-VL, first install the `rf100vl` pip package:

```
pip install rf100vl
```

RF100-VL is hosted on Roboflow Universe, the world's largest repository of annotated computer vision dataset. You will need a free Roboflow Universe API key to download the dataset. [Learn how to find your API key]()

Export your API key into an environment variable called `ROBOFLOW_API_KEY`:

```
export ROBOFLOW_API_KEY=YOUR_KEY
```

Several helper functions are available to download RF100-VL and its subsets. These are split up into two categories: functions that retrieve Dataset objects with the name of each project and its category. (that start with `get_`), and data downloaders (that start with `download_`).

| Data Loader Name               | Dataset Name           |
|--------------------------------|------------------------|
| `get_rf100vl_fsod_projects`      | RF100-VL-FSOD          |
| `get_rf100vl_projects`           | RF100-VL               |
| `get_rf20vl_fsod_projects`       | RF20-VL-FSOD           |
| `get_rf20vl_full_projects`       | RF20-VL           |
| `download_rf100vl_fsod`          | RF100-VL-FSOD          |
| `download_rf100vl`               | RF100-VL               |
| `download_rf20vl_fsod`           | RF20-VL-FSOD           |
| `download_rf20vl_full`           | RF20-VL           |

Each dataset object has its own `download` method.

Here is an example showing how to download the full dataset:

```python
from rf100vl import download_rf100vl

download_rf100vl(path="./rf100-vl/")
```

The datasets will be downloaded in COCO JSON format to a directory called `rf100-vl`. Every dataset will be in its own sub-folder.

## CVPR 2025 Workshop Challenge: Few-Shot Object Detection from Annotator Instructions

**Organized by:** Anish Madan, Neehar Peri, Deva Ramanan, Shu Kong

### Introduction

This challenge focuses on few-shot object detection (FSOD) with 10 examples of each class provided by a human annotator. Existing FSOD benchmarks repurpose well-established datasets like COCO by partitioning categories into base and novel classes for pre-training and fine-tuning respectively. However, these benchmarks do not reflect how FSOD is deployed in practice.

Rather than pre-training on only a small number of base categories, we argue that it is more practical to download a foundational model (e.g., a vision-language model (VLM) pretrained on web-scale data) and fine-tune it for specific applications. We propose a new FSOD benchmark protocol that evaluates detectors pre-trained on any external dataset (not including the target dataset), and fine-tuned on K-shot annotations per C target classes.

We evaluate a subset of 20 datasets from Roboflow-VL. Each dataset is independently evaluated using AP. Roboflow-VL includes datasets that are out-of-distribution from typical internet-scale pre-training data, making it a particularly challenging (even for VLMs) for Foundational FSOD..

### Benchmarking Protocols

**Goal:** Developing robust object detectors using few annotations provided by annotator instructions. The detector should detect object instances of interest in real-world testing images.

**Environment for model development:**
- **Pretraining:** Models are allowed to pre-train on any existing datasets.
- **Fine-Tuning:** Models can fine-tune on 10 shots from each of RF100-VL's datasets
- **Evaluation:** Models are evaluated on RF100-VL's test set. Each dataset is evaluated independently.

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

