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
</div>


Introduced in the paper "Roboflow 100-VL: A Multi-Domain Object Detection
Benchmark for Vision-Language Models", RF100-VL is a large-scale collection of 100 multi-modal datasets with diverse concepts not commonly found in VLM pre-training.

The benchmark includes images, with corresponding annotations, from seven domains: flora and fauna, sport, industry, document processing, laboratory imaging, aerial imagery, and miscellaneous datasets related to various use cases for which detection models are commonly used.

You can use RF100-VL to benchmark fully supervised, semi-supervised and few-shot object detection models, and Vision Language Models (VLMs) with localization capabilities.

![](https://media.roboflow.com/rf100vl/results.png)

## Download RF100-VL

To download RF100-VL, first clone this repository and install it from source:

```
git clone https://github.com/roboflow/rf100-vl
pip3 install -e .
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

## Acknowledgements

This work was supported in part by compute provided by NVIDIA, and the NSF GRFP (Grant No. DGE2140739).

## License

The datasets that comprise RF100-VL are licensed under an [Apache 2.0 license](LICENSE).

