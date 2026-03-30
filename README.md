<img src="docs/vesselfm_banner_updated.png">

**TL;DR**: VesselFM is a foundation model for universal 3D blood vessel segmentation. It is trained on three heterogeneous data sources: a large, curated annotated dataset, synthetic data generated through domain randomization, and data sampled from a flow matching-based deep generative model. These data sources provide enough diversity to enable vesselFM to achieve exceptional *zero*-shot blood vessel segmentation, even in completely unseen domains. For details, please refer to our [manuscript](https://openaccess.thecvf.com/content/CVPR2025/html/Wittmann_vesselFM_A_Foundation_Model_for_Universal_3D_Blood_Vessel_Segmentation_CVPR_2025_paper.html).

---


## 🟢 Installation
First, set up a conda environment and install dependencies:

    conda create -n vesselfm python=3.9

    conda activate vesselfm

    pip install -e .

Python 依赖包：
```bash
uv lock --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync

```

## 🟢 TopBrain 数据与权重下载
脚本位于 `scripts/`，默认会读取根目录 `.env` 中的变量：

```bash
DATASET_DIR=./data/datasets/topBrain-2025
DATASET_ROOT_DIR=./data/datasets
HF_HOME=./modelsweights
```

1) 下载 TopBrain 2025 数据集（Zenodo）并整理目录：

```bash
./scripts/download_topbrain_data.sh
```

- 下载到：`${DATASET_ROOT_DIR}/download`
- 解压整理到：`${DATASET_DIR}`（自动合并 `imagesTr/labelsTr/imagesTs/labelsTs`，若存在）
- 默认来源：
  - https://zenodo.org/records/16623496
  - https://zenodo.org/records/16878417

2) 下载 VesselFM 预训练权重：

```bash
./scripts/download_vesselfm_weights.sh
```

- 下载到：`${HF_HOME}`
- 默认下载源（Hugging Face 仓库 `bwittmann/vesselFM`）：
  - `https://huggingface.co/bwittmann/vesselFM/resolve/main/vesselFM_base.pt`
  - `https://huggingface.co/bwittmann/vesselFM/resolve/main/meta.yaml`

可选参数：
- 指定数据集 URL：`./scripts/download_topbrain_data.sh <url1> <url2> ...`
- 指定权重文件：`./scripts/download_vesselfm_weights.sh <hf_file1> <hf_file2> ...`

## 🟢 *Zero*-Shot Segmentation
If you are solely interested in running vesselFM's inference script for *zero*-shot segmentation of data at hand, adjust the respecitve [config file](vesselfm/seg/configs/inference.yaml) (see `#TODO`) and run:

    python vesselfm/seg/inference.py

Additional information on inference, pre-training, and fine-tuning are available [here](./vesselfm/seg). Checkpoints will be downloaded automatically and are also available on [Hugging Face 🤗](https://huggingface.co/bwittmann/vesselFM).


## 🟢 Data Sources
<img src="docs/data_sources.png">

We also provide individual instructions for generating our three proposed data sources.

$\mathcal{D}_\text{drand}$: Domain randomized synthetic data ([here](./vesselfm/d_drand)).

$\mathcal{D}_\text{flow}$: Synthetic data sampled from our flow matching-based deep generative model ([here](./vesselfm/d_flow)).

$\mathcal{D}_\text{real}$: Real data curated from 17 annotated blood vessel segmentation datasets ([here](./vesselfm/d_real)).


## 🟢 Citing vesselFM
If you find our work useful for your research, please cite:

```bibtex
@InProceedings{Wittmann_2025_CVPR,
    author    = {Wittmann, Bastian and Wattenberg, Yannick and Amiranashvili, Tamaz and Shit, Suprosanna and Menze, Bjoern},
    title     = {vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20874-20884}
}
```

## 🟢 License
Code in this repository is licensed under [GNU General Public License v3.0](LICENSE). Model weights are released under [Open RAIL++-M License](https://huggingface.co/bwittmann/vesselFM/blob/main/LICENSE) and are restricted to research and non-commercial use only. Model use must comply with potential licenses, regulations, and restrictions arising from the use of named data sets during model training.
