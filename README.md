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
uv pip install -e .

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

## 🟢 TopBrain 训练与微调
VesselFM 当前微调代码读取的是目录结构（`train/val/test` 下每个样本目录内有 `img.*` + `mask.*`），不是 CSV 驱动。
`scripts/prepare_topbrain_finetune_data.py` 会自动把 `imagesTr/labelsTr(/imagesTs/labelsTs)` 转成该结构，并额外生成一个绝对路径清单 `manifest.csv` 便于追踪。

1) 准备微调数据（生成 `vesselfm_finetune` + `manifest.csv`）：

```bash
python ./scripts/prepare_topbrain_finetune_data.py --force
```

2) 开始微调（自动读取 `.env`，模型保存到 `checkpoints/`）：

```bash
bash ./scripts/train_topbrain_finetune.sh
```

3) 用微调模型推理（输出保存到 `outputs/`）：

```bash
bash ./scripts/test_topbrain_inference.sh
```

脚本默认路径：
- 微调数据目录：`${TOPBRAIN_FINETUNE_DIR}`（默认 `./data/datasets/topBrain-2025/vesselfm_finetune`）
- 预训练权重：`${PRETRAIN_CKPT}`（默认 `./modelsweights/vesselFM_base.pt`）
- 训练 checkpoint：`${CHECKPOINTS_DIR}`（默认 `./checkpoints`）
- 推理输出：`${OUTPUTS_DIR}`（默认 `./outputs`）

可选环境变量：
- `NUM_SHOTS`：`all` 或整数（few-shot）
- `BATCH_SIZE`、`INPUT_SIZE`：按显存调参
- `CKPT_PATH`：测试脚本指定要使用的 checkpoint（不设则自动选 `checkpoints/` 下最新 `.ckpt`）
- `TRAIN_DEVICES`：训练使用的设备，可写 `0` 或 `0,1`（多卡）
- `INFER_DEVICE`：推理使用单卡，可写 `0` 或 `cuda:0`
- 脚本会把 `.env` 当作默认值；你在命令前临时设置的环境变量会优先级更高。

多卡训练与 CUDA 设备可见性：

```bash
# 例1：使用物理 GPU 2,3 做训练（进程内映射为 cuda:0,cuda:1）
export CUDA_VISIBLE_DEVICES=2,3
TRAIN_DEVICES=0,1 bash ./scripts/train_topbrain_finetune.sh

# 例2：单卡训练/推理
export CUDA_VISIBLE_DEVICES=0
TRAIN_DEVICES=0 bash ./scripts/train_topbrain_finetune.sh
INFER_DEVICE=0 bash ./scripts/test_topbrain_inference.sh
```

注意：环境变量名是 `CUDA_VISIBLE_DEVICES`（不是 `CUDA_VISABLE_DEVICE`）。

常见报错排查（`__nvJitLinkComplete_12_4`）：
- 现象：
  - `ImportError: ... libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4`
- 原因：
  - 运行时加载到了系统/conda 的旧 `libnvJitLink.so.12`，与 PyTorch wheel 里的 `cusparse` 版本不匹配。
- 处理：
  - 使用本仓库 `scripts/train_topbrain_finetune.sh` / `scripts/test_topbrain_inference.sh`（脚本已自动把 `.venv` 的 `nvidia/*/lib` 注入 `LD_LIBRARY_PATH`）。
  - 建议退出 `base` 后再运行，仅保留项目虚拟环境。

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
