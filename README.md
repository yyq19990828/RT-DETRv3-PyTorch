# DETR-series

[![CI](https://github.com/yyq19990828/DETR-series/actions/workflows/ci.yml/badge.svg)](https://github.com/yyq19990828/DETR-series/actions/workflows/ci.yml)
[![Docs](https://github.com/yyq19990828/DETR-series/actions/workflows/docs.yml/badge.svg)](https://github.com/yyq19990828/DETR-series/actions/workflows/docs.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.12-blue.svg)](pyproject.toml)

**English** | [简体中文](README.zh-CN.md)

DETR-series is a unified PyTorch collection of real-time DETR object detectors. Six model families with **30 COCO variants** run on a single training, evaluation, inference, checkpoint, and deployment runtime: write one config, then train, evaluate, or export any family through the same CLI.

The repository grew out of a PaddlePaddle-to-PyTorch migration of RT-DETRv3. The Python package and PyPI project name are `detrs`, and every workflow is provided by subcommands of a single `detrs` entry point (`train` / `eval` / `infer` / `export` / `convert` / `models`). The official Paddle implementation is kept as a read-only reference submodule; the core PyTorch runtime never imports Paddle.

**Highlights**

- **Unified runtime**: all six families share the data pipeline, two-stage training protocol, EMA, checkpoint resume, the `bbox`/`bbox_num` inference contract, and the ONNX/TorchScript export boundary.
- **Evidence-driven acceptance**: every official checkpoint records its upstream revision, SHA-256, key mapping, and numerical-alignment evidence; full val2017 results are checked against published APs under pre-registered tolerances.
- **Deployment friendly**: inference runs no additional NMS; ONNX (opset 17, fixed height/width, dynamic batch) and TorchScript exports are verified value-by-value; DEIMv2 covers every size tier from 50.3M down to 0.5M parameters for GPU, edge, and mobile deployment.

See the [model documentation](docs/models/README.md) for the current support matrix; RT-DETRv3 standard schedules, multi-seed runs, and R34/R50 long training remain deferred — see the [roadmap](ROADMAP.md).

## Model overview

COCO val2017 bbox AP values are the officially published numbers; this repository's measured results and acceptance evidence live in each model document.

| Family | Paper | Variants | Official COCO AP | Docs |
|---|---|---|---|---|
| RT-DETRv3 | RT-DETRv3 (2024) | R18/R34/R50 | 48.1 (R18) | [rtdetrv3](docs/models/rtdetrv3/README.md) |
| D-FINE | D-FINE (2024) | N/S/M/L/X | 42.8 – 55.8 | [dfine](docs/models/dfine/README.md) |
| DEIM-D-FINE | DEIM (CVPR 2025) | N/S/M/L/X | 43.0 – 56.5 | [deim](docs/models/deim/README.md) |
| DEIM-RT-DETRv2 | DEIM (CVPR 2025) | S/M/M*/L/X | 49.0 – 55.5 | [deim](docs/models/deim/README.md) |
| RT-DETRv4 | RT-DETRv4 (2025) | S/M/L/X | 49.8 – 57.0 | [rtdetrv4](docs/models/rtdetrv4/README.md) |
| DEIMv2 | DEIMv2 (2025) | X/L/M/S; N/Pico/Femto/Atto | 23.8 – 57.8 | [deimv2](docs/models/deimv2/README.md) |

The runtime family names, config directories, and checkpoint manifests are listed below. All 27 COCO variants outside RT-DETRv3 have completed official-checkpoint, full val2017, reduced-training resume, and deployment acceptance; their weights are hosted upstream and do not belong to the initial `v0.1.0` release.

| Runtime family | Variants | Configs | Checkpoint manifest |
|---|---|---|---|
| `rtdetrv3` | R18/R34/R50 | [`configs/rtdetrv3`](configs/rtdetrv3/) | [`rtdetrv3_coco.yml`](configs/checkpoints/rtdetrv3_coco.yml) |
| `dfine` | N/S/M/L/X | [`configs/dfine`](configs/dfine/) | [`dfine_coco.yml`](configs/checkpoints/dfine_coco.yml) |
| `deim-dfine` | N/S/M/L/X | [`configs/deim/dfine`](configs/deim/dfine/) | [`deim_dfine_coco.yml`](configs/checkpoints/deim_dfine_coco.yml) |
| `deim-rtdetrv2` | S/M/M*/L/X | [`configs/deim/rtdetrv2`](configs/deim/rtdetrv2/) | [`deim_rtdetrv2_coco.yml`](configs/checkpoints/deim_rtdetrv2_coco.yml) |
| `rtdetrv4` | S/M/L/X | [`configs/rtdetrv4`](configs/rtdetrv4/) | [`rtdetrv4_coco.yml`](configs/checkpoints/rtdetrv4_coco.yml) |
| `deimv2` | X/L/M/S; N/Pico/Femto/Atto | [`configs/deimv2`](configs/deimv2/) | [`deimv2_coco.yml`](configs/checkpoints/deimv2_coco.yml) |

## Quick start

The project supports Python 3.9–3.12 and manages environments with uv 0.11.29 through 0.12.x. The default lockfile targets PyTorch CUDA 12.1 on Linux x86_64 or Windows amd64; other platforms need a matching PyTorch index.

```bash
git clone --recurse-submodules https://github.com/yyq19990828/DETR-series.git
cd DETR-series
uv sync

# Inspect model and checkpoint status
uv run detrs models list

# Run inference with a prepared checkpoint
uv run detrs infer \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --infer-img path/to/image.jpg \
  --output-dir output/infer \
  --save-results
```

Full installation profiles, checkpoint acquisition, and train/eval/infer/convert/export examples are in the user guide (Chinese): [使用指南](docs/guides/README.md).

## Common workflows

```bash
# Train
uv run detrs train \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --seed 0

# Evaluate
uv run detrs eval \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth

# Export ONNX and TorchScript
uv run --extra export detrs export \
  -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
  --checkpoint path/to/model.pth \
  --format both \
  --output-dir output/export
```

Every workflow is provided by the single `detrs` entry point with the subcommands `train` / `eval` / `infer` / `export` / `convert` / `models`. Inference runs no additional NMS; ONNX/TorchScript use dynamic batch with a fixed exported height/width, so changing the spatial size requires re-export. See the user guide (Chinese) and the [RT-DETRv3 CLI contract](docs/models/rtdetrv3/cli-and-export.md) for parameters and deployment boundaries.

## Install profiles

| Purpose | Command | Paddle |
|---|---|---|
| Core training and inference | `uv sync` | Not installed |
| Non-Paddle tests | `uv sync --extra test` | Not installed |
| Development, Paddle conversion, numerical alignment | `uv sync --extra dev` | Installed |
| ONNX CPU / CUDA | `uv sync --extra export` / `uv sync --extra export-gpu` | Not installed |
| Ruff and Mypy | `uv sync --extra quality` | Not installed |
| RT-DETRv4 DINOv3 teacher training | `uv sync --extra teacher` | Not installed |

Paddle and migration-only dependencies belong exclusively to the `dev` extra. The DINOv3 teacher is only constructed for RT-DETRv4 training; student eval, infer, and export need no teacher checkout, gated weights, or the `teacher` extra. Platform and conflict boundaries of each extra are in the user guide (Chinese): [使用指南](docs/guides/README.md).

## Testing and quality

```bash
# Tests that do not depend on Paddle
uv run --extra test pytest -m "not paddle"

# Including Paddle alignment tests
uv run --extra dev pytest

# Ruff format, Ruff lint, and Mypy
uv run --extra quality python scripts/check_quality.py
```

The test matrix, coverage, documentation gates, and release checks are described in the developer guide (Chinese): [开发者指南](docs/development/README.md).

## Documentation

All in-depth documentation is written in Chinese; code identifiers, commands, and tables are language-neutral.

- [Documentation site](https://yyq19990828.github.io/DETR-series/): online docs with quick start, user guide, model documentation, and an auto-generated API reference.
- [User guide](docs/guides/README.md): installation, model assets, training, evaluation, inference, conversion, and export.
- [Model documentation](docs/models/README.md): current support contracts, per-variant metrics, and validation reports.
- [Migration notes](docs/migrations/README.md): framework semantics, configuration, weight conversion, training validation, and troubleshooting.
- [Developer guide](docs/development/README.md): testing, quality, releases, and documentation maintenance.
- [Execution plans](docs/plans/README.md): active, deferred, and completed plan entries.
- [Archive](docs/archive/README.md): dated plans, reports, papers, and machine-readable evidence.

## Citation

The BibTeX entries below are taken verbatim from each upstream repository's official Citation section. If this repository helps your work, please cite both the corresponding model paper and this repository.

**RT-DETRv3** (upstream [clxia12/RT-DETRv3](https://github.com/clxia12/RT-DETRv3))

```bibtex
@article{wang2024rt,
  title={RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Supervision},
  author={Wang, Shuo and Xia, Chunlong and Lv, Feng and Shi, Yifeng},
  journal={arXiv preprint arXiv:2409.08475},
  year={2024}
}
```

**D-FINE** (upstream [Peterande/D-FINE](https://github.com/Peterande/D-FINE))

```bibtex
@misc{peng2024dfine,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**DEIM** (upstream [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM); used by both the DEIM-D-FINE and DEIM-RT-DETRv2 profiles)

```bibtex
@misc{huang2024deim,
      title={DEIM: DETR with Improved Matching for Fast Convergence},
      author={Shihua, Huang and Zhichao, Lu and Xiaodong, Cun and Yongjun, Yu and Xiao, Zhou and Xi, Shen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```

**RT-DETRv2** (upstream [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR); decoder source of the DEIM-RT-DETRv2 profile)

```bibtex
@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140},
}
```

**RT-DETRv4** (upstream [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4))

```bibtex
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Zijun Liao and Yian Zhao and Xin Shan and Yan Yan and Chang Liu and Lei Lu and Xiangyang Ji and Jie Chen},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
```

**DEIMv2** (upstream [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2))

```bibtex
@article{huang2025deimv2,
  title={Real-Time Object Detection Meets DINOv3},
  author={Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal={arXiv},
  year={2025}
}
```

The [DINOv3](https://github.com/facebookresearch/dinov3) forward code used by the DEIMv2 backbone is vendored under its [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/346f38fee679c56a6888f91c51670fae61d364e0/LICENSE.md); licensing and attribution boundaries are in the [NOTICE](NOTICE).

## Release status

The initial release [`v0.1.0`](https://github.com/yyq19990828/DETR-series/releases/tag/v0.1.0) targets RT-DETRv3 and ships a wheel, sdist, R18/R34/R50 detection weights, the R18-vd backbone weights, mapping reports, and `SHA256SUMS`. All 11 pinned-tag assets have completed public download and checksum verification; the historical environment, commands, and limitations are recorded in the release validation report (Chinese): [发布验证报告](docs/archive/rtdetrv3-v0.1.0/reports/release-validation.md).

Later releases must be rebuilt with fresh checksums; the current worktree or historical build records do not constitute releasable assets.

## Repository layout

```text
.
├── src/detrs/                # Installable PyTorch package
├── configs/                  # Model and checkpoint configurations
├── tests/                    # Unit, integration, and numerical tests
├── tools/dev/                # Development-time alignment and validation tools
├── docs/                     # User, model, migration, plan, and archive docs
├── ROADMAP.md                # Unfinished migration outline
└── third-party/
    └── RT-DETRv3-paddle/     # Read-only Paddle reference submodule
```
