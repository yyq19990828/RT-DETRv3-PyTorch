# 发布候选验证报告

- 状态：`v0.1.0 published; all 11 assets publicly read back`
- 验证日期：`2026-07-19`
- 发布加固基线提交：`dc09cd8`
- 发布 tag 提交：`c0317ef8475f82b53951ef88b92120b63c08aaa6`

本报告记录 `v0.1.0` annotated tag、GitHub Release 和 11 个公开 asset 的重建、上传及回读证据。wheel 尚未发布到 PyPI，权重也未镜像到 Hugging Face Model Hub；这些未执行项不影响 GitHub Release 的已验证状态。

## 源码包与 CI

- 根目录已包含 Apache-2.0 `LICENSE` 和上游归属 `NOTICE`；`pyproject.toml` 记录 SPDX license expression、项目 URL 和支持的 Python `3.9–3.12`。
- 发布加固基线 wheel 包含 `26` 个 YAML 配置和五个 CLI entry point；本轮新增第六个 `rtdetrv3-models` entry point。wheel 不包含 checkpoint、Paddle 子模块或 dev-only 依赖；sdists 同样排除 `third-party/` 和 `pretrained_models/`。
- `scripts/check_release.py --require-models` 本机检查 `4` 个 manifest 条目和 `12` 个本地源权重、转换权重及 mapping report，大小、SHA-256 和 mapping 数全部一致。
- 从本地 wheel 在仓库外的全新 UV 环境安装后，五个 CLI `--help`、包内 R18 config 加载和 `22,942,893` 参数的模型构建通过。
- GitHub Actions [run 29678063506](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29678063506) 在 `dc09cd8` 上全部 `6` 个 job 通过：Ruff `167` 文件、Mypy `105` source file；Python 3.9–3.12 均为 `254 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为全包 `47.14%` 和直接维护范围 `67.00%`；wheel smoke 为 `34 passed`。
- 本轮加入可视化脚本后，本地质量门禁为 Ruff `169` 文件、Mypy `106` source file；隐藏 GPU 的默认回归为 `286 passed, 8 skipped`；本地模型清单仍为 `4` 个条目、`12` 个文件/报告全部通过。
- GitHub Actions [run 29678700952](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29678700952) 在可视化提交 `118fd43` 上再次通过全部 `6` 个 job：Ruff `169` 文件、Mypy `106` source file；Python 3.9–3.12 均为 `255 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为 `47.14%/67.00%`；发布归档检查和 wheel smoke `34 passed`。
- Models CLI 加固后的本机工作树通过 Ruff `172` 文件、Mypy `107` source file、默认回归 `303 passed, 8 skipped` 和非 Paddle CPU 回归 `272 passed, 5 skipped, 34 deselected`；覆盖率为全包 `48.10%`、直接维护范围 `71.85%`。新 wheel/sdist 通过 `4` 个 manifest 条目和 `12` 个本地文件/报告校验；wheel 在仓库外的全新 UV 环境安装后，当时的 `rtdetrv3-models list --json` 从包内 manifest 正确列出 R18/R34/R50。
- GitHub Actions [run 29679546423](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29679546423) 在 `e1a306b` 上全部 `6` 个 jobs 通过：Ruff `172` 文件、Mypy `107` source file；Python 3.9–3.12 均为 `272 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为 `48.11%/71.85%`。wheel 从 sdist 重建后通过归档检查，安装后的六个 CLI（含 Models）和 `34 passed` smoke 均通过。
- 四产物发布合同批次将 R18-vd backbone 初始化权重加入 Models CLI。`scripts/check_release.py --require-models` 和四次 `rtdetrv3-models verify` 均通过，确认 `4` 个 manifest 条目、`4` 个唯一发布 alias 和 `12` 个本地源/转换/mapping 文件；临时构建的 wheel/sdist 通过归档检查，从解包 wheel 在仓库外读取包内 manifest 后按顺序列出 `r18/r34/r50/r18-backbone`。Ruff `174` 个文件、Mypy `107` 个 source file、14 项发布定向回归和非 Paddle CPU `308 passed, 5 skipped, 34 deselected` 均通过；本段为本机证据，托管证据如下。
- GitHub Actions [run 29683414810](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29683414810) 在 `aa3ccac` 上通过全部 6 个 jobs：Python 3.9–3.12 均为 `308 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为 `50.14%/85.11%`；质量 job 为 Ruff `174` 个文件和 Mypy `107` 个 source file；wheel/sdist 成功构建，包内清单为 `4` 个 manifest 条目和 `4` 个发布产物，安装后六个 CLI 及 wheel smoke `47 passed`。官方 PyTorch 索引上的 Torch 冷下载与各环境准备均在约 21–28 秒完成。
- checksum 生成器本地预检从当前工作树临时构建 wheel/sdist，先通过 `4` 个 manifest 条目和 `12` 个本地源/转换/mapping 文件检查，再对四个 `.pth`、四份 `.mapping.json`、wheel 和 sdist 原子生成 `10` 行 `SHA256SUMS`。将这些文件以扁平 Release 名称链入临时目录后，系统 `sha256sum --check` 独立复核全部通过；临时构建和校验目录随后清理。定向回归为 `7 passed`，质量门禁为 Ruff `174` 个文件和 Mypy `107` 个 source file；隐藏 GPU 的非 Paddle 覆盖率回归为 `312 passed, 5 skipped, 34 deselected`，全包/直接维护范围为 `50.14%/85.11%`。
- GitHub Actions [run 29683881309](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29683881309) 在 checksum 生成器提交 `75d7bab` 上通过全部 `6` 个 jobs：Python 3.9–3.12 均为 `312 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为全包 `6,803/13,567`（`50.14%`）和直接维护范围 `1,720/2,021`（`85.11%`）；质量 job 为 Ruff `174` 个文件和 Mypy `107` 个 source file；wheel/sdist 构建与发布检查通过，wheel smoke 为 `47 passed`。
- 公开回读预演将四份 mapping report 的 size/SHA-256 加入 manifest，再从当前工作树构建 wheel/sdist，与真实四权重/四报告组成 11 个普通文件的扁平目录。新的 `--verify-release-dir` 对完整资产集、10 行 checksum 清单、实际文件摘要、manifest 独立摘要与 wheel/sdist 内容全部校验通过，输出 `11 release assets, 10 checksummed assets`；临时产物已清理。发布定向回归为 `12 passed`，质量门禁为 Ruff `174` 个文件和 Mypy `107` 个 source file；隐藏 GPU 的非 Paddle 覆盖率回归为 `317 passed, 5 skipped, 34 deselected`，全包/直接维护范围为 `50.14%/85.11%`。
- GitHub Actions [run 29684281347](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29684281347) 在公开回读预演提交 `ecabe32` 上通过全部 `6` 个 jobs：Python 3.9–3.12 均为 `317 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为全包 `6,803/13,567`（`50.14%`）和直接维护范围 `1,720/2,021`（`85.11%`）；质量 job 为 Ruff `174` 个文件和 Mypy `107` 个 source file；wheel/sdist 发布检查通过，wheel smoke 为 `47 passed`。
- GitHub Actions [run 29684794341](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29684794341) 在覆盖率收口提交 `48cc134` 上通过全部 `6` 个 jobs：Python 3.9–3.12 均为 `328 passed, 7 skipped, 17 deselected`；Python 3.12 全包/直接维护覆盖率为 `50.99%/90.80%`；质量 job 为 Ruff `174` 个文件和 Mypy `107` 个 source file；wheel/sdist 发布检查通过，wheel smoke 为 `49 passed`。
- GitHub Actions [run 29685452042](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29685452042) 在端到端 runner 提交 `d823edf` 上继续通过全部 `6` 个 jobs：Python 3.9–3.12 均为 `336 passed, 7 skipped, 17 deselected`，覆盖率仍为 `50.99%/90.80%`；Ruff/Mypy、wheel/sdist 发布检查和 `49 passed` wheel smoke 全部通过。
- 原子暂存批次增加 `--stage-release-dir`：在同一父目录的隐藏临时目录复制 10 个输入、生成 checksum 并严格校验，通过后才原子更名；目标已存在时拒绝覆盖，校验失败时清理半成品。当前工作树临时构建的 wheel/sdist 与真实四权重/四报告一次暂存为 `11 release assets, 10 checksummed assets`，随后的独立 `--verify-release-dir` 再次通过；`438 MiB` 临时目录已清理。定向回归为 `16 passed`；全仓质量门禁为 Ruff `174` 个文件、Mypy `107` 个 source file，隐藏 GPU 的非 Paddle 回归为 `340 passed, 5 skipped, 34 deselected`，全包/直接维护范围为 `50.98%/90.80%`。
- GitHub Actions [run 29686126647](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29686126647) 在原子暂存提交 `51847eb` 上通过全部 `6` 个 jobs：Python 3.9–3.12 均为 `340 passed, 7 skipped, 17 deselected`；Python 3.12 全包/直接维护覆盖率为 `50.99%/90.80%`；Ruff `174` 个文件、Mypy `107` 个 source file、wheel/sdist 发布检查和 `49 passed` wheel smoke 全部通过。
- `v0.1.0` 发布元数据批次将 distribution repository/tag 与四个权重的精确 asset URL 固化到 manifest，发布检查不再只验证 HTTPS 前缀，还会拒绝 repository、tag 或文件名不一致。Models/manifest/release 定向回归为 `27 passed`，四个 manifest 条目和 `12` 个真实本地文件/报告校验通过；全仓质量门禁为 Ruff `174` 个文件、Mypy `107` 个 source file，隐藏 GPU 的非 Paddle 回归为 `340 passed, 5 skipped, 34 deselected`，覆盖率为 `50.98%/90.80%`。本段证明 tag 提交的输入元数据可用，不替代后续从 tag 重建和公开 URL 回读。
- 本地 annotated tag `v0.1.0` 指向提交 `c0317ef8475f82b53951ef88b92120b63c08aaa6`。从该 tag 的 detached 独立工作树构建 wheel/sdist，与真实四权重/四 mapping report 原子暂存为 `dist/releases/v0.1.0/` 下恰好 `11` 个普通文件。tag 内发布检查、独立 `--verify-release-dir` 和系统 `sha256sum --check` 全部通过，输出 `11 release assets, 10 checksummed assets`。wheel/sdist 内嵌 manifest 与 tag 源文件的 SHA-256 均为 `7f0dcc2b95cb3c0b7049185c0d9f267f00544fafba6a08cd874714f993001620`；解包 wheel 在仓库外按 `r18/r34/r50/r18-backbone` 读出四个 `published` 固定 URL。临时工作树和构建目录已清理，实际上传目录按维护者要求保留。
- GitHub Actions [run 29687238968](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29687238968) 在发布文档提交 `80d2a80` 上通过全部 `6` 个 job：Python 3.9–3.12 均为 `340 passed, 7 skipped, 17 deselected`；Python 3.12 全包/直接维护覆盖率为 `50.99%/90.80%`；Ruff `174` 个文件、Mypy `107` 个 source file、wheel/sdist 发布检查和 `49 passed` wheel smoke 全部通过。
- [`v0.1.0` GitHub Release](https://github.com/yyq19990828/RT-DETRv3-PyTorch/releases/tag/v0.1.0) 于 `2026-07-19T12:37:30Z` 公开，非草稿、非预发布，包含恰好 `11` 个 asset。GitHub 返回的 size 与 digest 均和 tag 重建目录一致；先通过 Release 下载接口回读，再从无认证固定 URL 回读全部资产，两次均通过严格目录校验和系统 checksum。两个约 `438 MiB` 的临时下载目录均已清理。
- `rtdetrv3-models download r18` 从公开 URL 下载 `92,075,629` 字节的 R18 权重并验证 SHA-256 `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547`。该文件在 CPU/FP32 上完成 COCO `000000000139.jpg` 单图推理，阈值 `0.3` 生成 `30` 条检测和可视化；四图 COCO Eval 处理 `2` 个 batch 并写出 `1,200` 条候选。四图 AP 只证明 Eval 链路可运行，不作为正式精度证据；临时权重、子集和输出已清理。

## v0.1.0 实际上传归档

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| `rtdetrv3_pytorch-0.1.0-py3-none-any.whl` | `330,416` | `ffd8db68649abf132216105c48cdb6cccdf1ed9b0ebf94dc40c3036e70b33ee9` |
| `rtdetrv3_pytorch-0.1.0.tar.gz` | `719,418` | `0d42b6935ecae2d0fe12e2b1be388b3647819822a60feda66ac5fdaff5ef176a` |
| `SHA256SUMS` | `982` | `edaada576b9afe66469d2eda818d15dbab68d0d715d5c6f8c2aa8e0bb3c0f799` |

四个权重和四份 mapping report 的大小/SHA-256 与下方 manifest 表及 `SHA256SUMS` 一致。该目录是实际 GitHub Release 上传输入，不属于 Git 跟踪源码；如任一内容改变，必须更换版本而不能覆盖 `v0.1.0` 同名资产。

下表保留发布加固基线工作树构建时的候选产物快照。归档容器可带构建时间，后续源码与文档已经变化，因此这些 SHA-256 不能用于本轮或最终发布；最终发布必须从 tag 重建并对实际上传文件重新计算。

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| `rtdetrv3_pytorch-0.1.0-py3-none-any.whl` | `325,179` | `dd40257ccd095936ac1912a2d1ef884bc26d67168d2b22c817b8310d0d38233d` |
| `rtdetrv3_pytorch-0.1.0.tar.gz` | `696,099` | `549ed3f2b04088d5e0744f8903a6a3438dae1323338f2aafd05374020d40f557` |

## 模型产物

| 模型 | 转换文件大小 | 已发布 SHA-256 |
|---|---:|---|
| R18 | `92,075,629` | `cb89c589c0a37fbe060554bc26bd662885702c72e3ef0890a54338e9746d0547` |
| R34 | `137,170,947` | `e69207749b37e493596086579f435d5f08e9f058b66322452456053b78a4f272` |
| R50 | `182,510,207` | `5e3e34ac3d3d14f57ebf6100b146b5702f8dface24fbe57cbc993f59381b67f7` |
| R18-vd backbone 初始化权重 | `44,876,108` | `2483b5b00ed2b84192540bbd1bd1768e3e4422c2f8fa1598ae96e0c2d6f64db2` |

三个检测权重合计 `411,756,783` 字节（约 `392.7 MiB`）；加上 R18-vd 训练初始化权重后，四个 Release 权重合计 `456,632,891` 字节（约 `435.5 MiB`）。来源 URL、上游 revision、源/转换权重 checksum、mapping report 的 size/checksum/映射数和 CLI alias 以 [`configs/checkpoints/rtdetrv3_coco.yml`](../../configs/checkpoints/rtdetrv3_coco.yml) 为机器可读真值。三变体同图片可视化见[预测对比报告](prediction-visualization.md)，完整 val2017 数值门禁目前只覆盖 R18，见[精度报告](accuracy-validation.md)。

## 权重托管决策

**建议当前以 GitHub Releases 为版本绑定的主托管，Hugging Face Model Hub 作可选镜像和模型卡入口。**

- GitHub 官方文档说明每个 Release 最多 `1000` 个 asset，单文件小于 `2 GiB`，不限制单个 Release 的总体积或带宽。当前四个权重均远低于单文件限制，而 Release 又直接绑定 Git tag，适合作为 `v0.1.0` 的权威下载源。见 [GitHub Releases 存储与带宽配额](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas)。
- GitHub 普通 Git 会阻止超过 `100 MiB` 的文件，官方也建议用 Releases 分发大型二进制。R34/R50 不应进入 `main` 历史，R18 也应保持相同分发策略。见 [GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#file-size-limits)。
- Hugging Face Model Hub 针对模型二进制优化；官方对 Git-backed model repo 建议单文件小于 `200 GB`，硬上限为 `500 GB`。公开免费存储是 best-effort，但当前约 `435.5 MiB` 规模很小。见 [Hugging Face Hub 存储限制](https://huggingface.co/docs/hub/main/storage-limits)。
- Hub 的 model card 可结构化记录 license、COCO 数据集、评估结果、预期用途和限制，对模型发现和长期文档更友好。见 [Hugging Face Model Cards](https://huggingface.co/docs/hub/en/model-cards)。

建议的 Release assets 是三个检测 `.pth`、一个 R18-vd backbone 初始化 `.pth`、四份对应的 `.mapping.json`、wheel、sdist 和 `SHA256SUMS`，共 11 个 asset。checksum 清单覆盖除自身外的 10 个文件。资产一旦发布不应覆盖同名文件；内容变化则提升 tag，并使用固定 tag 的下载 URL。Hub 镜像同样固定 tag 或 commit revision，不把可变的 `main` URL 写入发布清单。

不建议把权重放入普通 Git、Git LFS 或 PyPI wheel：普通 Git 会增大所有用户的 clone 历史，Git LFS 会让源码 checkout 与模型下载耦合，wheel 则会让只需要代码的安装承担约 `435.5 MiB` 模型体积。GitHub Release 作为唯一权威源，Hub 只作同 checksum 镜像，可以让版本归属和用户下载路径都保持简单。

`rtdetrv3-models list/verify/download` 已完成 manifest 解析、四个转换产物的本地校验、HTTPS 限制、临时文件校验后原子替换和不匹配既有文件保护。三个检测权重使用 `r18/r34/r50`，训练初始化权重使用 `r18-backbone`；alias 由 manifest 声明，并由 CLI 与发布检查共同拒绝重复或非法值。维护者已确认 `v0.1.0`，manifest 的四个产物已写入固定 tag URL；发布检查会要求 URL 严格等于 distribution repository、tag 和权重文件名的组合。公开 Release 已完成 11-asset 整体回读；Models CLI 还使用公开 R18 asset 完成了 checksum、Infer 和 Eval 冒烟，因此 R18 下载是端到端证据。R34、R50 和 R18-vd backbone 的公开 URL 已由整体回读验证，但未分别重复运行模型 CLI 推理。

## 发布动作验收

- [x] 确认 `v0.1.0` 版本号和 asset 文件名，在 tag 所指提交中将四个产物改为 `published` 并写入该 tag 的固定 HTTPS URL。
- [x] 创建本地 annotated tag，从 tag 的干净工作树重建 wheel/sdist，用 `scripts/check_release.py --stage-release-dir` 生成并验证完整 11-asset 目录。
- [x] 上传 GitHub Release，并核对 11 个 asset 的 size 和 GitHub digest。Hugging Face 镜像未启用，仍是可选后续项。
- [x] 在空目录中从公开 URL 下载全部 asset，运行 `scripts/check_release.py --verify-release-dir` 和系统 checksum，并用公开 R18 权重完成 Models/Infer/Eval smoke。

直接维护范围的 90% 覆盖率目标已有托管证据，R18 CUDA/COCO 端到端 DataLoader/profile 已有本机证据，`v0.1.0` GitHub Release 已完成公开回读；M4 完整长训仍按维护者决策 deferred。本报告不会把 PyPI、Hub 镜像、完整 R34/R50 精度或四图 AP 改写为已验证。
