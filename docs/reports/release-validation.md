# 发布候选验证报告

- 状态：`release candidate validated; artifacts not published`
- 验证日期：`2026-07-19`
- 发布加固基线提交：`dc09cd8`

本报告证明当前源码包和模型清单具备发布条件，不表示 Git tag、GitHub Release、PyPI 或模型 Hub 已经发布。发布前还需要维护者确认版本号和对外上传。

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

下表保留发布加固基线工作树构建时的候选产物快照。归档容器可带构建时间，后续源码与文档已经变化，因此这些 SHA-256 不能用于本轮或最终发布；最终发布必须从 tag 重建并对实际上传文件重新计算。

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| `rtdetrv3_pytorch-0.1.0-py3-none-any.whl` | `325,179` | `dd40257ccd095936ac1912a2d1ef884bc26d67168d2b22c817b8310d0d38233d` |
| `rtdetrv3_pytorch-0.1.0.tar.gz` | `696,099` | `549ed3f2b04088d5e0744f8903a6a3438dae1323338f2aafd05374020d40f557` |

## 模型产物

| 模型 | 转换文件大小 | 发布 SHA-256 候选 |
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

`rtdetrv3-models list/verify/download` 已完成 manifest 解析、四个转换产物的本地校验、HTTPS 限制、临时文件校验后原子替换、不匹配既有文件保护和未发布显式失败。三个检测权重使用 `r18/r34/r50`，训练初始化权重使用 `r18-backbone`；alias 由 manifest 声明，并由 CLI 与发布检查共同拒绝重复或非法值。当前四个转换产物都标记 `unpublished` 且没有 URL；这是如实状态，不是下载端到端证据。

## 尚未完成的发布动作

2026-07-19 实际查询时，远程仓库尚无 tag 和 GitHub Release；因此本地扁平目录预演不能替代真实公开 URL 证据。

- [ ] 确认 `v0.1.0` 版本号和发布说明，创建签名或受保护 tag。
- [ ] 从 tag 的干净工作树重建 wheel/sdist，运行 `scripts/check_release.py --require-models`。
- [ ] 对实际上传的四个 `.pth`、四份 `.mapping.json`、wheel 和 sdist 运行 `scripts/check_release.py --write-sha256sums`，生成 `SHA256SUMS`。
- [ ] 上传 GitHub Release；如启用 Hub 镜像，同步 model card、license、config 和同一批权重。
- [ ] 在空目录中从公开 URL 下载全部 asset，运行 `scripts/check_release.py --verify-release-dir` 与 Infer/Eval smoke，再把固定下载 URL 写入 manifest 和 README。

直接维护范围的 90% 覆盖率目标已有托管证据，R18 CUDA/COCO 端到端 DataLoader/profile 已有本机证据；M4 完整长训仍按维护者决策 deferred。本报告不会把未验证项改写为已验证。
