# 发布候选验证报告

- 状态：`release candidate validated; artifacts not published`
- 验证日期：`2026-07-19`
- 发布加固基线提交：`dc09cd8`

本报告证明当前源码包和模型清单具备发布条件，不表示 Git tag、GitHub Release、PyPI 或模型 Hub 已经发布。发布前还需要维护者确认版本号和对外上传。

## 源码包与 CI

- 根目录已包含 Apache-2.0 `LICENSE` 和上游归属 `NOTICE`；`pyproject.toml` 记录 SPDX license expression、项目 URL 和支持的 Python `3.9–3.12`。
- wheel 包含 `26` 个 YAML 配置和五个 CLI entry point，不包含 checkpoint、Paddle 子模块或 dev-only 依赖；sdists 同样排除 `third-party/` 和 `pretrained_models/`。
- `scripts/check_release.py --require-models` 本机检查 `4` 个 manifest 条目和 `12` 个本地源权重、转换权重及 mapping report，大小、SHA-256 和 mapping 数全部一致。
- 从本地 wheel 在仓库外的全新 UV 环境安装后，五个 CLI `--help`、包内 R18 config 加载和 `22,942,893` 参数的模型构建通过。
- GitHub Actions [run 29678063506](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29678063506) 在 `dc09cd8` 上全部 `6` 个 job 通过：Ruff `167` 文件、Mypy `105` source file；Python 3.9–3.12 均为 `254 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为全包 `47.14%` 和直接维护范围 `67.00%`；wheel smoke 为 `34 passed`。
- 本轮加入可视化脚本后，本地质量门禁为 Ruff `169` 文件、Mypy `106` source file；隐藏 GPU 的默认回归为 `286 passed, 8 skipped`；本地模型清单仍为 `4` 个条目、`12` 个文件/报告全部通过。
- GitHub Actions [run 29678700952](https://github.com/yyq19990828/RT-DETRv3-PyTorch/actions/runs/29678700952) 在可视化提交 `118fd43` 上再次通过全部 `6` 个 job：Ruff `169` 文件、Mypy `106` source file；Python 3.9–3.12 均为 `255 passed, 7 skipped, 17 deselected`；Python 3.12 覆盖率为 `47.14%/67.00%`；发布归档检查和 wheel smoke `34 passed`。

从本轮已验证工作树构建的本地候选产物如下。归档容器可带构建时间，所以这些 SHA-256 是本次观测；最终发布必须对实际上传文件重新计算。

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

三个文件合计 `411,756,783` 字节（约 `392.7 MiB`）。来源 URL、上游 revision、源权重 checksum、转换后 checksum 和 mapping 数以 [`configs/checkpoints/rtdetrv3_coco.yml`](../../configs/checkpoints/rtdetrv3_coco.yml) 为机器可读真值。R18 同图片可视化见[预测对比报告](prediction-visualization.md)，完整 val2017 数值门禁见[精度报告](accuracy-validation.md)。

## 权重托管决策

**建议当前以 GitHub Releases 为版本绑定的主托管，Hugging Face Model Hub 作可选镜像和模型卡入口。**

- GitHub 官方文档说明每个 Release 最多 `1000` 个 asset，单文件小于 `2 GiB`，不限制单个 Release 的总体积或带宽。当前三个权重均远低于单文件限制，而 Release 又直接绑定 Git tag，适合作为 `v0.1.0` 的权威下载源。见 [GitHub Releases 存储与带宽配额](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas)。
- GitHub 普通 Git 会阻止超过 `100 MiB` 的文件，官方也建议用 Releases 分发大型二进制。R34/R50 不应进入 `main` 历史，R18 也应保持相同分发策略。见 [GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#file-size-limits)。
- Hugging Face Model Hub 针对模型二进制优化；官方对 Git-backed model repo 建议单文件小于 `200 GB`，硬上限为 `500 GB`。公开免费存储是 best-effort，但当前约 `393 MiB` 规模很小。见 [Hugging Face Hub 存储限制](https://huggingface.co/docs/hub/main/storage-limits)。
- Hub 的 model card 可结构化记录 license、COCO 数据集、评估结果、预期用途和限制，对模型发现和长期文档更友好。见 [Hugging Face Model Cards](https://huggingface.co/docs/hub/en/model-cards)。

建议的 Release assets 是三个 `.pth`、`SHA256SUMS`、wheel 和 sdist。资产一旦发布不应覆盖同名文件；内容变化则提升 tag，并使用固定 tag 的下载 URL。Hub 镜像同样固定 tag 或 commit revision，不把可变的 `main` URL 写入发布清单。

## 尚未完成的发布动作

- [ ] 确认 `v0.1.0` 版本号和发布说明，创建签名或受保护 tag。
- [ ] 从 tag 的干净工作树重建 wheel/sdist，运行 `scripts/check_release.py --require-models`。
- [ ] 对实际上传的三个 `.pth`、wheel 和 sdist 生成 `SHA256SUMS`。
- [ ] 上传 GitHub Release；如启用 Hub 镜像，同步 model card、license、config 和同一批权重。
- [ ] 从公开 URL 在干净环境下下载，校验 checksum，完成 Infer/Eval smoke，再把固定下载 URL 写入 manifest 和 README。

90% 覆盖率目标、end-to-end DataLoader/profile 和 M4 完整长训仍未完成或已按决策 deferred；本报告不会把它们改写为已验证。
