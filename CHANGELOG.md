# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Chinese documentation site for GitHub Pages (MkDocs Material): quick start,
  curated user/model/migration navigation, and mkdocstrings-generated API
  reference for every `detrs` subpackage. The new `Docs` workflow builds with
  `uv sync --extra docs` and `mkdocs build --strict`, and `scripts/docs_hooks.py`
  rewrites docs links that escape `docs/` into absolute GitHub URLs at build
  time so existing documents remain unchanged.
- Docstring coverage for the entire registered (YAML-facing) API surface:
  all 68 registered components now document their purpose, and constructor
  arguments are described on the base class of each wrapper hierarchy,
  turning the site's API reference into a usable configuration reference.
  Docstring-only change with no behavioral impact.
- Beautified CLI and training output with rich: model listing renders a
  styled table, inference/export/train print structured banners, training
  shows a live progress bar, and evaluation shows a progress bar plus a
  metrics table on interactive terminals. Piped or CI output automatically
  degrades to plain text; JSON outputs and the Paddle-style log line for
  non-terminal runs are unchanged.
- Rendered logger console output through rich (`RichHandler` on the shared
  console): colored levels, `file:line` source paths, rich tracebacks, and
  coordinated interleaving with live progress bars. Log files keep the
  original Paddle-style format and piped/CI output stays plain text.
- Relayed third-party pycocotools progress prints (annotation loading,
  `Running per image evaluation...`) through the structured logger via
  `detrs.utils.stdio.relay_prints`, so evaluation output shares the rich log
  format and shows each line's third-party origin (`coco.py:79`). Lines are
  relayed as they are printed, and the native multi-column AP summary table
  is left untouched. Dataset-build chatter now goes to module loggers that
  have no console handler, so it is silent instead of interleaved bare text.
- Replaced deprecated `torch.cuda.amp` usage (trainer `GradScaler`, matcher
  autocast) with the `torch.amp` equivalents, silencing FutureWarnings at
  trainer startup and during matcher forward.
- Beautified `--help` output with rich-argparse. A custom formatter keeps
  argparse's native lowercase `usage:`/`options:` headings so tests, piped
  output, and the generated CLI reference page stay byte-compatible with the
  historical format.
- Auto-generated CLI reference page: `scripts/generate_cli_reference.py`
  renders the live `--help` output of `detrs` and every subcommand into
  `docs/guides/cli-reference.md` (committed), and the Docs workflow fails
  when the page drifts from the actual parsers.
- Split the single-page user guide into a landing overview plus six topic
  pages (install, model assets, training/evaluation, inference,
  conversion/export, CLI boundaries) with matching site navigation; content
  is moved verbatim and all in-repo links were re-validated.

## [1.0.0] - 2026-08-15

First stable release. The repository has grown from the RT-DETRv3
Paddle-to-PyTorch migration into DETR-series: six real-time DETR families
(30 COCO variants) on a single PyTorch runtime.

### Added

- Five new model families with 27 COCO variants, each shipping configs,
  upstream-hosted checkpoint manifests, full val2017 parity evidence against
  the published APs, reduced train/resume validation, and ONNX/TorchScript
  export acceptance:
  - D-FINE (N/S/M/L/X), official COCO AP 42.8 – 55.8.
  - DEIM-D-FINE (N/S/M/L/X), official COCO AP 43.0 – 56.5.
  - DEIM-RT-DETRv2 (S/M/M*/L/X), official COCO AP 49.0 – 55.5.
  - RT-DETRv4 (S/M/L/X), official COCO AP 49.8 – 57.0.
  - DEIMv2 (X/L/M/S; N/Pico/Femto/Atto), official COCO AP 23.8 – 57.8,
    spanning 50.3M down to 0.5M parameters for GPU, edge, and mobile targets.
- YOLO-format data support end to end: `YOLODataSet` (images/labels folders)
  and `YOLOMetric` built on the pycocotools protocol, so no COCO annotation
  file is required for evaluation.
- List-valued `anno_path` for `COCODataSet`/`LVISDataSet`: multi-folder
  datasets merge logically with per-file `image_dir` override, identical
  category-table enforcement, `im_id` offsetting, and a global `sample_num`
  quota.
- ONNX and TorchScript inference validated on CUDA; R34/R50 export device
  matrix closed.

### Changed

- Package and CLI renamed to `detrs` (previously `rtdetrv3_pytorch`), with
  all workflows unified under one entry point:
  `detrs train | eval | infer | export | convert | models`.
- Repository reorganized as DETR-series with a refreshed README and official
  citations; migration plans, validation reports, and benchmark evidence
  archived under `docs/`.

### Fixed

- TorchScript exports are now portable across devices.
- CI now passes on clean checkouts.
- Teacher asset preflight now runs before the Python version gate.

## [0.1.0] - 2026-07-19

First alpha release of the RT-DETRv3 Paddle-to-PyTorch migration.

### Added

- Installable PyTorch package for Python 3.9–3.12 with Train, Eval, Infer,
  Convert, Export, and Models CLIs.
- RT-DETRv3 R18/R34/R50 COCO configurations and converted official
  checkpoints, plus a separate R18-vd ImageNet backbone initialization
  checkpoint for training reproduction.
- Manifest-driven model aliases: `r18`, `r34`, `r50`, and `r18-backbone`.
- Auditable conversion mapping reports and `SHA256SUMS` for every uploaded
  binary.
- ONNX opset 17 and traced TorchScript export paths with documented
  boundaries.
- Verified R18 official shared-checkpoint CPU/FP32 COCO val2017 parity:
  Paddle/PyTorch AP `0.480477300367`/`0.480477134768`, absolute difference
  `1.65599e-7`.

[Unreleased]: https://github.com/yyq19990828/DETR-series/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yyq19990828/DETR-series/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/yyq19990828/DETR-series/releases/tag/v0.1.0
