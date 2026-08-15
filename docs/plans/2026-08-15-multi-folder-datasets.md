# 多文件夹数据集支持:YOLO 数据集 + YOLO 评估 + anno_path 列表

- 状态：`completed`
- 创建日期：`2026-08-15`
- 最后更新：`2026-08-15`
- 负责人：`yyq08228`

## 背景

此前 `COCODataSet`/`LVISDataSet` 一个实例只绑定一个标注 json 和一个 `image_dir`,训练入口也没有多数据集合并机制;跨多个物理目录的同格式数据集只能物理合并。仓库不支持 YOLO 格式。本计划实现"逻辑合并、不移动文件"的多文件夹数据加载,并新增 YOLO 格式数据集与其评估。

## 范围

- 包含:`anno_path` 列表(COCO/LVIS)、`YOLODataSet` 多文件夹数据集、`YOLOMetric`(pycocotools 口径)与 trainer/eval/infer 接入、`configs/datasets/yolo_detection.yml` 示例、单元测试、文档沉淀。
- 不包含:ultralytics 数值口径的 mAP 复刻;`anno_path` 列表用于评估/推理(评估链路要求单个 GT 文件,列表场景显式报错);LVIS 依赖安装(相关测试 `importorskip`)。

## 依赖

- 现有 `pycocotools` 核心依赖;`PIL`(已有)用于读取图片头。
- `lvis` 包不在核心/测试依赖中,LVIS 多文件测试自动跳过。

## 目标与非目标

### 目标

- `COCODataSet`/`LVISDataSet` 的 `anno_path` 支持 str 或 `List[str|dict]`(dict 项 `{anno_path, image_dir}` 覆盖全局 `image_dir`),多文件按顺序合并 roidbs,类别表必须一致,`im_id` 跨文件偏移防撞号,`sample_num` 全局截断;单文件行为不变。
- `YOLODataSet`:`image_dir`/`label_dir` 支持平行列表;标签 `class cx cy w h` 归一化转绝对像素 xyxy;缺标签文件默认跳过、`allow_empty` 时为空样本;`label_list` 提供类名并校验 class id 上界;roidb 记录 schema 与 COCO 一致。
- `YOLOMetric`:从已解析的 roidbs 构建内存 COCO GT,经 `cocoapi_eval(coco_gt=...)` 输出 COCO 口径 AP;`metric: YOLO` 时 trainer 训练中验证与 `detrs eval` CLI 自动选用;`get_categories` 支持 `yolo`(类名 txt,恒等 catid)。
- 评估/推理链路遇列表 `anno_path` 报带指引的 `ValueError`。

### 非目标

- 不复刻 ultralytics 的 AP 匹配算法(AGPL-3.0 与本仓库 Apache-2.0 不兼容,且口径与 COCO 指标不可比)。
- 不实现 `EvalDataset` 多文件评估。
- 不改动 VOC/ImageFolder 现有行为。

## 实施步骤

- [x] 重构 `coco.py`/`lvis.py`:`_normalize_anno_entries` + `_parse_single` + 聚合器;验证:`tests/unit/data` 现有 48 例全绿。
- [x] 新增 `tests/unit/data/test_coco_multi_anno.py`(合并/覆盖/类别不一致/`im_id` 偏移/`sample_num` 截断/LVIS skip);验证:5 passed, 1 skipped。
- [x] 新增 `yolo.py` `YOLODataSet` 并在 `source/__init__.py` 注册;验证:`create({name: YOLODataSet, ...})` 构建并解析。
- [x] 新增 `tests/unit/data/test_yolo_dataset.py` 8 例;验证:8 passed。
- [x] 新增 `YOLOMetric` + `metrics/__init__.py` 导出 + `trainer._build_validation`/`cli/eval.py` 的 `metric: YOLO` 分支 + `category.py` yolo 分支;验证:`tests/unit/metrics/test_yolo_metric.py` 5 passed(完美预测 AP50/AP50-95=1.0)。
- [x] 新增 `configs/datasets/yolo_detection.yml` 示例(平行列表 + `label_list` + ImageFolder 测试集)。
- [x] 质量门禁;验证:`uv run pytest tests/unit -q` 756 passed / 19 skipped,`uv run ruff check src tests` 通过,`uv run mypy src/detrs` 无错误。
- [x] 文档与索引更新。

## 风险与回退

- 风险:YOLO 解析需读图片头,大库首次 `parse_dataset` 变慢(仅头部解析,可接受);YOLO mAP 与 ultralytics 官方数值存在口径差异(用户已确认接受);类别一致性断言会拒绝类别不同的组合(报错列出差异文件)。
- 回退:改动均为新增或局部重构,单次提交 revert 即可;单 str `anno_path` 与 COCO 评估路径行为不变,由现有测试守护。

## 验收

- [x] `uv run pytest tests/unit -q`:756 passed, 19 skipped(lvis 等依赖缺失项)。
- [x] `uv run ruff check src tests` 与 `uv run mypy src/detrs`:通过。
- [x] 完美预测下 `YOLOMetric` AP50-95/AP50 = 1.0(`tests/unit/metrics/test_yolo_metric.py`)。
- [x] 文档索引更新,路径均为仓库相对路径。

## 决策记录

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-15 | YOLO 评估采用 pycocotools 口径(GT 转内存 COCO 结构)而非复刻 ultralytics 算法 | ultralytics 为 AGPL-3.0,不能拷入 Apache-2.0 仓库;与仓库现有 COCO 指标同口径可比;用户已确认接受数值口径差异 |
| 2026-08-15 | `anno_path` 列表形态为"字符串或 dict 混合列表" | 纯字符串项共享全局 `image_dir` 覆盖简单场景,dict 项覆盖 `image_dir` 支持跨根目录;用户选定 |
| 2026-08-15 | YOLO 多文件夹接口为 `image_dir`/`label_dir` 平行列表 | 与仓库现有风格(ImageFolder 已支持序列 `image_dir`)一致;用户选定 |
| 2026-08-15 | 多文件合并时 `im_id` 逐份偏移(max+1),类别表取第一份并要求一致 | 防 `image_id` 撞号;类别语义由 YAML `num_classes` 与标签文件共同约束,静默重映射易错 |

## 完成记录

- 实测(Python 3.12.13,uv `.venv`,CPU):`uv run pytest tests/unit -q` → 756 passed, 19 skipped;`uv run ruff check src tests` → All checks passed;`uv run mypy src/detrs` → no issues in 126 files。
- 新增 19 个测试:`test_coco_multi_anno.py`(6,1 skip)、`test_yolo_dataset.py`(8)、`test_yolo_metric.py`(5);适配 `tests/unit/cli/test_eval.py` 的 cfg stub 增加 `get` 方法(新逻辑读取 `cfg.get("metric")`)。
- pycocotools `loadRes` 要求 GT dict 含 `info`/`licenses` 键,内存 GT 已补空值;此结论已沉淀至 `docs/migrations/dataset-extension.md`。
- 偏差:无。后续可选事项:YOLO 评估的 per-class 表格(`classwise`)已随 `cocoapi_eval` 免费获得;`terminaltables` 未安装时该表格会报 ImportError,属既有行为。
