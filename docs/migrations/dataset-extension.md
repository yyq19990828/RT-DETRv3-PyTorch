# 数据集与指标扩展模式

- 日期:2026-08-15(基于多文件夹数据集支持计划,见 [2026-08-15-multi-folder-datasets](../plans/2026-08-15-multi-folder-datasets.md);状态:已通过测试)

新增数据集格式或评估指标时可复用的仓库契约与接入点。以下结论均"已通过测试"(tests/unit/data、tests/unit/metrics)。

## 数据集类注册三要素

1. 类定义在 `src/detrs/data/source/<name>.py`,带 `@register` 与 `@serializable` 装饰器。
2. 必须在 `src/detrs/data/source/__init__.py` 导入并加入 `__all__`——注册发生在导入时,漏掉会导致 `create()` 报"not registered"。
3. 构造函数参数必须显式声明(schema strict 校验,无 `**kwargs` 时多余 YAML 键直接报错)。

## roidb 记录 schema(与 COCODataSet 一致)

下游 Reader/transform 依赖以下结构,新数据集类必须产出同构记录:

- 必有:`im_file`(str)、`im_id`(`np.array([id])`)、`h`/`w`(float,标注尺寸)。
- gt 键按 `data_fields` 过滤后加入:`gt_bbox` `(N,4)` float32 绝对像素 xyxy、`gt_class` `(N,1)` int32(0-based 连续类别)、`is_crowd` `(N,1)` int32(YOLO/LVIS 恒 0)。
- 空样本(allow_empty)也带空数组 gt(`(0,4)`/`(0,1)`),不能缺键。
- `curr_iter`/`curr_epoch` 由 `DetDataset.__getitem__` 注入,数据集类不要自己写。

## 多文件夹/多标注文件合并语义(已验证)

- `anno_path` 支持 str 或 `List[str|dict]`:str 项共享全局 `image_dir`,dict 项 `{anno_path, image_dir}` 覆盖(跨根目录场景)。
- 合并规则:类别表(cat id + name 序列)必须逐文件一致,否则 `ValueError`;`im_id` 按"上一份 max_img_id+1"逐份偏移防撞号;`sample_num` 为全局配额,逐份扣减。
- 列表 `anno_path` 仅限训练:评估/推理链路(`get_anno`、trainer `_build_validation`、`detrs eval` CLI)遇到列表抛带指引的 `ValueError`。
- 平行列表接口参考 `YOLODataSet`:`image_dir`/`label_dir` 接受单值或等长列表,按索引配对。

## 无标注文件格式的图片尺寸

YOLO 等标签格式不含图片尺寸,用 `PIL.Image.open(path).size` 只读头部获取(`with` 块内),不要解码全图;缺标签且 `allow_empty=True` 时同样需要读头以填充 `h`/`w`。

## 内存 COCO GT 评估(不落盘标注文件)

- `cocoapi_eval(jsonfile, style, coco_gt=<COCO 对象>)` 已支持传入内存 GT(`src/detrs/metrics/coco_utils.py`)。
- 从 roidbs 构建 GT dict:`images`(id/file_name/width/height)、`annotations`(bbox 转 xywh + area + iscrowd=0)、`categories`(id/name)。
- **坑**:pycocotools `loadRes` 要求 GT dict 含 `info` 与 `licenses` 键,即使为空(`KeyError: 'info'`),否则崩溃。
- `COCO()` 无参构造后赋值 `coco_gt.dataset = {...}` + `createIndex()` 即可,无需临时文件。
- 预测侧复用 `get_infer_results`:模型输出 `bbox` 每行 `[cls_id, score, x1, y1, x2, y2]` + `bbox_num` + `im_id`,映射由 `clsid2catid` 控制(YOLO 用恒等映射)。

## 指标接入点(`metric: <NAME>` 配置)

- 训练中验证:`src/detrs/engine/trainer.py` `_build_validation`(reader 构建后、roidbs 就绪)。
- 独立评估:`src/detrs/cli/eval.py` main 中按 `cfg.get("metric")` 分支构造指标;`_configure_dataset` 对非 COCO 覆盖语义要先短路。
- 推理类名:`src/detrs/data/source/category.py` `get_categories` 需为新 metric 类型加分支(YOLO 要求类名 txt,无默认类别表)。
