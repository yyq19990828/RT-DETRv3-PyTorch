# CLI 参考

本页由 [`scripts/generate_cli_reference.py`](../../scripts/generate_cli_reference.py)
从 `detrs` 各命令的 `--help` 输出自动生成,**请勿手改**。参数变更后重新运行
`uv run python scripts/generate_cli_reference.py` 并提交结果;CI 会对生成结果做
一致性检查。

## detrs

```text
usage: detrs [-h] {train,eval,infer,export,convert,models} ...

DETR-series PyTorch toolbox: train, evaluate, run inference on, export, and convert detectors, and
manage released checkpoints.

positional arguments:
  {train,eval,infer,export,convert,models}
    train               Train a detector from a YAML config.
    eval                Evaluate a checkpoint on COCO-style data.
    infer               Run inference on an image or a directory.
    export              Export a detector for deployment (ONNX/TorchScript).
    convert             Convert PaddlePaddle weights to PyTorch.
    models              List, verify, or download released checkpoints.

options:
  -h, --help            show this help message and exit
```

## detrs train

```text
usage: detrs train [-h] -c CONFIG [-o [OPT ...]] [--eval] [-r RESUME] [--slim_config SLIM_CONFIG]
                   [--enable_ce ENABLE_CE] [--seed SEED] [--amp] [--ddp]
                   [--use_tensorboard USE_TENSORBOARD] [--tensorboard_log_dir TENSORBOARD_LOG_DIR]
                   [--use_wandb USE_WANDB] [--save_prediction_only]
                   [--profiler_options PROFILER_OPTIONS] [--save_proposals]
                   [--proposals_path PROPOSALS_PATH] [--local_rank LOCAL_RANK]

options:
  -h, --help            show this help message and exit
  -c, --config CONFIG   configuration file to use
  -o, --opt [OPT ...]   set configuration options
  --eval                Whether to perform evaluation in train
  -r, --resume RESUME   weights path for resume
  --slim_config SLIM_CONFIG
                        Configuration file of slim method.
  --enable_ce ENABLE_CE
                        If set True, enable continuous evaluation job.This flag is only used for
                        internal test.
  --seed SEED           Base random seed for reproducible training.
  --amp                 Enable auto mixed precision training.
  --ddp                 Use DistributedDataParallel for multi-GPU training
  --use_tensorboard USE_TENSORBOARD
                        whether to record the data to TensorBoard.
  --tensorboard_log_dir TENSORBOARD_LOG_DIR
                        TensorBoard logging directory.
  --use_wandb USE_WANDB
                        whether to record the data to wandb.
  --save_prediction_only
                        Whether to save the evaluation results only
  --profiler_options PROFILER_OPTIONS
                        The option of profiler, which should be in format
                        "key1=value1;key2=value2;key3=value3".please see detrs/utils/profiler.py
                        for detail.
  --save_proposals      Whether to save the train proposals
  --proposals_path PROPOSALS_PATH
                        Train proposals directory
  --local_rank LOCAL_RANK
                        Local rank for distributed training
```

## detrs eval

```text
usage: detrs eval [-h] -c CONFIG --checkpoint CHECKPOINT [--anno-file ANNO_FILE]
                  [--image-dir IMAGE_DIR] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
                  [--output-dir OUTPUT_DIR] [--use-ema] [--device DEVICE] [-o [OVERRIDE ...]]

RT-DETRv3 COCO evaluation

options:
  -h, --help            show this help message and exit
  -c, --config CONFIG
  --checkpoint CHECKPOINT
  --anno-file, --anno_file ANNO_FILE
  --image-dir, --image_dir IMAGE_DIR
  --batch-size, --batch_size BATCH_SIZE
  --num-workers, --num_workers NUM_WORKERS
  --output-dir, --output_dir OUTPUT_DIR
                        Keep COCO prediction files in this directory.
  --use-ema, --use_ema  Evaluate the EMA state stored in a training checkpoint.
  --device DEVICE
  -o, --override [OVERRIDE ...]
```

## detrs infer

```text
usage: detrs infer [-h] -c CONFIG
                   (--checkpoint CHECKPOINT | --onnx-model ONNX_MODEL | --torchscript-model TORCHSCRIPT_MODEL)
                   (--infer-img INFER_IMG | --infer-dir INFER_DIR) [--output-dir OUTPUT_DIR]
                   [--save-results] [--threshold THRESHOLD] [--batch-size BATCH_SIZE]
                   [--imgsz IMGSZ] [--anno-file ANNO_FILE] [--use-ema] [--device DEVICE]
                   [-o [OVERRIDE ...]]

RT-DETRv3 inference

options:
  -h, --help            show this help message and exit
  -c, --config CONFIG
  --checkpoint CHECKPOINT
  --onnx-model ONNX_MODEL
                        Run a tensor-only ONNX export with ONNX Runtime CPU or CUDA.
  --torchscript-model TORCHSCRIPT_MODEL
                        Run a tensor-only traced TorchScript export on a PyTorch device.
  --infer-img, --infer_img INFER_IMG
                        Path to one image.
  --infer-dir, --infer_dir INFER_DIR
                        Directory containing images (non-recursive).
  --output-dir, --output_dir OUTPUT_DIR
  --save-results, --save_results
                        Save thresholded detections to detections.json.
  --threshold, --draw-threshold THRESHOLD
                        Minimum score used for visualization and saved results.
  --batch-size, --batch_size BATCH_SIZE
  --imgsz IMGSZ         Override the square Resize target in TestReader.
  --anno-file ANNO_FILE
                        Optional annotation JSON/TXT used for category names.
  --use-ema             Use EMA weights from a training checkpoint.
  --device DEVICE
  -o, --override [OVERRIDE ...]
```

## detrs export

```text
usage: detrs export [-h] -c CONFIG --checkpoint CHECKPOINT [--format {onnx,torchscript,both}]
                    [--output-dir OUTPUT_DIR] [--input-size HEIGHT WIDTH]
                    [--batch-size BATCH_SIZE] [--opset-version OPSET_VERSION] [--fixed-batch]
                    [--use-ema] [--force] [--no-verify] [-o [OVERRIDE ...]]

Export a detector for deployment

options:
  -h, --help            show this help message and exit
  -c, --config CONFIG
  --checkpoint CHECKPOINT
  --format {onnx,torchscript,both}
  --output-dir OUTPUT_DIR
  --input-size HEIGHT WIDTH
                        Fixed spatial size; defaults to TestReader.inputs_def.image_shape.
  --batch-size BATCH_SIZE
  --opset-version OPSET_VERSION
  --fixed-batch         Do not mark the ONNX batch axes as dynamic.
  --use-ema
  --force
  --no-verify           Skip ONNX Runtime/TorchScript output comparison.
  -o, --override [OVERRIDE ...]
```

## detrs convert

```text
usage: detrs convert [-h] --input INPUT --output OUTPUT [--config CONFIG]
                     [--manual-mapping MANUAL_MAPPING] [--save-mapping SAVE_MAPPING]
                     [--strict | --permissive] [--no-validate] [--force] [--batch]
                     [--summary SUMMARY] [--memory-efficient]
                     [--parameter-batch-size PARAMETER_BATCH_SIZE]
                     [--log-level {DEBUG,INFO,WARNING,ERROR}] [--quiet] [--version]

Convert RT-DETRv3 model weights from PaddlePaddle to PyTorch format

options:
  -h, --help            show this help message and exit
  --input, -i INPUT     Source PaddlePaddle checkpoint, or a directory/glob when --batch is set
  --output, -o OUTPUT   Output .pth file, or output directory when --batch is set
  --config, -c CONFIG   PyTorch model config used to build the target state_dict (required unless
                        --no-validate is set)
  --manual-mapping, -m MANUAL_MAPPING
                        Path to JSON file with manual parameter name mapping overrides
  --save-mapping, -s SAVE_MAPPING
                        Export generated parameter name mapping to JSON file
  --strict              Fail on tensor conversion errors and shape mismatches
  --permissive          Enable permissive mode (skip mismatched parameters, continue conversion)
  --no-validate         Skip shape validation against target model
  --force, -f           Overwrite existing output files without confirmation
  --batch               Convert every discovered input independently and continue on failures
  --summary SUMMARY     Write a JSON batch summary (only valid with --batch)
  --memory-efficient    Release source tensors incrementally during conversion
  --parameter-batch-size PARAMETER_BATCH_SIZE
                        Source tensors released between garbage-collection passes
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Set logging verbosity level
  --quiet, -q           Suppress all output except errors
  --version             show program's version number and exit
```

## detrs models

```text
usage: detrs models [-h] [--family {rtdetrv3,dfine,deim-dfine,deim-rtdetrv2,rtdetrv4,deimv2}]
                    [--manifest MANIFEST]
                    {list,verify,download} ...

List, verify, or download PyTorch model weights.

positional arguments:
  {list,verify,download}
    list                List known model artifacts
    verify              Verify a local model against the manifest
    download            Download and verify a published model

options:
  -h, --help            show this help message and exit
  --family {rtdetrv3,dfine,deim-dfine,deim-rtdetrv2,rtdetrv4,deimv2}
                        Model family; defaults to rtdetrv3.
  --manifest MANIFEST   Checkpoint manifest; defaults to the repository or packaged manifest.
```

