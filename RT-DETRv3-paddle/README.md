English | [简体中文](README_cn.md)

## RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision

:fire::fire:**[WACV 2025 Oral]** The official implementation of the paper "[RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision](https://arxiv.org/pdf/2409.08475)". \
[[`arXiv`](https://arxiv.org/pdf/2409.08475)] 
![image](https://github.com/user-attachments/assets/5910d729-cc44-49f4-b404-b6631576930f)


## Model Zoo on COCO

| Model | Epoch | Backbone  | Input shape | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) |  T4 TensorRT FP16(FPS) | Weight | Config | Log
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|:---|
| RT-DETRv3-R18 | 6x |  ResNet-18 | 640 | 48.1 | 66.2 | 20 | 60 | 217 |[baidu 网盘](https://pan.baidu.com/s/1s7lyT6_fHmczoegQZXdX-w?pwd=54jp)  [google drive](https://drive.google.com/file/d/1zIDOjn1qDccC3TBsDlGQHOjVrehd26bk/view?usp=drive_link)| [config](./configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml) | 
| RT-DETRv3-R34 | 6x |  ResNet-34 | 640 | 49.9 | 67.7 | 31 | 92 | 161 | [baidu 网盘](https://pan.baidu.com/s/1VCg6oqNVF9_ZZdmlhUBgSA?pwd=pi32) [google drive](https://drive.google.com/file/d/12-wqAF8i67eqbocaWPK33d4tFkN2wGi2/view?usp=drive_link)| [config](./configs/rtdetrv3/rtdetrv3_r34vd_6x_coco.yml) | 
| RT-DETRv3-R50 | 6x |  ResNet-50 | 640 | 53.4 | 71.7 | 42 | 136 | 108 | [baidu 网盘](https://pan.baidu.com/s/1DuvrpMIqbU5okoDp16C94g?pwd=wrxy) [google drive](https://drive.google.com/file/d/1wfJE-QgdgqKE0IkiTuoD5HEbZwwZg3sQ/view?usp=drive_link)| [config](./configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml) | 
| RT-DETRv3-R101 | 6x |  ResNet-101 | 640 | 54.6 | 73.1 | 76 | 259 | 74 |  | [config](./configs/rtdetrv3/rtdetrv3_r101vd_6x_coco.yml) | 


**Notes:**
- RT-DETRv3 uses 4 GPUs for training.
- RT-DETRv3 was trained on COCO train2017 and evaluated on val2017.

## Model Zoo on LVIS

| Model | Epoch | Backbone  | Input shape | AP | $AP_{r}$ | $AP_{c}$ | $AP_{f}$ | Weight | Config | Log
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|:---|
| RT-DETRv3-R18 | 6x |  ResNet-18 | 640 | 26.5 | 12.5 | 24.3 | 35.2 |  | [config](./configs/rtdetrv3/rtdetrv3_r18vd_6x_lvis.yml) | 
| RT-DETRv3-R50 | 6x |  ResNet-50 | 640 | 33.9 | 20.2 | 32.5 | 41.5 |  | [config](./configs/rtdetrv3/rtdetrv3_r50vd_6x_lvis.yml) |


## Quick start

<details open>
<summary>Install requirements</summary>

<!-- - PaddlePaddle == 2.4.2 -->
```bash
pip install -r requirements.txt
```

</details>

<details>
<summary>Compile (optional)</summary>

```bash
cd ./ppdet/modeling/transformers/ext_op/

python setup_ms_deformable_attn_op.py install
```
See [details](./ppdet/modeling/transformers/ext_op/)
</details>


<details>
<summary>Data preparation</summary>

- Download and extract COCO 2017 train and val images.
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
- Modify config [`dataset_dir`](configs/datasets/coco_detection.yml)
</details>


<details>
<summary>Training & Evaluation & Testing</summary>

- Training on a Single GPU:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --eval
```

- Training on Multiple GPUs:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --fleet --eval
```

- Evaluation:

```shell
python tools/eval.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv3_r18vd_6x_coco.pdparams
```

- Inference:

```shell
python tools/infer.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv3_r18vd_6x_coco.pdparams \
              --infer_img=./demo/000000570688.jpg
```

</details>


## Deploy

<details open>
<summary>1. Export model </summary>

```shell
python tools/export_model.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv3_r18vd_6x_coco.pdparams trt=True \
              --output_dir=output_inference
```

</details>

<details>
<summary>2. Convert to ONNX </summary>

- Install [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) and ONNX

```shell
pip install onnx==1.13.0
pip install paddle2onnx==1.0.5
```

- Convert:

```shell
paddle2onnx --model_dir=./output_inference/rtdetrv3_r18vd_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetrv3_r18vd_6x_coco.onnx
```
</details>

<details>
<summary>3. Convert to TensorRT </summary>

- TensorRT version >= 8.5.1
- Inference can refer to [Bennchmark](../benchmark)

```shell
trtexec --onnx=./rtdetrv3_r18vd_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetrv3_r18vd_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```
-
</details>

## Citation

If you find RT-DETRv3 useful in your research, please consider giving a star ⭐ and citing:

```
@article{wang2024rt,
  title={RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision},
  author={Wang, Shuo and Xia, Chunlong and Lv, Feng and Shi, Yifeng},
  journal={arXiv preprint arXiv:2409.08475},
  year={2024}
}
```
