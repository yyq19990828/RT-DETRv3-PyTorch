PY37=/root/paddlejob/workspace/env_run/ws/py37_meta_pd-2.4_cu11_comer/bin/python3.7
# PY37=../anaconda3/envs/py37_meta_pd-2.3.0_cu11/bin/python3.7
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup $PY37 -m paddle.distributed.launch --gpus=0,1,2,3 \
            tools/train.py \
            -c configs/artdetrv3/rtdetrv3_final_r18vd_6x_coco.yml --eval\
            -r output/rtdetrv3_final_r18vd_6x_coco/1 \
            -o save_dir=output/rtdetrv3_final_r18vd_6x_coco \
            &> output/train_rtdetrv3_final_r18vd_6x_coco.log&