#!/usr/bin/env bash
EXP_DIR=v3-large-minimalistic_224_1.0_float


CUDA_VISIBLE_DEVICES=$1 python post_training_quantization.py \
    --model_name=mobilenet_v3_large_minimalistic \
    --checkpoint_path=${EXP_DIR}/ema/model-342500 \
    --output_tflite=${EXP_DIR}/model_quantized_fixed.tflite \
    --image_size=$2 \
    --dataset_dir=/nobackup/users/jzpan/datasets/imagenet-tfds \
    --enable_ema=False

