#!/bin/bash


# Single GPU
CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_path <path_to_data> \
    --output_dir <path_to_output> \
    --model_name t5-large \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-5

# for accelerate MULTI_GPU

accelerate launch --config_file accelerate_config.yaml \
    train.py \
    --data_path <path_to_data> \
    --output_dir <path_to_output> \
    --model_name t5-large \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-5