#!/bin/bash

GPU=0
SAVE_DIR="../weight/yolov8/AstraGO"
YAML="/DATA/datasets/step3_1_test/step3_1_test.yaml"
MODEL="yolov8l.yaml"
PRETRAIN="../weights/yolov8l.pt"

CUDA_VISIBLE_DEVICES=$GPU python train.py \
	--data_dir $YAML \
	--model $MODEL \
	--model_pt $PRETRAIN \
	--image_size 640 \
	--epochs 100 \
	--batch_size 2 \
	--learning_rate 0.01 \
	--save_model_dir $SAVE_DIR
