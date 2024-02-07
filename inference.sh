#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 inference.py \
	--weights "../weights/yolov8l.pt" \
	--data_dir "../../DATA/datasets/step3_1_test/test/" \
	--image_size 640 \
	--batch_size 1 \
	--conf_thres 0.25 \
	--iou_thres 0.7