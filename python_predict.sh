#!/bin/bash
SAVE_DIR="../weight/yolov8/AstraGO/Detect"
SOURCE="/DATA/datasets/step3_1_test/test"
MODEL="../weights/yolov8l.pt"

python predict.py \
	--source $SOURCE \
	--model $MODEL \
	--imgsz 640 \
	--save_dir $SAVE_DIR
