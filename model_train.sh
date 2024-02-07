yolo detect train data=../../DATA/datasets/step3_1_test/step3_1_test.yaml model=yolov8l.pt batch=16 epochs=100 device=0

# 실행 횟수
# num_runs=4

# for i in $(seq 2 $((num_runs-1))); do
#     batch=$(echo "2^$i" | bc)  # 배치 크기 계산
#     yolo detect train data=../../DATA/datasets/step3_1_test/step3_1_test.yaml model=yolov5s.pt batch=$batch epochs=100 device=3
# done



# # batch = 1, 2, 4, 8, 16, 32
# num_runs=6

# for i in $(seq 0 $((num_runs-1))); do
#     batch=$((1 << $i))
    
#     if (( $i >= 4 )); then
#         batch=$((batch * 3))
#     fi
    
#     yolo detect train data=../../DATA/datasets/step3_1_test/step3_1_test.yaml model=yolov8l.pt imgsz=320 batch=$batch epochs=100 device=1
# done



