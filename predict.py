import os
import argparse
from ultralytics import YOLO

#save dir filter same folder
def make_unique_folder(path):
    path = path.rstrip("/")
    folder_name = os.path.basename(path)
    folder_path = os.path.dirname(path)
    new_folder_path = path
    i = 1
    while os.path.exists(new_folder_path): 
        new_folder_name = f"{folder_name}_{i}"
        new_folder_path = os.path.join(folder_path, new_folder_name)
        i += 1
    os.makedirs(new_folder_path)
    return new_folder_path


def predicter(config) :
    new_folder = make_unique_folder(config['save_dir'])
    print(f"New folder created: {new_folder}")

    model = YOLO(config['model'])

    model.predict(
        source = config['source'],
        save=True,
        imgsz = config['imgsz'],
        save_dir = new_folder,
        iou = 0.7,
        conf=0.25
    )
    

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Detecting')
    #user setting
    parser.add_argument('--save_dir', type=str, default="../detect/run", help="detect result save dir")
    parser.add_argument('--source', type=str, default="/DATA/datasets/step3_1_test/test", help="data folder path")    
    parser.add_argument('--imgsz', type=int, default=640, help="input image scale")
    parser.add_argument('--model', type=str, default="../weights/yolov8l.pt", help="pre-trained model path")
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()
    predicter(args)
