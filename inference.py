import os
import argparse
from ultralytics import YOLO
from ultralytics.utils.make_folder import make_unique_folder


from ultralytics import YOLO

<<<<<<< HEAD
def inference(config) :
    new_folder_path, new_folder_name = make_unique_folder(os.path.join(config['save_dir'],"result_pred"))
    model = YOLO(config['weights'])  # pretrained YOLOv8n model
=======

def inference(config):
    new_folder = make_unique_folder(config["save_dir"])
    model = YOLO(config["weights"])  # pretrained YOLOv8n model
>>>>>>> 70693e73370777bf3e2abb88dccf58e63d04ab38
    model.predict(
        source=config["data_dir"],
        save=True,  # save result
        imgsz=config["image_size"],
        conf=config["conf_thres"],
        iou=config["iou_thres"],
        save_conf=True,
        show_boxes=True,
        show_labels=True,
<<<<<<< HEAD
        project=config['save_dir'],
        name=new_folder_name,
        save_frames=True
        #stream=True # derectory
        )
    
    
=======
        name=new_folder,
        save_frames=True,
        # stream=True # directory
    )
    print(f"save_dir: {new_folder}")


>>>>>>> 70693e73370777bf3e2abb88dccf58e63d04ab38
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 training")
    # user setting
    parser.add_argument("--weights", type=str, default="../weights/yolov8.pt", help="yaml file")
    parser.add_argument(
        "--data_dir", type=str, default="analysis image or video path or dir", help="single or folder(multi)"
    )
    parser.add_argument("--image_size", type=int, default=640, help="input image scale")
    parser.add_argument("--batch_size", type=int, default=1, help="only 1 batch")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.7, help="iou threshold")
    parser.add_argument("--save_dir", type=str, default="../detect/AstraGO_pred/yolo", help="save result")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    inference(args)
