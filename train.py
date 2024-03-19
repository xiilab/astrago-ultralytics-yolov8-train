import argparse
import os

from ultralytics import YOLO
from ultralytics.utils.astrago_db import MariaDBHandler


# model freeze
def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are freezed.")


# save dir filter same folder
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


def trainer(config):
    new_folder = make_unique_folder(config['save_model_dir'])
    print(f"New folder created: {new_folder}")

    # Load a model
    # model = YOLO('yolov8l.yaml')  # build a new model from YAML
    # model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)
    model = YOLO(config['model']).load(config['model_pt'])  # build from YAML and transfer weights
    model.add_callback("on_train_start", freeze_layer)

    # Train the model
    results = model.train(
        data=config['data_dir'],
        imgsz=config['image_size'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        lr0=config['learning_rate'],
        lrf=config['learning_rate'],
        project=new_folder,

        device=[0],
        workers=config['worker'],
        optimizer=config['opt'],
        pretrained=config['pretrained'],
        single_cls=config['single_cls'],
        label_smoothing=config['label_smoothing'],
        patience=config['patience'],
    )
    model.val()
    db_handler = MariaDBHandler(
        host='10.61.3.12',
        port='30756',
        user='root',
        password='root',
        database='astrago'
    )
    db_handler.connect()
    parameter_id = db_handler.insert_prediction_parameter(('yolov8l.yaml', '../weights/yolov8l.pt',
                                                           '../astrago-ultralytics-yolov8-train/ultralytics/cfg/datasets/coco128.yaml',
                                                           640, 100, 16, 0.01, '../detect/run', 0, 8, 'SGD', True, 0.0,
                                                           False))
    db_handler.insert_epoch_log(parameter_id, ('yolov8l.yaml', 8, 'TESLA V100', 235, 8,  # CLASS_NUM
                                               8, 8, 8, 8, 360,  # IMG_SIZE
                                               30, 1, 10, 10, 10,  # VALID_TIME
                                               10, 10, 10, 10, 10,  # REMAINING_TIME
                                               3, 10
                                               ))
    db_handler.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 training')
    # user setting
    parser.add_argument('--data_dir', type=str,
                        default="../astrago-ultralytics-yolov8-train/ultralytics/cfg/datasets/coco128.yaml",
                        help="yaml file")
    parser.add_argument('--image_size', type=int, default=640, help="input image scale")
    parser.add_argument('--epochs', type=int, default=100, help="num epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="number of images per batch (-1 for AutoBatch)")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="output checkpoint, info save_dir")
    parser.add_argument('--save_model_dir', type=str, default="../detect/run", help="output checkpoint, info save_dir")

    parser.add_argument('--model', type=str, default="yolov8l.yaml", help="model .yaml default yolov8n.yaml")
    parser.add_argument('--model_pt', type=str, default="../weights/yolov8l.pt", help="pre-trained model path")
    parser.add_argument('--patience', type=int, default=0, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--worker', type=int, default=8,
                        help="number of worker threads for data loading (per RANK if DDP)")
    parser.add_argument('--pretrained', type=bool, default=False, help="whether to use a pretrained model")
    parser.add_argument('--opt', type=str, default='SGD',
                        help="optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']")
    parser.add_argument('--single_cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    trainer(args)
