import csv
import glob
import json
import os
import time
import xml.etree.ElementTree as ET

import psutil
import torch
import yaml
from tqdm import tqdm
from .astrago_db import KubernetesInfo


class Astrago(tqdm):
    csv_file_path = "/tmp/epoch_log.csv"
    model_name = 'yolov8l'
    param = 0
    gpu = ''
    FLOPS = 14
    yolo_data_info = {'train': [], 'val': [], 'test': []}
    data_info = {'train': [], 'val': [], 'test': []}
    image_size = 0
    batch_size = 0
    gpu_memory_usage = [0]
    cpu_usage = [0]
    preprocess_time = 0
    train_time = 0
    val_time = 0
    savemodel_time = 0
    scheduler_time = 0

    # csv 파일 저장 함수
    @staticmethod
    def save_metrics_to_csv(model_name, param, gpu, FLOPS, class_num, t_img_num, t_instance_num, v_img_num,
                            v_instance_num,
                            imgsz, batch, epoch, preprocess_time, train_time, val_time, save_model_time, scheduler_time,
                            epoch_time,
                            elapsed, remaining, gpu_usage, cpu_usage):
        '''
            model_name : model 이름
            param : model 파라미터 수
            gpu : gpu 종류
            FLOPS : gpu flops
            class_num : 클래스 수
            t_img num : train 이미지 수
            t_instance_num : trian 객체 수
            v_img num : validation 이미지 수
            v_instance_num : validation 객체 수
            imgsz : input 이미지 사이즈 (pixel)
            batch : 배치 사이즈
            epoch : 에폭 수
            preprocess_time : 학습 전 초기화하는데 걸리는 시간 (sec)
            train_time : 학습하는 데 걸린 시간 (sec)
            val_time : 검증하는 데 걸린 시간 (sec)
            save_model_time : 매 에폭마다 학습 가중치 저장하는 시간 (sec)
            scheduler_timer : 스케줄러 업데이트 시간 (sec)
            epoch_time : 에폭 처리 시간 (sec)
            elapsed : 누적 학습 시간 (sec)
            remaining : 예상 남은 시간 (sec)
            gpu_usage : gpu 사용량 (GiB)
            cpu_usage : cpu 사용률 (%)
        '''
        k8s_info = KubernetesInfo()
        if epoch == 1:
            k8s_info.change_initial_time_annotation(remaining)
        k8s_info.change_remaining_time_annotation(remaining)
        with open(Astrago.csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(
                    ['model_name', 'param', 'gpu', 'FLOPS', 'class_num', 't_img_num', 't_instance_num', 'v_img_num',
                     'v_instance_num', 'imgsz', 'batch', 'epoch', 'preprocess_time', 'train_time', 'val_time',
                     'save_model_time',
                     'scheduler_time', 'epoch_time', 'elapsed', 'remaining', 'gpu_usage', 'cpu_usage'])

            writer.writerow(
                [model_name, param, gpu, FLOPS, class_num, t_img_num, t_instance_num, v_img_num, v_instance_num,
                 imgsz, batch, epoch, preprocess_time, train_time, val_time, save_model_time, scheduler_time,
                 epoch_time,
                 elapsed, remaining, gpu_usage, cpu_usage])

    @staticmethod
    def format_meter(n, total, elapsed, rate=None, initial=0, *args, **kwargs):
        torch.cuda.synchronize()
        t = Astrago.format_interval((total - n) / rate) if rate else 0
        t = Astrago.time_to_seconds(str(t))
        epoch_per_time = (elapsed / n) if n > 0 else 0
        gpu_usage = sum(Astrago.gpu_memory_usage) / (len(Astrago.gpu_memory_usage) - 1) if n != 0 else sum(
            Astrago.gpu_memory_usage) / len(Astrago.gpu_memory_usage)
        cpu_usage = sum(Astrago.cpu_usage) / (len(Astrago.cpu_usage) - 1) if n != 0 else sum(Astrago.cpu_usage) / len(
            Astrago.cpu_usage)
        d_info = Astrago.data_info if len(Astrago.data_info['train']) != 0 else Astrago.yolo_data_info
        t_img_num, t_instance_num, class_num = d_info['train']
        v_img_num, v_instance_num, _ = d_info['val']
        imgsz = Astrago.image_size
        batch = Astrago.batch_size
        preprocess_time = Astrago.preprocess_time
        train_time = Astrago.train_time
        val_time = Astrago.val_time
        savemodel_time = Astrago.savemodel_time
        scheduler_time = Astrago.scheduler_time

        # 마지막 에폭만 두 번 출력되고 저장되는 경우가 있어 임시 해결책
        if n == total:
            if not hasattr(Astrago, 'last_epoch_done') or not Astrago.last_epoch_done:  # 마지막 에폭이 한 번만 출력되도록 확인하는 변수
                Astrago.save_metrics_to_csv(Astrago.model_name, Astrago.param, Astrago.gpu, Astrago.FLOPS, class_num,
                                            t_img_num,
                                            t_instance_num, v_img_num, v_instance_num, imgsz, batch, n, preprocess_time,
                                            train_time,
                                            val_time, savemodel_time, scheduler_time, epoch_per_time, elapsed, t,
                                            gpu_usage, cpu_usage)

                print(f'\n현재 epoch > {n}/{total}')
                print(f'preprocess_time > {preprocess_time}')
                print(f'train 시간 > {train_time}')
                print(f'validation 시간 > {val_time}')
                print(f'save model time > {savemodel_time}')
                print(f'scheduler_time > {scheduler_time}')
                print(f'처리 시간 > {epoch_per_time}s')
                print(f'종료까지 남은 예상 시간 > {t}')
                print(f'현재까지 작업에 소요된 시간 > {elapsed}s')
                print(f'GPU 메모리 사용량: {gpu_usage:.2f}G')
                print(f'CPU 메모리 사용율: {cpu_usage:.2f}%')
                Astrago.last_epoch_done = True  # 마지막 에폭이 출력되었음을 표시


        elif n > 0:  # 마지막 에폭이 아닐 때는 출력 및 저장을 수행
            Astrago.save_metrics_to_csv(Astrago.model_name, Astrago.param, Astrago.gpu, Astrago.FLOPS, class_num,
                                        t_img_num,
                                        t_instance_num, v_img_num, v_instance_num, imgsz, batch, n, preprocess_time,
                                        train_time,
                                        val_time, savemodel_time, scheduler_time, epoch_per_time, elapsed, t, gpu_usage,
                                        cpu_usage)

            print(f'\n현재 epoch > {n}/{total}')
            print(f'preprocess_time > {preprocess_time}')
            print(f'train 시간 > {train_time}')
            print(f'validation 시간 > {val_time}')
            print(f'save model time > {savemodel_time}')
            print(f'scheduler_time > {scheduler_time}')
            print(f'처리 시간 > {epoch_per_time}s')
            print(f'종료까지 남은 예상 시간 > {t}')
            print(f'현재까지 작업에 소요된 시간 > {elapsed}s')
            print(f'GPU 메모리 사용량: {gpu_usage:.2f}G')
            print(f'CPU 메모리 사용율: {cpu_usage:.2f}%')

        return tqdm.format_meter(n, total, elapsed, rate=rate, initial=initial, *args, **kwargs)

    def get_elapsed_preprocess_time(start_time):
        Astrago.preprocess_time = time.time() - start_time

    def get_elapsed_train_time(start_time):
        Astrago.train_time = time.time() - start_time

    def get_elapsed_val_time(start_time):
        Astrago.val_time = time.time() - start_time

    def get_elapsed_savemodel_time(start_time):
        Astrago.savemodel_time = time.time() - start_time

    def get_elapsed_scheduler_time(start_time):
        Astrago.scheduler_time = time.time() - start_time

    # 모델 train 시 사용하고 있는 gpu 정보
    def get_gpu_info():
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        Astrago.gpu = gpu_name

    # 시간 변환 함수
    def time_to_seconds(time_str):
        components = [int(x) for x in time_str.split(':')]
        while len(components) < 3:
            components.insert(0, 0)

        hours, minutes, seconds = components

        return hours * 3600 + minutes * 60 + seconds

    # 모델 predict 시 gpu 사용량 추출 함수
    def get_gpu_memory_usage():
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}'
        Astrago.gpu_memory_usage.append(float(mem))

    # 모델 predict 시 cpu 사용률 추출 함수    
    def get_cpu_usage():
        cpu = psutil.cpu_percent(interval=1)
        Astrago.cpu_usage.append(cpu)

    # 모델 파라미터 수 추출 함수    
    def get_model_params(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'\n[Params]')
        print(f'파라미터 수 : {total_params}')
        Astrago.param = total_params

    # 모델 train 시, input image 사이즈 추출 함수     
    def get_image_size(image_size_argument):
        image_size = image_size_argument
        print(f'\n[Image Size]')
        print(f'이미지 사이즈 : {image_size}')
        Astrago.image_size = image_size

    # 모델 train 시, 배치 사이즈 추출 함수
    def get_batch_size(batch_argument):
        batch_size = batch_argument
        print(f'\n[Batch Size]')
        print(f'배치 사이즈 : {batch_size}')
        Astrago.batch_size = batch_size

    # coco & voc 형식 데이터 이미지 및 객체 수량 추출 함수    
    def get_data_info(dataset_folder_path):
        file_formats = ['.xml', '.json']
        dataset_list = ['train', 'val', 'test']

        found_folder, format = None, None

        for dataset in dataset_list:
            for path, dir, files in os.walk(dataset_folder_path):
                if dataset in os.path.basename(path):
                    found_folder = path
                    break

            for file_format in file_formats:
                matching_files = glob.glob(os.path.join(found_folder, f'**/*{file_format}'), recursive=True)
                if matching_files:
                    found_folder = os.path.dirname(matching_files[0])
                    format = file_format
                    break

            if found_folder and format:
                if format == '.json':
                    data_path = ''.join(glob.glob(found_folder + f'**/*.json'))
                    with open(data_path, 'r') as file:
                        data = json.load(file)
                        class_num = len(data['categories'])
                        data_num = len(data['images'])
                        instance_num = len(data['annotations'])
                elif format == '.xml':
                    data_num = len(os.listdir(found_folder))

                    classes = set()
                    instance_num = 0
                    for xml_file in os.listdir(found_folder):
                        xml_file_path = os.path.join(found_folder, xml_file)

                        tree = ET.parse(xml_file_path)
                        root = tree.getroot()

                        for instance in root.findall('.//object'):
                            class_name = instance.find('name').text
                            classes.add(class_name)
                            instance_num += 1
                    class_num = len(classes)

                Astrago.data_info[dataset].append(data_num)
                Astrago.data_info[dataset].append(instance_num)
                Astrago.data_info[dataset].append(class_num)

                print(f'[{dataset}]')
                print(f'전체 이미지 수 : {data_num}')
                print(f'전체 객체 수 : {instance_num}')
                print(f'클래스 수 : {class_num}\n')

    # yolo 모델 형식 데이터 이미지 및 객체 수량 추출 함수        
    def get_data_info_yolo(yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        class_num = len(data['names'])
        print(f'클래스 수 : {class_num}\n')

        dataset_list = ['train', 'val']

        for dataset in dataset_list:
            dataset_path = data[dataset]

            data_num = int(len(os.listdir(dataset_path)) / 2)

            instance_num = 0
            for file in os.listdir(dataset_path):
                if file.endswith('.txt'):
                    txt_path = os.path.join(dataset_path, file)
                    with open(txt_path, 'r') as f:
                        contents = f.readlines()
                        contents = set(contents)
                        instance_num += len(contents)

            Astrago.yolo_data_info[dataset].append(data_num)
            Astrago.yolo_data_info[dataset].append(instance_num)
            Astrago.yolo_data_info[dataset].append(class_num)

            print(f'[{dataset}]')
            print(f'전체 이미지 수 : {data_num}')
            print(f'전체 객체 수 : {instance_num}')


class Inference(tqdm):
    csv_file_path = "/DATA/job_time_prediction/datasets/inference_data/predict_yolov8s.csv"
    model_name = 'yolov8s'
    param = 0
    gpu = ''
    FLOPS = 14
    data_num = 0
    image_size = 0
    gpu_memory_usage = [0]
    cpu_usage = [0]
    inference_time = 0
    save_time = 0
    single_data_inference_time = 0

    # csv 파일 저장 함수
    def save_metrics_to_csv(model_name, param, gpu, FLOPS, data_num, imgsz, num, inference_time,
                            save_time, single_data_inference_time, gpu_usage, cpu_usage):
        '''
            model_name : model 이름
            param : model 파라미터 수
            gpu : gpu 종류
            FLOPS : gpu flops
            data_num : 예측할 이미지 수
            imgsz : input 이미지 사이즈 (pixel)
            num : 처리 중인 이미지(프레임) 순번
            inference_time : inference 하는 데만 걸린 시간
            save_time : inference 결과 저장하는 데 걸린 시간
            single_data_inference_time : 이미지 하나 당 inference 속도 (sec) == inference_time + save_time
            gpu_usage : gpu 사용량 (GiB)
            cpu_usage : cpu 사용률 (%)
        '''

        with open(Inference.csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(['model_name', 'param', 'gpu', 'FLOPS', 'data_num', 'imgsz', 'num', 'inference_time',
                                 'save_time', 'single_data_inference_time', 'gpu_usage', 'cpu_usage'])

            writer.writerow([model_name, param, gpu, FLOPS, data_num, imgsz, num, inference_time,
                             save_time, single_data_inference_time, gpu_usage, cpu_usage])

    @staticmethod
    def format_meter(n, total, elapsed, rate=None, initial=0, *args, **kwargs):
        torch.cuda.synchronize()
        gpu_usage = sum(Inference.gpu_memory_usage) / (len(Inference.gpu_memory_usage) - 1) if n != 0 else sum(
            Inference.gpu_memory_usage) / len(Inference.gpu_memory_usage)
        cpu_usage = sum(Inference.cpu_usage) / (len(Inference.cpu_usage) - 1) if n != 0 else sum(
            Inference.cpu_usage) / len(Inference.cpu_usage)
        data_num = Inference.data_num
        imgsz = Inference.image_size
        inference_time = Inference.inference_time
        save_time = Inference.save_time
        single_data_inference_time = inference_time + save_time

        if total != 1:

            # 마지막 에폭만 두 번 출력되고 저장되는 경우가 있어 임시 해결책
            if n == total:
                if not hasattr(Inference,
                               'last_epoch_done') or not Inference.last_epoch_done:  # 마지막 에폭이 한 번만 출력되도록 확인하는 변수
                    Inference.save_metrics_to_csv(Inference.model_name, Inference.param, Inference.gpu, Inference.FLOPS,
                                                  data_num, imgsz, n,
                                                  inference_time, save_time, single_data_inference_time, gpu_usage,
                                                  cpu_usage)

                    print(f'\n현재 진행 순번> {n}/{total}')
                    print(f'inference time > {inference_time}')
                    print(f'save time > {save_time}')
                    print(f'이미지 1장 당 추론 시간 > {single_data_inference_time}')
                    print(f'GPU 메모리 사용량: {gpu_usage:.2f}G')
                    print(f'CPU 메모리 사용율: {cpu_usage:.2f}%')
                    Inference.last_epoch_done = True  # 마지막 에폭이 출력되었음을 표시


            else:  # 마지막 에폭이 아닐 때는 출력 및 저장을 수행
                Inference.save_metrics_to_csv(Inference.model_name, Inference.param, Inference.gpu, Inference.FLOPS,
                                              data_num, imgsz, n,
                                              inference_time, save_time, single_data_inference_time, gpu_usage,
                                              cpu_usage)

                print(f'\n현재 진행 순번> {n}/{total}')
                print(f'inference time > {inference_time}')
                print(f'save time > {save_time}')
                print(f'이미지 1장 당 추론 시간 > {single_data_inference_time}')
                print(f'GPU 메모리 사용량: {gpu_usage:.2f}G')
                print(f'CPU 메모리 사용율: {cpu_usage:.2f}%')

        return tqdm.format_meter(n, total, elapsed, rate=rate, initial=initial, *args, **kwargs)

    # 모델 predict 시 inference 시간 추출 함수
    def get_elapsed_inference_time(start_time):
        Inference.inference_time = time.time() - start_time

    # 모델 predict 시 inference 결과 저장 시간 추출 함수
    def get_elapsed_save_time(start_time):
        Inference.save_time = time.time() - start_time

    # predict 이미지/프레임 수 추출 함수
    def get_data_num(data_info_argument):
        data_num = len(data_info_argument)
        print(f'\n[Data]')
        print(f'데이터 수 : {data_num}')
        Inference.data_num = data_num

    # 모델 predict 시 사용하고 있는 gpu 정보 추출 함수
    def get_gpu_info():
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        Inference.gpu = gpu_name

    # 시간 변환 함수
    def time_to_seconds(time_str):
        components = [int(x) for x in time_str.split(':')]
        while len(components) < 3:
            components.insert(0, 0)

        hours, minutes, seconds = components

        return hours * 3600 + minutes * 60 + seconds

    # 모델 predict 시 gpu 사용량 추출 함수
    def get_gpu_memory_usage():
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}'
        Inference.gpu_memory_usage.append(float(mem))

    # 모델 predict 시 cpu 사용률 추출 함수    
    def get_cpu_usage():
        cpu = psutil.cpu_percent(interval=1)
        Inference.cpu_usage.append(cpu)

    # 모델 파라미터 수 추출 함수    
    def get_model_params(model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'\n[Params]')
        print(f'파라미터 수 : {total_params}')
        Inference.param = total_params

    # 모델 predict 시, input image 사이즈 추출 함수     
    def get_image_size(image_size_argument):
        image_size = image_size_argument
        print(f'\n[Image Size]')
        print(f'이미지 사이즈 : {image_size}')
        Inference.image_size = image_size
