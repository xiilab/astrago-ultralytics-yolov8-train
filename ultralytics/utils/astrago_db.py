import os

import mysql.connector
from kubernetes import client, config


class KubernetesInfo:

    def __init__(self):
        config.load_incluster_config()
        self.pod_name = os.environ.get("POD_NAME")
        self.pod_namespace = os.environ.get("POD_NAMESPACE")
        self.api_instance = client.CoreV1Api()

    def change_estimated_remaining_time(self, estimated_remaining_time):
        try:
            # Pod 조회
            pod = self.api_instance.read_namespaced_pod(name=self.pod_name, namespace=self.pod_namespace)

            # 기존 annotation에 새로운 값을 추가 또는 업데이트
            if pod.metadata.annotations is None:
                pod.metadata.annotations = {}
            pod.metadata.annotations['estimated_remaining_time'] = str(estimated_remaining_time)

            # Pod 업데이트
            self.api_instance.patch_namespaced_pod(name=self.pod_name, namespace=self.pod_namespace, body=pod)
            print("Pod의 annotation이 성공적으로 업데이트되었습니다.")
        except Exception as e:
            print(f"Pod annotation 업데이트 중 오류 발생: {e}")

    def change_estimated_initial_time(self, estimated_initial_time):
        try:
            # Pod 조회
            pod = self.api_instance.read_namespaced_pod(name=self.pod_name, namespace=self.pod_namespace)

            # 기존 annotation에 새로운 값을 추가 또는 업데이트
            if pod.metadata.annotations is None:
                pod.metadata.annotations = {}
            pod.metadata.annotations['estimated_initial_time'] = str(estimated_initial_time)

            # Pod 업데이트
            self.api_instance.patch_namespaced_pod(name=self.pod_name, namespace=self.pod_namespace, body=pod)
            print("Pod의 annotation이 성공적으로 업데이트되었습니다.")
        except Exception as e:
            print(f"Pod annotation 업데이트 중 오류 발생: {e}")

    def get_node_port(self):
        # Kubernetes 클러스터 설정
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        # 현재 파드의 네임스페이스와 파드 이름 가져오기
        namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()
        pod_name = open("/var/run/secrets/kubernetes.io/serviceaccount/pod-name").read()

        # 서비스 이름과 노드 포트 가져오기
        service_name = "astrago-mariadb"  # 원하는 서비스 이름
        node_port = None

        # 서비스 목록 가져오기
        services = v1.list_namespaced_service(namespace)
        for svc in services.items:
            if svc.metadata.name == service_name:
                node_port = svc.spec.ports[0].node_port
                break

        print(f"서비스 이름: {service_name}")
        print(f"노드 포트: {node_port}")


class MariaDBHandler:
    def __init__(self, host, port, user, password, database):
        """
        MariaDBHandler 클래스의 생성자 메서드입니다.

        :param host: MariaDB 호스트 주소
        :param user: MariaDB 사용자 이름
        :param password: MariaDB 암호
        :param database: 연결할 데이터베이스 이름
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.conn = None
        self.cursor = None

    def connect(self):
        """
        MariaDB에 연결하는 메서드입니다.
        """
        try:
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                port=self.port,
                password=self.password,
                database=self.database
            )
            self.cursor = self.conn.cursor()
            print("MariaDB에 연결되었습니다.")
        except mysql.connector.Error as e:
            print(f"MariaDB 연결 오류: {e}")

    def disconnect(self):
        """
        MariaDB 연결을 해제하는 메서드입니다.
        """
        if self.conn.is_connected():
            self.cursor.close()
            self.conn.close()
            print("MariaDB 연결이 닫혔습니다.")

    def insert_epoch_log(self, parameter_id, values):
        """
        TB_JOB_PREDICTION_EPOCH_LOG 테이블에 데이터를 삽입하는 메서드입니다.

        :param values: 삽입할 데이터 값
        """
        try:
            # 데이터 삽입 쿼리 생성
            sql = """
            INSERT INTO TB_JOB_PREDICTION_EPOCH_LOG 
            (JOB_PREDICTION_PARAMETER_ID, MODEL_NAME, MODEL_PARAM_NUM,
             GPU_TYPE, GPU_FLOPS, CLASS_NUM, TRAIN_IMG_NUM, TRAIN_INSTANCE_NUM, VALID_IMG_NUM, VALID_INSTANCE_NUM,
             IMG_SIZE, BATCH_SIZE, EPOCH_CNT, PREPROCESS_TIME, TRAIN_TIME, VALID_TIME, TIME_PER_EPOCH, SCHEDULER_TIME,
             EPOCH_TIME, ELAPSED_TIME, REMAINING_TIME, GPU_USAGE, CPU_USAGE) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # 데이터 삽입
            self.cursor.execute(sql, (parameter_id,) + values)

            # 변경사항 커밋
            self.conn.commit()
            print("데이터가 성공적으로 삽입되었습니다.")
        except Exception as e:
            print(f"데이터 삽입 중 오류 발생: {e}")
            # 롤백
            self.conn.rollback()

    def insert_prediction_parameter(self, values):
        """
        TB_JOB_PREDICTION_PARAMETER 테이블에 데이터를 삽입하는 메서드입니다.

        :param values: 삽입할 데이터 값
        """
        try:
            # 데이터 삽입 쿼리 생성
            sql = """
            INSERT INTO TB_JOB_PREDICTION_PARAMETER 
            (MODEL, MODEL_PT, DATA_DIR, IMAGE_SIZE, EPOCH, BATCH_SIZE, LEARNING_RATE,
             SAVE_MODEL_DIR, PATIENCE, WORKER, OPT, SINGLE_CLS, LABEL_SMOOTHING, PRETRAINED) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # 데이터 삽입
            self.cursor.execute(sql, values)

            # 변경사항 커밋
            self.conn.commit()
            print("데이터가 성공적으로 삽입되었습니다.")
            return self.cursor.lastrowid
        except Exception as e:
            print(f"데이터 삽입 중 오류 발생: {e}")
            # 롤백
            self.conn.rollback()


if __name__ == '__main__':
    client = KubernetesInfo()
    client.change_estimated_remaining_time(10)
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
