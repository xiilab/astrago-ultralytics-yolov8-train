import os
import mysql.connector
from kubernetes import client, config

class kubernetes_info :

    def get_node_port :
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

class db :

    def connection_db():
        # MariaDB에 연결
        astra_db = mysql.connector.connect(
        host="10.61.3.12",
        port=int(30756),
        user="root",
        password="root",
        database="astrago"
        )

        return astra_db
    def insert_prediction_parameter(astra_db):
        #TB_JOB_PREDICTION_PARAMETER
        cursor = astra_db.cursor()
        sql_insert = f"""
                    INSERT INTO TB_JOB_PREDICTION_PARAMETER
                    (
                        MODEL
                        ,MODEL_PT
                        ,DATA_DIR
                        ,IMAGE_SIZE
                        ,EPOCH
                        ,BATCH_SIZE
                        ,LEARNING_RATE
                        ,SAVE_MODEL_DIR
                        ,PATIENCE
                        ,WORKER
                        ,OPT
                        ,SINGLE_CLS
                        ,LABEL_SMOOTHING
                        ,PRETRAINED
                    )
                    VALUES
                    (
                        '{annotationSet}'
                        ,"{data_obj['DATA_ID']}"
                        ,"{data_obj['LABEL_ID']}"
                        ,"{label_type}"
                        ,"{last_worker_account}"
                        ,"SYSTEM"
                        ,DATE_FORMAT(CURRENT_TIMESTAMP, '%Y-%m-%d %T')
                        ,"SYSTEM"
                        ,DATE_FORMAT(CURRENT_TIMESTAMP, '%Y-%m-%d %T')
                    )
                    """
        cursor.execute(sql_insert)
        cursor.close()
        astra_db.commit()

        pk_value = cursor.lastrowid
        return pk_value

    def insert_EPOCH_LOG(astra_db):
        #TB_JOB_PREDICTION_PARAMETER
        cursor = astra_db.cursor()
        sql_insert = f"""
                    INSERT INTO TB_JOB_PREDICTION_EPOCH_LOG
                    (
                        JOB_PREDICTION_EPOCH_LOG_ID
                        ,JOB_PREDICTION_PARAMETER_ID
                        ,TOTAL_EPOCH
                        ,CURRENT_EPOCH
                        ,PREPROCESS_TIME
                        ,VALIDATION_TIME
                        ,SAVE_MODEL_TIME
                        ,SCHEDULER_TIME
                        ,EPOCH_PER_TIME
                        ,PREDICTION_END_TIME
                        ,ELAPSED_TIME
                        ,GPU_USAGE
                        ,CPU_USAGE
                    )
                    VALUES
                    (
                        '{annotationSet}'
                        ,"{data_obj['DATA_ID']}"
                        ,"{data_obj['LABEL_ID']}"
                        ,"{label_type}"
                        ,"{last_worker_account}"
                        ,"SYSTEM"
                        ,DATE_FORMAT(CURRENT_TIMESTAMP, '%Y-%m-%d %T')
                        ,"SYSTEM"
                        ,DATE_FORMAT(CURRENT_TIMESTAMP, '%Y-%m-%d %T')
                    )
                    """
        cursor.execute(sql_insert)
        cursor.close()
        astra_db.commit()
