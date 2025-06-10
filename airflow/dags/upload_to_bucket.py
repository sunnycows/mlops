from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime
from pathlib import Path
import boto3
from botocore.client import Config
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path(__file__).parent / 'secrets_db.env')

# ---- S3/MinIO Config from .env ----
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
REGION = os.getenv('REGION')

# Local file
LOCAL_DIR = Path('input/train')
PREFIX = 'train'
FILENAME = 'train_030625.csv'
LOCAL_FILE_PATH = LOCAL_DIR / FILENAME


def upload_to_bucket():
    """Upload train.csv to the MinIO bucket"""
    s3 = boto3.client('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name=REGION
    )

    if not LOCAL_FILE_PATH.exists():
        raise FileNotFoundError(f"{LOCAL_FILE_PATH} not found")

    print(f"Uploading {LOCAL_FILE_PATH} to bucket {BUCKET_NAME} with prefix {PREFIX}")
    s3.upload_file(str(LOCAL_FILE_PATH), BUCKET_NAME, PREFIX + '/' + FILENAME)
    print(f"Uploaded {FILENAME} to bucket {BUCKET_NAME}")


with DAG(
    dag_id='upload_to_bucket',
    start_date=datetime(2025, 1, 1),
    schedule='@daily',  # Or use None if you want to run it manually
    catchup=False,
    tags=['minio', 'upload'],
) as dag:

    upload_task = PythonOperator(
        task_id='upload_train_csv',
        python_callable=upload_to_bucket
    )

    trigger_databricks = DatabricksRunNowOperator(
            task_id='971401967069115',
            databricks_conn_id='Databricks',
            job_id='971401967069115'
        )

    upload_task >> trigger_databricks

# upload_to_bucket()