import os

import boto3
from botocore.exceptions import ClientError

client = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
try:
    client.head_bucket(Bucket="mlflow-artifacts")
except ClientError as error:
    if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
        raise
    client.create_bucket(Bucket="mlflow-artifacts")
