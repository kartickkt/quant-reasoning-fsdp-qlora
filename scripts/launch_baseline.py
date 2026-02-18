import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path
import boto3

# -------------------------------------------------------
# Config
# -------------------------------------------------------

REGION = "us-east-1"
ROLE = "arn:aws:iam::995610519414:role/SageMakerExecutionRole-MarketStructureML"
INSTANCE_TYPE = "ml.g5.2xlarge"
INSTANCE_COUNT = 1

PROJECT_NAME = "quant-reasoning-fsdp-qlora"
JOB_NAME = "baseline-infer-v4"

DATA_FILE = "quant_reasoning_dataset.jsonl"

# -------------------------------------------------------
# Session + paths (explicit region = fewer surprises)
# -------------------------------------------------------

boto_session = boto3.Session(region_name=REGION)
session = sagemaker.Session(boto_session=boto_session)

bucket = "quant-reasoning-fsdp-qlora-artifacts"

s3_prefix = f"{PROJECT_NAME}/baseline"
s3_input_path = f"s3://{bucket}/{s3_prefix}/input"
s3_output_path = f"s3://{bucket}/{s3_prefix}/output"

print(f"Using bucket: {bucket}")
print(f"S3 input:  {s3_input_path}")
print(f"S3 output: {s3_output_path}")

# -------------------------------------------------------
# Upload dataset
# -------------------------------------------------------

local_data_path = Path("data") / DATA_FILE

if not local_data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {local_data_path}. "
        "Run data/generate_dataset.py first."
    )

print("Uploading dataset to S3...")
s3_dataset_path = session.upload_data(
    path=str(local_data_path),
    bucket=bucket,
    key_prefix=f"{s3_prefix}/input",
)

print(f"Dataset uploaded to: {s3_dataset_path}")

# -------------------------------------------------------
# Define PyTorch Estimator (minimal + stable)
# -------------------------------------------------------

estimator = PyTorch(
    entry_point="baseline_infer.py",
    source_dir="train",
    role=ROLE,
    framework_version="2.1",
    py_version="py310",
    instance_count=INSTANCE_COUNT,
    instance_type=INSTANCE_TYPE,
    base_job_name=JOB_NAME,
    output_path=s3_output_path,
    hyperparameters={},  # none needed
)

# -------------------------------------------------------
# Launch job
# -------------------------------------------------------

print("Launching SageMaker baseline inference job...")

estimator.fit({"training": s3_dataset_path})

print("Baseline job submitted successfully.")
print(f"Check logs in SageMaker console under job name prefix: {JOB_NAME}")
