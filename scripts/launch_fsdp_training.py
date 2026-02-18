import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()

role = "arn:aws:iam::995610519414:role/SageMakerExecutionRole-MarketStructureML"

bucket = "quant-reasoning-fsdp-qlora-artifacts"
prefix = "quant-reasoning-fsdp-qlora/fsdp-run"

train_data = f"s3://{bucket}/data/quant_reasoning_dataset.jsonl"

estimator = PyTorch(
    entry_point="train_fsdp_lora.py",
    source_dir="train",
    role=role,
    framework_version="2.1.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g5.12xlarge",
    hyperparameters={
        "train_data": "/opt/ml/input/data/train/quant_reasoning_dataset.jsonl",
    },
    distribution={"torch_distributed": {"enabled": True}},
    output_path=f"s3://{bucket}/{prefix}/output",
)

estimator.fit({"train": train_data})