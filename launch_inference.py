from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::995610519414:role/SageMakerExecutionRole-MarketStructureML"
bucket = "quant-reasoning-fsdp-qlora-artifacts"

estimator = PyTorch(
    entry_point="infer_entry.py",
    source_dir="train",
    role=role,
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.1",
    py_version="py310",
    base_job_name="quant-reasoning-compare",
)

estimator.fit(
    {
        "val": f"s3://{bucket}/data/quant_reasoning_val.jsonl",
        "adapter": "s3://quant-reasoning-fsdp-qlora-artifacts/quant-reasoning-fsdp-qlora/fsdp-run/output/pytorch-training-2026-02-15-11-03-45-496/output/model.tar.gz",
    }
)