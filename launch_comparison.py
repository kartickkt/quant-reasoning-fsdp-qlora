from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::995610519414:role/SageMakerExecutionRole-MarketStructureML"

estimator = PyTorch(
    entry_point="comparison_entry.py",
    source_dir="train",
    role=role,
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.1",
    py_version="py310",
    base_job_name="quant-reasoning-comparison-v2",
)

estimator.fit(
    {
        "val": "s3://quant-reasoning-fsdp-qlora-artifacts/data/quant_reasoning_val.jsonl",
        "adapter": "s3://quant-reasoning-fsdp-qlora-artifacts/quant-reasoning-fsdp-qlora/fsdp-run/output/pytorch-training-2026-02-15-11-03-45-496/output/model.tar.gz",
    }
)