import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# -----------------------------
# Explicit SageMaker session (CRITICAL)
# -----------------------------
session = sagemaker.Session()

# -----------------------------
# Config
# -----------------------------
ROLE = "arn:aws:iam::995610519414:role/SageMakerExecutionRole-MarketStructureML"

MODEL_S3_PATH = (
    "s3://quant-reasoning-fsdp-qlora-artifacts/"
    "quant-reasoning-fsdp-qlora/fsdp-run/output/"
    "pytorch-training-2026-02-15-11-03-45-496/output/model.tar.gz"
)

ENDPOINT_NAME = "quant-reasoning-mistral-prod-v9"
INSTANCE_TYPE = "ml.g5.xlarge"

# -----------------------------
# Create HuggingFace model
# -----------------------------
huggingface_model = HuggingFaceModel(
    model_data=MODEL_S3_PATH,
    role=ROLE,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
    entry_point="inference.py",
    source_dir="scripts",
    sagemaker_session=session,  # ⭐ critical fix
)

# -----------------------------
# Deploy endpoint
# -----------------------------
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    endpoint_name=ENDPOINT_NAME,
)

print(f"\n✅ Endpoint deployed: {ENDPOINT_NAME}")
