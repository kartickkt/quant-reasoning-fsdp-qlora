import boto3

s3 = boto3.client("s3", region_name="us-east-1")

resp = s3.list_objects_v2(
    Bucket="quant-reasoning-fsdp-qlora-artifacts",
    Prefix="data/"
)

print(resp.get("Contents"))