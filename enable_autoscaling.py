import boto3

region = "us-east-1"
endpoint_name = "quant-reasoning-mistral-prod-v9"

autoscaling = boto3.client("application-autoscaling", region_name=region)

# 1️⃣ Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:endpoint:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=3,
)

# 2️⃣ Put scaling policy
autoscaling.put_scaling_policy(
    PolicyName="InvocationsScalingPolicy",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:endpoint:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 1.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
        },
        "ScaleOutCooldown": 60,
        "ScaleInCooldown": 180,
    },
)

print("✅ Autoscaling enabled.")
