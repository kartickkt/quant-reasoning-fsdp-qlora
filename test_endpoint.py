import boto3
import json
import time

ENDPOINT_NAME = "quant-reasoning-mistral-prod-v9"
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

payload = {
    "prompt": "If a car travels 150 km in 3 hours, what is its average speed?"
}

latencies = []

for i in range(20):
    start = time.time()

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode())
    client_latency = time.time() - start
    latencies.append(client_latency)

    print(f"Call {i+1}: {result} | Client latency: {round(client_latency, 3)} sec")

print("\n==== SUMMARY ====")
print("Average latency:", round(sum(latencies) / len(latencies), 3))
print("Max latency:", round(max(latencies), 3))
print("Min latency:", round(min(latencies), 3))
