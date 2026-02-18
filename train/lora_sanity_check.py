import os
import tarfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "/opt/ml/input/data/adapter"


def extract_adapter():
    tar_path = os.path.join(ADAPTER_PATH, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(ADAPTER_PATH)


def count_lora_params(model):
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


def main():
    print("\n=== Extracting adapter ===")
    extract_adapter()

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=False)

    print("\n=== Loading BASE model ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("\n=== Attaching LoRA adapter ===")
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("\n=== Parameter counts ===")
    total, trainable = count_lora_params(ft_model)
    print(f"Total params: {total:,}")
    print(f"Trainable params (LoRA): {trainable:,}")

    if trainable == 0:
        raise RuntimeError("❌ LoRA NOT attached — trainable params = 0")

    print("\n=== Logit difference test ===")

    prompt = "<s>[INST] What is 2 + 2? [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(ft_model.device)

    with torch.no_grad():
        base_logits = base_model(**inputs).logits
        ft_logits = ft_model(**inputs).logits

    diff = torch.abs(base_logits - ft_logits).sum().item()
    print(f"Logit absolute difference sum: {diff:.6f}")

    if diff == 0:
        raise RuntimeError("❌ LoRA has NO effect on logits")

    print("\n✅ SUCCESS: LoRA is ACTIVE and modifying model outputs.")


if __name__ == "__main__":
    main()