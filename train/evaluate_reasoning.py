import json
import re
import os
import tarfile
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


VAL_PATH = "/opt/ml/input/data/val/quant_reasoning_val.jsonl"
OUTPUT_PATH = "/opt/ml/output/final_metrics.json"

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "/opt/ml/input/data/adapter"

REL_TOL = 0.01  # 1% tolerance


# -----------------------------
# Utils
# -----------------------------

def extract_number(text: str):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None


def numeric_correct(pred, gt):
    if pred is None or gt is None:
        return False
    if gt == 0:
        return abs(pred) < REL_TOL
    return abs(pred - gt) / abs(gt) < REL_TOL


def parse_example(text: str):
    q = text.split("Question:")[1].split("\nReasoning:")[0].strip()
    c = text.split("Conclusion:")[1].strip()
    return q, c


def load_val():
    data = []
    with open(VAL_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(question):
    return f"<s>[INST] {question} [/INST]"


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------
# Adapter extraction
# -----------------------------

def extract_adapter():
    tar_path = os.path.join(ADAPTER_PATH, "model.tar.gz")

    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(ADAPTER_PATH)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate():
    # 1️⃣ Extract adapter
    extract_adapter()

    # 2️⃣ Load validation data
    val = load_val()

    # 3️⃣ Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=False)

    # 4️⃣ Load BASE model ONCE
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # -----------------------------
    # BASELINE inference
    # -----------------------------
    base_correct = 0
    total = len(val)

    base_outputs = []

    for ex in tqdm(val, desc="Baseline"):
        q, conclusion = parse_example(ex["text"])
        gt = extract_number(conclusion)

        prompt = build_prompt(q)
        base_out = generate(model, tokenizer, prompt)

        base_outputs.append(base_out)

        base_num = extract_number(base_out)
        if numeric_correct(base_num, gt):
            base_correct += 1

    # -----------------------------
    # Attach LoRA adapter IN-PLACE
    # -----------------------------
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # -----------------------------
    # FINETUNED inference
    # -----------------------------
    ft_correct = 0

    for ex in tqdm(val, desc="Finetuned"):
        q, conclusion = parse_example(ex["text"])
        gt = extract_number(conclusion)

        prompt = build_prompt(q)
        ft_out = generate(model, tokenizer, prompt)

        ft_num = extract_number(ft_out)
        if numeric_correct(ft_num, gt):
            ft_correct += 1

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics = {
        "total": total,
        "baseline_accuracy": base_correct / total,
        "finetuned_accuracy": ft_correct / total,
        "gain": (ft_correct - base_correct) / total,
        "relative_tolerance": REL_TOL,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if __name__ == "__main__":
    evaluate()