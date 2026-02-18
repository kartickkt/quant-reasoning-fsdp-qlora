import json
import re
import os
import tarfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


VAL_PATH = "/opt/ml/input/data/val/quant_reasoning_val.jsonl"
OUTPUT_PATH = "/opt/ml/model/comparison_samples.json"

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "/opt/ml/input/data/adapter"

NUM_SAMPLES = 10


def extract_number(text: str):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None


def parse_example(text: str):
    q = text.split("Question:")[1].split("\nReasoning:")[0].strip()
    c = text.split("Conclusion:")[1].strip()
    return q, c


def load_val():
    data = []
    with open(VAL_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data[:NUM_SAMPLES]


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


def extract_adapter():
    tar_path = os.path.join(ADAPTER_PATH, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(ADAPTER_PATH)


def main():
    extract_adapter()

    val = load_val()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=False)

    # load base once
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    results = []

    # ---------- baseline ----------
    base_outputs = []
    for ex in val:
        q, conclusion = parse_example(ex["text"])
        prompt = build_prompt(q)
        base_outputs.append(generate(model, tokenizer, prompt))

    # ---------- attach LoRA ----------
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # ---------- finetuned ----------
    for ex, base_out in zip(val, base_outputs):
        q, conclusion = parse_example(ex["text"])
        prompt = build_prompt(q)

        ft_out = generate(model, tokenizer, prompt)

        results.append(
            {
                "question": q,
                "ground_truth": conclusion,
                "base_output": base_out,
                "finetuned_output": ft_out,
                "base_number": extract_number(base_out),
                "finetuned_number": extract_number(ft_out),
                "ground_truth_number": extract_number(conclusion),
            }
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()