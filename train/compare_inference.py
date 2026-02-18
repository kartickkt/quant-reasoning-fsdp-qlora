import json
import os
import tarfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

VAL_PATH = "/opt/ml/input/data/val/quant_reasoning_val.jsonl"
ADAPTER_PATH = "/opt/ml/input/data/adapter"
OUTPUT_PATH = "/opt/ml/output/comparison_outputs.json"

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
NUM_SAMPLES = 10


def extract_adapter():
    tar_path = os.path.join(ADAPTER_PATH, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(ADAPTER_PATH)


def parse_question(text: str):
    return text.split("Question:")[1].split("\nReasoning:")[0].strip()


def build_prompt(q):
    return f"<s>[INST] {q} [/INST]"


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def load_first_n():
    data = []
    with open(VAL_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= NUM_SAMPLES:
                break
            data.append(json.loads(line))
    return data


def main():
    extract_adapter()

    val = load_first_n()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=False)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    results = []

    for ex in val:
        question = parse_question(ex["text"])
        prompt = build_prompt(question)

        base_out = generate(base_model, tokenizer, prompt)
        ft_out = generate(ft_model, tokenizer, prompt)

        results.append(
            {
                "question": question,
                "base_output": base_out,
                "finetuned_output": ft_out,
            }
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved comparison outputs.")


if __name__ == "__main__":
    main()