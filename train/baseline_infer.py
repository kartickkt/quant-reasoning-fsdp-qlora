print("### NEW BASELINE SCRIPT LOADED ###")

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DATA_PATH = Path("/opt/ml/input/data/training/quant_reasoning_dataset.jsonl")
OUT_PATH = Path("/opt/ml/output/data/baseline_outputs.json")


def load_prompts(n: int = 10):
    prompts = []

    with DATA_PATH.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break

            obj = json.loads(line)
            text = obj["text"]

            # Keep only the question line
            question = text.split("\n")[0].replace("Question: ", "")
            prompts.append(question)

    return prompts


def main():
    # --------------------------------------------------
    # Environment info (critical for SageMaker debugging)
    # --------------------------------------------------
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --------------------------------------------------
    # Load tokenizer  ✅ FIXED HERE
    # --------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,          # ← CRITICAL FIX for SageMaker tokenizer crash
        trust_remote_code=True,  # ← ensures correct tokenizer class loads
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    model.to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # Load prompts
    # --------------------------------------------------
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    results = []

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    print("Running baseline inference...")

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,   # deterministic baseline
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)

        results.append(
            {
                "prompt": p,
                "output": text,
            }
        )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    print(f"Saving outputs to {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print("Baseline inference completed successfully.")


if __name__ == "__main__":
    main()
