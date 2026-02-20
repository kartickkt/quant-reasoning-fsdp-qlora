import json
import logging
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
MAX_RETRIES = 2

SYSTEM_INSTRUCTION = (
    "You are a precise math reasoning assistant. "
    "Return ONLY valid JSON in the format: "
    '{"answer": <number>}. '
    "Do not include explanations."
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------
# Helper: Robust JSON extraction
# ---------------------------------------------------------
def _extract_json(text: str):
    """
    Extract last valid JSON object from model output.
    Robust against prefix/suffix noise.
    """
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    for match in reversed(matches):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except Exception:
            continue

    return None


# ---------------------------------------------------------
# SageMaker: Model load (runs ONCE at container startup)
# ---------------------------------------------------------
def model_fn(model_dir, context=None):
    logger.info("Loading tokenizer and LoRA model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    logger.info("Model + tokenizer loaded successfully.")
    return {"model": model, "tokenizer": tokenizer}


# ---------------------------------------------------------
# Core generation helper
# ---------------------------------------------------------
def _generate_json_response(prompt: str, model, tokenizer):

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=TEMPERATURE,
            top_p=1.0,
        )

    # Only decode newly generated tokens
    generated_tokens = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return text


# ---------------------------------------------------------
# SageMaker: Per-request inference
# ---------------------------------------------------------
def predict_fn(data, context):
    model = context["model"]
    tokenizer = context["tokenizer"]

    prompt = data.get("prompt", "")
    if not prompt:
        return {"error": "Empty prompt"}

    logger.info("Received inference request.")

    start_time = time.time()

    # -----------------------------------------------------
    # Retry loop for JSON correctness
    # -----------------------------------------------------
    raw_text = None
    for attempt in range(MAX_RETRIES + 1):
        raw_text = _generate_json_response(prompt, model, tokenizer)
        parsed = _extract_json(raw_text)

        if parsed is not None:
            latency = time.time() - start_time
            logger.info("Valid JSON generated.")
            return {
                "answer": parsed["answer"],
                "latency_sec": round(latency, 4),
            }

        logger.warning(f"JSON parse failed (attempt {attempt + 1}). Retrying...")

    # -----------------------------------------------------
    # Final fallback
    # -----------------------------------------------------
    latency = time.time() - start_time
    logger.error("Failed to produce valid JSON after retries.")

    return {
        "answer": None,
        "error": "Model did not return valid JSON",
        "raw_output": raw_text,
        "latency_sec": round(latency, 4),
    }
