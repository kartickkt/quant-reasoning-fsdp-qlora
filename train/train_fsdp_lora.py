import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch


# ---------------------------------------------------------
# Args
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
    )
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    return parser.parse_args()


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
def load_train_dataset(path):
    if not os.path.exists(path):
        raise ValueError(f"Dataset not found at: {path}")

    dataset = load_dataset("json", data_files=path)["train"]

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    print(f"\nLoaded dataset with {len(dataset)} samples")
    print("Sample:", dataset[0])

    return dataset


def split_sample(text: str):
    q = text.split("Reasoning:")[0].replace("Question:", "").strip()
    r_c = text.split("Reasoning:")[1].strip()
    return q, r_c


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    print("\n===== BF16 LoRA + FSDP TRAINING START =====\n")

    dataset = load_train_dataset(args.train_data)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # Correct instruction-tuning tokenization (FINAL FIX)
    # -----------------------------------------------------
    def tokenize(example):
        question, answer = split_sample(example["text"])

        # Full conversation
        full_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=512,
        )

        # User prompt WITH generation boundary
        user_prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,  # 🔴 critical for correct masking
            truncation=True,
            max_length=512,
        )

        start = len(user_prompt_ids)

        labels = [-100] * start + full_ids[start:]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # -----------------------------------------------------
    # CPU-first load for FSDP sharding
    # -----------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    # -----------------------------------------------------
    # LoRA config (correct for Mistral)
    # -----------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # -----------------------------------------------------
    # TRAINING ARGS — STABLE DISTRIBUTED CONFIG
    # -----------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,

        # stability
        learning_rate=2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,

        # precision (FINAL choice)
        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",

        # FSDP
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="MistralDecoderLayer",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("\nStarting training loop...\n")
    trainer.train()

    print("\nSaving model...\n")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n===== TRAINING COMPLETE =====\n")


if __name__ == "__main__":
    main()