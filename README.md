# Quantitative Reasoning Fine-Tuning with LoRA + FSDP

This project demonstrates an **end-to-end distributed LLM fine-tuning pipeline** that improves **numerical financial reasoning** in a 7B-parameter model.

The focus is **real reasoning improvement and production-credible ML engineering**, not prompt tricks.

---

## Goal

Show measurable reasoning gains using:

- **LoRA** parameter-efficient fine-tuning  
- **Multi-GPU FSDP** distributed training on AWS SageMaker  
- **Numeric-correct evaluation** on unseen problems  

---

## Dataset

Synthetic but structured quantitative reasoning tasks:

- Compounding  
- Drawdown  
- Sharpe comparison  
- Volatility annualization  
- Risk sizing & ranking  

**900 training samples**  
**Held-out unseen validation split**

Each example:

Question → Reasoning → Final numeric conclusion


Matches the ground truth within tolerance.

---

## Key Takeaways

- Parameter-efficient fine-tuning can **substantially improve numerical reasoning** with small datasets.  
- **FSDP-based distributed training** is practical and reproducible on cloud GPUs.  
- **Metric design matters** when evaluating reasoning models.  

---

## Why This Project Exists

To demonstrate **production-credible ML engineering skills** relevant to:

- Quantitative AI  
- Applied LLM research  
- Distributed ML systems  

---

## Repo Structure (simplified)


Matches the ground truth within tolerance.

---

## Key Takeaways

- Parameter-efficient fine-tuning can **substantially improve numerical reasoning** with small datasets.  
- **FSDP-based distributed training** is practical and reproducible on cloud GPUs.  
- **Metric design matters** when evaluating reasoning models.  

---

## Why This Project Exists

To demonstrate **production-credible ML engineering skills** relevant to:

- Quantitative AI  
- Applied LLM research  
- Distributed ML systems  

---

## Repo Structure (simplified)

data/ → dataset generation + validation split
train/ → FSDP + LoRA training and evaluation scripts
launch_*.py → SageMaker job launchers


---

## Status

- Distributed training completed  
- Finetuned adapter validated  
- Measurable reasoning improvement demonstrated  

This repository serves as a **practical reference for distributed LoRA fine-tuning in real ML workflows**.

---

## Next Steps

- Evaluate on **real financial datasets**  
- Expand validation benchmark  
- Deploy **autoscaling inference endpoint** for production use  
