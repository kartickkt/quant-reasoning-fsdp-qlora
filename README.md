# Quantitative Reasoning Fine-Tuning with LoRA + FSDP

This project demonstrates an **end-to-end distributed LLM fine-tuning pipeline** that improves numerical financial reasoning in a 7B-parameter model.

## Goal

Show measurable reasoning improvement using:

* **LoRA parameter-efficient fine-tuning**
* **Multi-GPU FSDP training on AWS SageMaker**
* **Numerically correct evaluation on unseen problems**

The focus is **engineering credibility and scientific validation**, not prompt tricks.

---

## Dataset

Synthetic but structured quantitative reasoning tasks:

* Compounding
* Drawdown
* Sharpe comparison
* Volatility annualization
* Risk sizing & ranking

**900 training samples**
**Held-out unseen validation split**

Each example contains:

Question → Reasoning → Final numeric conclusion

---

## Training

* Base model: **Mistral-7B-Instruct**
* Method: **LoRA fine-tuning**
* Distributed strategy: **FSDP (multi-GPU)**
* Platform: **AWS SageMaker**

Training converged smoothly:

Loss decreased from **~1.9 → 0.04**, confirming stable distributed optimization.

---

## Evaluation

Evaluation measures **numeric correctness of the final answer**, not text similarity.

| Model           | Accuracy                  |
| --------------- | ------------------------- |
| Base Mistral-7B | **2.7%**                  |
| LoRA-finetuned  | **32.7%**                 |
| Improvement     | **+30 percentage points** |

This confirms **real quantitative reasoning gain**, not memorization.

---

## Key Takeaways

* Parameter-efficient fine-tuning can significantly improve **numerical reasoning** with small datasets.
* **FSDP-based distributed training** is practical and reproducible on cloud GPUs.
* Careful **metric design** is critical for evaluating reasoning models.

---

## Why this project exists

To demonstrate **production-credible LLM training, evaluation, and deployment skills** relevant to:

* Quantitative AI
* Applied LLM research
* Distributed ML systems

---

## Repo Structure (simplified)

```
data/        → dataset generation + validation split  
train/       → FSDP + LoRA training and evaluation scripts  
scripts/     → utilities and helpers  
launch_*.py  → SageMaker job launchers  
```

---

## Status

✅ Distributed training complete
✅ Finetuned adapter validated
✅ Measurable reasoning improvement demonstrated

This repository serves as a **practical reference for distributed LoRA fine-tuning in real ML workflows**.
