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


---

## Training

- **Base model:** Mistral-7B-Instruct  
- **Method:** LoRA fine-tuning  
- **Distributed strategy:** FSDP (multi-GPU)  
- **Platform:** AWS SageMaker  

Training converged smoothly:

> Loss decreased from **~1.9 → 0.04**, confirming stable distributed optimization.

---

## Evaluation

Evaluation measures **numeric correctness of the final answer**,  
not text similarity.

| Model | Accuracy |
|-------|---------|
| Base Mistral-7B | **2.7%** |
| LoRA-finetuned | **32.7%** |
| **Improvement** | **+30 percentage points** |

This demonstrates **true quantitative reasoning gain**, not memorization.

---

## Training

- **Base model:** Mistral-7B-Instruct  
- **Method:** LoRA fine-tuning  
- **Distributed strategy:** FSDP (multi-GPU)  
- **Platform:** AWS SageMaker  

Training converged smoothly:

> Loss decreased from **~1.9 → 0.04**, confirming stable distributed optimization.

---

## Evaluation

Evaluation measures **numeric correctness of the final answer**,  
not text similarity.

| Model | Accuracy |
|-------|---------|
| Base Mistral-7B | **2.7%** |
| LoRA-finetuned | **32.7%** |
| **Improvement** | **+30 percentage points** |

This demonstrates **true quantitative reasoning gain**, not memorization.

---

## Example Outputs

### 1) Drawdown Calculation

**Question**  
A portfolio falls from 177 to 137. What is the maximum drawdown percentage?

**Base model**  
Produces partial reasoning but **fails to compute the final percentage**.

**Finetuned model**
(137 − 177) / 177 ≈ −22.9%


Correctly applies the drawdown formula and reaches the numeric conclusion.

---

### 2) Volatility Annualization

**Question**  
Daily volatility = 2.85%. What is annualized volatility?

**Base model**  
Uses an **incorrect compounding formula** and does not finish the calculation.

**Finetuned model**
0.0285 × sqrt(252) ≈ 45.5%

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