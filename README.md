# Quantitative Reasoning Fine-Tuning with LoRA + FSDP

This project demonstrates an **end-to-end distributed LLM fine-tuning pipeline** that improves **numerical financial reasoning** in a 7B-parameter model.

The focus is on **real reasoning improvement and production-credible ML engineering**, not prompt engineering tricks.

---

## Key Results

Evaluation measures **numeric correctness of the final answer**, not text similarity.

| Model           | Accuracy                  |
| --------------- | ------------------------- |
| Base Mistral-7B | **2.7%**                  |
| LoRA-finetuned  | **32.7%**                 |
| **Improvement** | **+30 percentage points** |

This demonstrates **true quantitative reasoning gain**, not memorization.

---

## Example Outputs

### Drawdown Calculation

**Question**
A portfolio falls from 177 to 137. What is the maximum drawdown percentage?

**Base model**
Produces partial reasoning but **fails to compute the final percentage**.

**Finetuned model**
(137 − 177) / 177 ≈ −22.9%

Correctly applies the drawdown formula and reaches the numeric conclusion.

---

### Volatility Annualization

**Question**
Daily volatility = 2.85%. What is annualized volatility?

**Base model**
Uses an **incorrect compounding formula** and does not finish the calculation.

**Finetuned model**
0.0285 × sqrt(252) ≈ 45.5%

Matches the ground truth within tolerance.

---

## Method

* **Base model:** Mistral-7B-Instruct
* **Fine-tuning:** LoRA (parameter-efficient)
* **Distributed training:** Multi-GPU **FSDP**
* **Platform:** AWS SageMaker

Training converged smoothly:

> Loss decreased from **~1.9 → 0.04**, confirming stable distributed optimization.

---

## Dataset

Synthetic but structured **quantitative reasoning tasks** covering:

* Compounding
* Drawdown
* Sharpe comparison
* Volatility annualization
* Risk sizing & ranking

**900 training samples** with a **held-out unseen validation split**.

Each example follows:

```
Question → Reasoning → Final numeric conclusion
```

---

## Repository Structure

* **data/** – dataset generation and validation split
* **train/** – FSDP + LoRA training, inference, and evaluation pipeline
* **launch_*.py** – AWS SageMaker job launch scripts

---

## Key Takeaways

* Parameter-efficient fine-tuning can **substantially improve numerical reasoning** with small datasets.
* **FSDP-based distributed training** is practical and reproducible on cloud GPUs.
* **Metric design is critical** when evaluating reasoning models.

---

## Status

* Distributed training completed
* Finetuned adapter validated
* Measurable reasoning improvement demonstrated

This repository serves as a **practical reference for distributed LoRA fine-tuning in real ML workflows**.

---

## Next Steps

* Evaluate on **real financial datasets**
* Expand the validation benchmark
* Deploy an **autoscaling inference endpoint** for production use

This project is part of an ongoing effort to build **production-ready quantitative reasoning systems with distributed LLM training**.
