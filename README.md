# Quantitative Reasoning Fine-Tuning with LoRA + FSDP

This project demonstrates an **end-to-end distributed LLM fine-tuning and deployment pipeline** that improves numerical financial reasoning in a 7B-parameter model.

The objective is measurable reasoning improvement combined with production-grade ML systems design.

---

## Key Results

Evaluation measures **numeric correctness of final answers**, not text similarity.

| Model           | Accuracy |
|-----------------|----------|
| Base Mistral-7B | **2.7%** |
| LoRA-finetuned  | **32.7%** |
| **Improvement** | **+30 percentage points** |

This reflects genuine quantitative reasoning improvement.

---

## Example Tasks

### Drawdown Calculation

**Question:**  
A portfolio falls from 177 to 137. What is the maximum drawdown percentage?

**Finetuned model:**  
\[
(137 - 177) / 177 \approx -22.9\%
\]

Correctly applies the drawdown formula and reaches the numeric conclusion.

---

### Volatility Annualization

**Question:**  
Daily volatility = 2.85%. What is annualized volatility?

**Finetuned model:**  
\[
0.0285 \times \sqrt{252} \approx 45.5\%
\]

Matches the ground truth within tolerance.

---

## Training Architecture

- **Base Model:** Mistral-7B-Instruct  
- **Fine-Tuning:** LoRA (parameter-efficient adapters)  
- **Distributed Training:** Multi-GPU FSDP  
- **Precision:** BF16 training for numerical stability  
- **Platform:** AWS SageMaker (ml.g5.12xlarge)

Training loss decreased from **~1.9 → 0.04**, confirming stable distributed optimization.

---

## Inference & Deployment

- Real-time SageMaker endpoint using HuggingFace DLC  
- Adapter-based loading (LoRA applied at runtime)  
- Deterministic structured JSON output for numeric answers  
- FP16 inference aligned with A10G GPU optimization  

**Benchmark:**

- Steady-state latency ≈ **0.73 seconds per request**
- ~1 request/sec safe capacity per instance

---

## Autoscaling Strategy

Autoscaling configured via Application Auto Scaling:

- Min capacity: 1 instance  
- Max capacity: configurable (modeled up to 10 instances)  
- Target tracking based on `InvocationsPerInstance`  
- Cooldown tuning to balance burst handling and cost control  

Capacity planning performed using latency benchmarking and per-instance throughput estimation.

---

## Dataset

Synthetic but structured quantitative reasoning tasks covering:

- Compounding
- Drawdown
- Sharpe comparison
- Volatility annualization
- Risk sizing & ranking

- **900 training samples**
- Held-out unseen validation split
- Format: `Question → Reasoning → Final numeric answer`

---

## Repository Structure

- `data/` – dataset generation and validation split  
- `train/` – FSDP + LoRA training and evaluation pipeline  
- `scripts/` – deployment, inference, autoscaling  
- `launch_*.py` – SageMaker job launch scripts  

---

## Engineering Takeaways

- Parameter-efficient fine-tuning substantially improves domain-specific reasoning.  
- FSDP enables scalable distributed training of 7B models on cloud GPUs.  
- Numeric correctness is a more meaningful metric than text similarity for reasoning tasks.  
- Latency benchmarking and autoscaling are essential for production-ready LLM systems.

---

## Status

- Distributed training complete  
- Finetuned adapter validated  
- Real-time endpoint deployed and benchmarked  
- Autoscaling strategy modeled and configured  

This repository demonstrates a complete workflow from distributed training to scalable inference deployment.