import random
import math
from pathlib import Path
import json
import argparse


# ---------- Reasoning generators ----------

def sharpe_example():
    r1, v1 = random.uniform(0.05, 0.25), random.uniform(0.05, 0.30)
    r2, v2 = random.uniform(0.05, 0.25), random.uniform(0.05, 0.30)

    s1, s2 = r1 / v1, r2 / v2
    better = "Strategy A" if s1 > s2 else "Strategy B"

    question = (
        f"Two strategies have annual returns of {r1:.2%} and {r2:.2%}, "
        f"with volatilities of {v1:.2%} and {v2:.2%}. "
        "Which strategy offers the stronger risk-adjusted performance?"
    )

    reasoning = (
        "Risk-adjusted performance can be approximated using the Sharpe ratio, "
        "defined as return divided by volatility. "
        f"For Strategy A, {r1:.2f} / {v1:.2f} ≈ {s1:.2f}. "
        f"For Strategy B, {r2:.2f} / {v2:.2f} ≈ {s2:.2f}. "
        f"Comparing these values shows that {better} delivers more return per unit of risk."
    )

    conclusion = f"{better} has the higher Sharpe ratio and the stronger risk-adjusted profile."
    return question, reasoning, conclusion


def compounding_example():
    r = random.uniform(0.01, 0.05)
    years = random.randint(2, 10)
    growth = (1 + r) ** years

    question = (
        f"If capital compounds at {r:.2%} per year for {years} years, "
        "what multiple of the original capital is reached?"
    )

    reasoning = (
        "Compound growth is calculated as (1 + return) raised to the number of years. "
        f"That gives (1 + {r:.4f})^{years} ≈ {growth:.2f}. "
        "This reflects how steady percentage gains accumulate over time."
    )

    conclusion = f"The capital grows to roughly {growth:.2f} times the starting value."
    return question, reasoning, conclusion


def volatility_scaling_example():
    daily_vol = random.uniform(0.005, 0.03)
    annual_vol = daily_vol * math.sqrt(252)

    question = (
        f"A strategy has daily volatility of {daily_vol:.2%}. "
        "What is the approximate annualized volatility?"
    )

    reasoning = (
        "Volatility scales with the square root of time under standard assumptions. "
        f"Multiplying {daily_vol:.4f} by sqrt(252) gives ≈ {annual_vol:.2%}. "
        "This converts daily variability into an annual risk estimate."
    )

    conclusion = f"The annualized volatility is approximately {annual_vol:.2%}."
    return question, reasoning, conclusion


def drawdown_example():
    peak = random.uniform(100, 200)
    trough = peak * random.uniform(0.6, 0.95)
    dd = (trough - peak) / peak

    question = (
        f"A portfolio falls from {peak:.0f} to {trough:.0f}. "
        "What is the maximum drawdown percentage?"
    )

    reasoning = (
        "Drawdown measures the percentage decline from peak to trough. "
        f"The change is ({trough:.0f} − {peak:.0f}) / {peak:.0f} ≈ {dd:.2%}. "
        "This quantifies the severity of the loss during the downturn."
    )

    conclusion = f"The maximum drawdown is about {dd:.2%}."
    return question, reasoning, conclusion


def risk_ranking_example():
    sharpe_a = random.uniform(0.3, 1.5)
    sharpe_b = random.uniform(0.3, 1.5)
    better = "Strategy A" if sharpe_a > sharpe_b else "Strategy B"

    question = (
        f"Two strategies have Sharpe ratios of {sharpe_a:.2f} and {sharpe_b:.2f}. "
        "Which is more attractive from a risk-adjusted perspective?"
    )

    reasoning = (
        "Higher Sharpe ratios indicate more return per unit of risk. "
        f"Comparing {sharpe_a:.2f} and {sharpe_b:.2f} shows that {better} "
        "offers the more efficient risk-return trade-off."
    )

    conclusion = f"{better} is preferable on a risk-adjusted basis."
    return question, reasoning, conclusion


def position_sizing_example():
    capital = random.uniform(50_000, 200_000)
    risk_pct = random.uniform(0.005, 0.02)
    loss = capital * risk_pct

    question = (
        f"If a trader risks {risk_pct:.2%} of a {capital:,.0f} account on a position, "
        "how much capital is exposed to loss?"
    )

    reasoning = (
        "Position risk equals total capital multiplied by the risk percentage. "
        f"{capital:,.0f} × {risk_pct:.4f} ≈ {loss:,.0f}. "
        "This defines the maximum acceptable loss for disciplined risk control."
    )

    conclusion = f"Approximately {loss:,.0f} of capital is at risk."
    return question, reasoning, conclusion


GENERATORS = [
    sharpe_example,
    compounding_example,
    volatility_scaling_example,
    drawdown_example,
    risk_ranking_example,
    position_sizing_example,
]


# ---------- Main dataset build ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="quant_reasoning_dataset.jsonl")
    parser.add_argument("--samples_per_type", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_path = Path(__file__).parent / args.output
    samples = []

    for gen in GENERATORS:
        for _ in range(args.samples_per_type):
            q, r, c = gen()
            text = f"Question: {q}\nReasoning: {r}\nConclusion: {c}"
            samples.append({"text": text})

    random.shuffle(samples)

    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"✅ Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()