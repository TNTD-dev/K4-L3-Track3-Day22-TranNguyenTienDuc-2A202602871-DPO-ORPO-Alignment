"""Recompute submission statistics from collected evidence; no model/API calls."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lab22 import data as D
from lab22 import judge as J


def paired_accuracy(before: list[dict], after: list[dict], seed: int = 42) -> dict:
    if len(before) != len(after) or not before:
        raise ValueError("Before/after predictions must have the same nonzero length")
    if any(a["prompt"] != b["prompt"] or a["answer"] != b["answer"] for a, b in zip(before, after)):
        raise ValueError("Before/after evaluation questions must match")
    differences = [int(b["correct"]) - int(a["correct"]) for a, b in zip(before, after)]
    n = len(before)
    p0, p1 = sum(r["correct"] for r in before) / n, sum(r["correct"] for r in after) / n
    low, high = J.bootstrap_ci(differences, n_boot=10000, seed=seed)
    gains, losses = differences.count(1), differences.count(-1)
    discordant = gains + losses
    mcnemar_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2**discordant) if discordant else 1.0
    return {"n": n, "before": p0, "after": p1, "delta": p1 - p0,
            "before_se": math.sqrt(p0 * (1 - p0) / n), "after_se": math.sqrt(p1 * (1 - p1) / n),
            "gains": gains, "losses": losses, "paired_bootstrap_ci95": [low, high], "mcnemar_exact_p": mcnemar_p}


def main() -> None:
    import pyarrow.parquet as pq

    eval_dir = REPO / "data/eval"
    rows = [json.loads(line) for line in (eval_dir / "side_by_side.jsonl").read_text().splitlines() if line.strip()]
    train = pq.read_table(REPO / "data/pref/train.parquet").to_pylist()
    held = pq.read_table(REPO / "data/pref/eval.parquet").to_pylist()
    D.assert_disjoint(train, held)
    predictions = json.loads((eval_dir / "grpo_predictions.json").read_text())
    summary = json.loads((eval_dir / "judge_summary.json").read_text())
    rm = json.loads((eval_dir / "judge_results_rm.json").read_text())
    api = json.loads((eval_dir / "judge_results_api.json").read_text())
    recomputed = J.summarize([r for r in rm["records"] if r["category"] == "heldout"], seed=42)
    if recomputed != summary["heldout"]:
        raise ValueError("Saved RM held-out summary differs from recomputed verdicts")
    report = {
        "train_pairs": len(train), "eval_pairs": len(held), "prompt_overlap": 0,
        "distinct_train_prompts": len({D.normalize_prompt(r['prompt'][0]['content']) for r in train}),
        "distinct_eval_prompts": len({D.normalize_prompt(r['prompt'][0]['content']) for r in held}),
        "outputs": {c: {"n": len(group), "identical_answers": sum(r["sft"] == r["dpo"] for r in group)}
                    for c in ("heldout", "helpfulness", "safety")
                    for group in [[r for r in rows if r["category"] == c]]},
        "grpo_paired": paired_accuracy(predictions["before"], predictions["after"]),
        "cross_judge_all": J.agreement(rm["records"], api["records"]),
        "cross_judge_heldout": J.agreement([r for r in rm["records"] if r["category"] == "heldout"],
                                           [r for r in api["records"] if r["category"] == "heldout"]),
        "sweep": [],
    }
    for folder in sorted((REPO / "adapters").glob("dpo-b*")):
        m = json.loads((folder / "dpo_metrics.json").read_text())
        report["sweep"].append({"beta": m["beta"], "margin": m["eval_reward_gap"],
                                "margin_div_beta": m["eval_reward_gap"] / m["beta"],
                                "accuracy": m["eval_reward_accuracy"]})
    (REPO / "submission/analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # The component plot is derived from real logged values, never reconstructed rewards.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads((eval_dir / "grpo_history.json").read_text())
    logs = [r for r in history if "reward" in r]
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label in (("reward", "total"), ("rewards/correctness_reward/mean", "correctness"),
                       ("rewards/format_reward/mean", "format")):
        ax.plot([r["step"] for r in logs], [r[key] for r in logs], marker="o", label=label)
    ax.set(xlabel="GRPO step", ylabel="logged mean reward", title="GRPO components: real training logs")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPO / "submission/screenshots/08b-grpo-components.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
