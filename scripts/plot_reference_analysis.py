#!/usr/bin/env python3
"""Plot rating-reference analysis using explicit, count-based denominators."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("opus5", "gpt56", "kimi", "qwen")
LABELS = ("Claude\nOpus 5", "GPT-5.6\nSol", "Kimi K3", "Qwen3.8\nMax Prime")


def plot_values(row):
    """Derive all bars from counts rather than multiplying rounded percentages."""
    n, valid, refs, correct, correct_refs = (int(row[k]) for k in
                                           ("n", "valid_reports", "references", "correct", "correct_references"))
    if not (0 < valid <= n and 0 <= correct <= valid and 0 <= correct_refs <= refs <= valid
            and correct_refs <= correct and correct - correct_refs <= valid - refs):
        raise ValueError("Inconsistent reference-analysis counts")
    rates = {
        "reference_rate_valid_reports": refs / valid,
        "overall_accuracy": correct / n,
        "correct_references_fraction_all_attempts": correct_refs / n,
        "reference_share_of_correct_predictions": correct_refs / correct if correct else None,
    }
    for field, value in rates.items():
        if value is not None and row.get(field) not in (None, ""):
            if not math.isclose(float(row[field]), value, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Stored {field} differs from its count-based denominator")
    return {"model": row["model"], "task": row["task"], "rubric": row.get("rubric", ""),
            "n": n, "valid_reports": valid, "references": refs, "correct": correct,
            "correct_references": correct_refs,
            **{field + "_percent": 100 * value if value is not None else None for field, value in rates.items()}}


def render(input_csv, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    with input_csv.open(newline="") as handle:
        records = [plot_values(row) for row in csv.DictReader(handle)]
    index = {(r["model"], r["task"]): r for r in records}
    if len(records) != 8 or set(index) != {(model, task) for model in MODELS for task in ("factuality", "bias")}:
        raise ValueError("Expected exactly four models and two tasks")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    blue, orange, light = "#2A78D6", "#EB6834", "#AACFEE"
    for ax, task, title in zip(axes, ("factuality", "bias"), ("Factuality", "Political Bias")):
        data = [index[(model, task)] for model in MODELS]
        x = list(range(4))
        ax.bar([i-.22 for i in x], [r["reference_rate_valid_reports_percent"] for r in data], width=.23, color=blue)
        ax.bar([i+.13 for i in x], [r["overall_accuracy_percent"] for r in data], width=.36, color=light)
        ax.bar([i+.13 for i in x], [r["correct_references_fraction_all_attempts_percent"] for r in data],
               width=.22, color=orange)
        ax.set_title(title)
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#CCCCCC")
    axes[0].set_ylabel("Percentage")
    fig.legend([Patch(color=blue), Patch(color=orange), Patch(color=light)],
               ["Rating references\n/ valid reports", "Correct with references\n/ all attempts", "All correct\n/ all attempts"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .005), fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, 1))
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(prefix.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {prefix}.{{pdf,svg,png,csv}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/reproduced/reference_metrics.csv")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "work/figures/figure2")
    args = parser.parse_args()
    render(args.input, args.output_prefix)
