#!/usr/bin/env python3
"""Triage saved provider explanations for manual review; makes no model calls.

Pattern matches are overlapping wording indicators, not a contamination detector
or an estimate of how often a model actually relied on prior knowledge.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


PATTERNS = {
    "direct_rating_reference": r"\b(?:MBFC|media bias[ /-]*fact[ /-]*check(?:ing)?(?: site)?)\s+(?:rates?|rated|classifies|lists?|indicates|labels|categorizes|considers)\b",
    "metric_media_family": r"metric media|local\s*labs|timpone|pink[ -]?slime",
    "prior_knowledge_wording": r"\bI (?:know|recognize|recall|remember)\b|prior knowledge|background knowledge|known for|known to be",
    "analyst_neutrality_wording": r"(?:I should aim for|important to maintain|my goal.{0,30})(?:.{0,30})(?:least biased|neutral|unbiased)(?:.{0,40})(?:approach|perspective|way|analysis|stance)",
    "benchmark_speculation": r"\bbenchmark\b|dataset.{0,30}(?:ground truth|label)|(?:expected|intended) (?:answer|label)|simulated (?:site|environment)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--explanations-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    source, output = Path(args.explanations_dir), Path(args.output_dir)
    if output.exists():
        ap.error("Choose a new output directory")
    output.mkdir(parents=True)
    totals, matches = {}, []
    for model in ("opus", "openai", "kimi", "qwen"):
        counts = Counter()
        with (source / f"{model}_explanations.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                counts["records"] += 1
                text = row["provider_text"]
                if row["status"] != "ok" or not text.strip():
                    continue
                counts["valid_records_with_readable_text"] += 1
                for category, pattern in PATTERNS.items():
                    hit = re.search(pattern, text, re.IGNORECASE)
                    if not hit:
                        continue
                    counts[category] += 1
                    matches.append({
                        "model": model, "category": category,
                        **{key: row[key] for key in ("media_name", "task", "true_label", "predicted_label",
                                                    "correct", "response_id", "source_journal", "source_line")},
                        "excerpt": text[max(0, hit.start() - 80):min(len(text), hit.end() + 240)],
                    })
        totals[model] = dict(counts)
    result = {
        "counts": totals, "patterns": PATTERNS,
        "inclusion": "Only records with both a valid final label and nonempty provider-returned text.",
        "interpretation": "Overlapping lexical indicators for manual triage. Counts are not verified memory-use, task-confusion, or contamination rates.",
    }
    (output / "triage_counts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "category", "media_name", "task", "true_label",
                                                   "predicted_label", "correct", "response_id", "source_journal",
                                                   "source_line", "excerpt"])
        writer.writeheader()
        writer.writerows(matches)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
