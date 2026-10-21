#!/usr/bin/env python3
"""Export provider-returned explanations from saved audit records, without API calls."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def extract(tag, response):
    if tag == "opus":
        blocks = [b for b in response.get("content", []) if b.get("type") == "thinking"]
        return "\n\n".join(b.get("thinking", "") for b in blocks).strip(), "anthropic_thinking", len(blocks)
    if tag == "openai":
        items = [item for item in response.get("output", []) if item.get("type") == "reasoning"]
        text = "\n\n".join(block.get("text", "") for item in items for block in item.get("summary", []))
        return text.strip(), "openai_reasoning_summary", len(items)
    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    text = message.get("reasoning")
    if isinstance(text, str) and text.strip():
        return text.strip(), "openrouter_reasoning", 1
    details = message.get("reasoning_details") or []
    text = "\n\n".join(d.get("text", "") for d in details if isinstance(d, dict) and isinstance(d.get("text"), str))
    return text.strip(), "openrouter_reasoning_details", len(details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run, output = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    if output.exists():
        parser.error("Choose a new output directory")
    output.mkdir(parents=True)
    inventory, all_rows, source_hashes = {}, [], {}
    for tag in ("opus", "openai", "kimi", "qwen"):
        source = run / f"{tag}.audit/predictions.jsonl"
        counter, task_counts, rows, digest = Counter(), defaultdict(Counter), [], hashlib.sha256()
        with source.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                response = record.get("details", {}).get("response") or {}
                text, kind, blocks = extract(tag, response)
                usage = response.get("usage") or {}
                token_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
                tokens = token_details.get("reasoning_tokens", token_details.get("thinking_tokens", 0)) or 0
                row = {
                    "model": tag, "response_model": response.get("model", ""),
                    "task": record["task"], "media_name": record["media_name"], "url": record["url"],
                    "sample_id": record["sample_id"], "true_label": record["true_label"],
                    "predicted_label": record["predicted_label"], "raw_final_answer": record["raw_text"],
                    "correct": record["correct"], "status": record["status"],
                    "explanation_available": bool(text), "explanation_kind": kind,
                    "reasoning_blocks": blocks, "reasoning_tokens": tokens,
                    "finish_reason": record.get("details", {}).get("finish_reason", ""),
                    "response_id": response.get("id", ""),
                    "source_journal": str(source), "source_line": line_number,
                    "provider_text": text,
                }
                rows.append(row)
                for counts in (counter, task_counts[record["task"]]):
                    counts["records"] += 1
                    counts["readable"] += bool(text)
                    counts["readable_with_valid_final_label"] += bool(text) and record["status"] == "ok"
                    counts["readable_without_valid_final_label"] += bool(text) and record["status"] != "ok"
                    counts["reasoning_blocks"] += blocks
                    counts["reasoning_tokens"] += tokens
        source_hashes[tag] = digest.hexdigest()
        inventory[tag] = {"counts": dict(counter), "tasks": {k: dict(v) for k, v in task_counts.items()}}
        all_rows.extend(rows)
        with (output / f"{tag}_explanations.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source_run": str(run),
        "source_journal_sha256": source_hashes, "models": inventory,
        "scope": "Only provider-returned text already present in saved responses. No new model calls or inferred explanations.",
        "limitations": ["Returned summaries and rationales are model self-reports, not verified causal accounts.",
                        "Missing text cannot be reconstructed from token counts or opaque signatures.",
                        "Text from failed or unparseable Qwen records is not a valid final classification."],
    }
    (output / "availability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ["# Saved provider explanations", "", report["scope"], "",
          "| Model | Records | Readable text | Readable + valid final label |",
          "| --- | ---: | ---: | ---: |"]
    for tag, item in inventory.items():
        c = item["counts"]
        md.append(f"| {tag} | {c['records']} | {c['readable']} | {c['readable_with_valid_final_label']} |")
    md += ["", "Each model has an `*_explanations.csv` file, including unavailable and failed records.",
           "Fields include outlet, task, truth, prediction, correctness, response ID, source journal line,",
           "availability, and the provider-returned text. Empty fields remain empty; nothing is fabricated.",
           "", "## Interpretation limits", ""]
    md += ["- " + text for text in report["limitations"]]
    md += ["", "GPT fields are API reasoning summaries; they are not complete internal traces.",
           "Opus has signed thinking blocks and thinking-token usage but empty text in this run.",
           "Kimi/Qwen expose text through OpenRouter; coverage alone does not establish explanation faithfulness.",
           "", "## Why Opus text is missing", "",
           "Anthropic's documentation states that Opus 5.5 defaults to `display: omitted`.",
           "Our recorded request enabled adaptive thinking but omitted the display field, which explains",
           "the empty thinking strings. Future diagnostic requests can opt into provider summaries:", "",
           '```json\n{"thinking": {"type": "adaptive", "display": "summarized"}}\n```', "",
           "Source: https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display",
           "(checked 2026-10-02). Display controls visibility; adaptive thinking can still be skipped",
           "for a request. The documentation describes summaries rather than raw internal reasoning.",
           "New requests would generate new decisions; the old empty fields cannot reconstruct their",
           "missing explanations. Our current GPT requests already ask for `reasoning.summary: auto`.",
           "", "## Selected GPT summaries", "",
           "These are illustrative, deliberately selected cases rather than a random sample. Text below is",
           "the provider's recorded summary, not an independent explanation of the model's mechanism.", ""]
    selected = [r for r in all_rows if r["model"] == "openai" and r["explanation_available"]
                and (r["media_name"], r["task"]) in (("North Lancaster News", "bias"), ("ABQ Times", "factuality"))]
    identity_examples = [r for r in all_rows if r["model"] == "openai" and r["task"] == "bias"
                         and "metric media" in r["provider_text"].lower()]
    selected += identity_examples[:2]
    for row in selected:
        md += [f"### {row['media_name']} — {row['task']}", "",
               f"Dataset: **{row['true_label']}**. Prediction: **{row['predicted_label']}**.", "",
               row["provider_text"], "",
               f"Source: `openai.audit/predictions.jsonl`, line {row['source_line']}; response `{row['response_id']}`.", ""]
    md += ["## Future diagnostic protocol", "",
           "For new diagnostic calls, request a concise decision justification with a label, up to three",
           "visible cues with tile/location references, whether the outlet is recognized, and a short",
           "statement distinguishing visible evidence from prior publisher knowledge. This requests",
           "observable evidence and a brief explanation, not hidden internal reasoning.", "",
           "Such responses would be new judgments under a different output protocol, not recovered",
           "explanations of the original label-only calls. Preserve the original runs and validate",
           "any claimed visual evidence against the saved images. Identity-only/masked-image controls",
           "are needed to test causal reliance; self-reports alone cannot establish contamination.", ""]
    (output / "REVIEW.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"models": inventory, "selected_examples": [
        {k: r[k] for k in ("media_name", "task", "true_label", "predicted_label", "provider_text")} for r in selected],
        "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
