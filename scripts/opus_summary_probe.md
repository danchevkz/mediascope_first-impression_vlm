# Opus 5 native-summary / explicit source-recognition probe

Runner: `eval_opus_summary_parallel.py`.

## Fixed protocol

- `claude-opus-5` only; no automatic version substitution.
- Same seed-42, cluster/joint-label-stratified 212 unmasked test outlets.
- Both tasks: 424 case-specific requests total. No training.
- Existing saved full-page PNG tiles are reused and verified against their hashes.
- Exact request settings: `thinking: {type: adaptive, display: summarized}`,
  `output_config: {effort: high}` by default, `max_tokens: 8192`.
- Requested structured final assessment: label, concise evidence summary,
  recognized outlet/network, recalled factuality/bias ratings and their external
  source, reported prior-rating influence, possible-contamination self-report,
  and a concise basis for that self-report.
- Null/unknown are explicitly valid. No training-membership claim is required.
- The secondary `possible_contamination` self-report is requested but, if omitted,
  its absence is preserved and counted as `not_reported`. No model answer is
  invented. This does not relax the native-summary, label, explanation, source/
  recalled-rating-field, or prior-rating-influence validation requirements.

This is an **elicited recognition/recall condition**. Its reference rates must not
be pooled with the earlier neutral-rationale condition. A model's declaration of
possible contamination does not verify its training data or causal memory use.

## Observable summaries

Anthropic documents that Opus 5 defaults to omitted thinking text, and explicitly
supports `display: summarized`:
https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display

The same documentation states that adaptive thinking may skip a request even
with summarized display. No effort level guarantees a thinking block:
https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost#effort-levels

Therefore the runner **verifies actual response text**, not only request settings
or thinking-token usage. It never substitutes the final explanation for a native
provider summary. A missing native summary, invalid final assessment, model
mismatch, HTTP failure, or truncated response stops submission of new requests.
Already in-flight responses are saved. Failed outputs are never automatically
regenerated to select a better answer or a more favorable disclosure.

With the explicit `--record-missing-summaries` flag, valid structured responses
are retained as `ok_without_provider_summary` when Anthropic omits native text.
`provider_summary_available` is false and the origin is `unavailable`; no text is
invented and no retry is triggered to obtain a preferred summary. Native-summary
coverage is reported separately. All other validation failures still stop new
calls. This is the same observation policy used in the streaming GPT probe.

## Parallelism and smoke gate

- Eight initial calls cover HIGH/MIXED/LOW and LEFT examples, both tasks.
- Smoke concurrency is at most four. Every smoke result must pass native-summary
  and structured-output checks before the full sweep starts.
- Default full-sweep concurrency: twelve independent requests, globally paced
  at a minimum 0.5 seconds between submissions.
- Only transient transport/429/5xx failures are retried, at most three times;
  every attempt is journaled. Retry-After imposes a shared cooldown.
- Smoke cases are part of the full subset; they are not paid for a second time.

## Durable outputs

The run directory contains `plan.json`, `requests.jsonl`, `predictions.jsonl`,
`summary.json`, and atomic per-case files under `records/`. The per-case files
are the authoritative resume source. Summary metrics are full-subset accuracies
only when all 212 decisions for the task are recorded.

The plan fingerprints inputs, prompts, and code. Operational concurrency and
pacing may change on resume; input/prompt/model changes require a new run.
Use `--smoke-only` to stop after the gate, then the same command with `--resume`
to continue a successful smoke. Failed summary/output checks require inspection;
the runner refuses to silently resample them.
