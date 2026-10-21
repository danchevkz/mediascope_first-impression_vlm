# GPT-5.6 streaming recognition/recall probe

Runner: `eval_gpt56_summary_parallel.py`.

Use the same 212 unmasked test outlets and exact system/user probe prompts as
Opus 5. The final classification and all structured explanation/recognition/
recalled-rating fields remain required. This is an elicited knowledge probe,
separate from spontaneous-reference prevalence or verified training contamination.

## Approved missing-summary policy

Launch with `--stream --record-missing-summaries`. The request explicitly sends
`reasoning: {effort: high, summary: detailed}`. Collect both completed-response
summary arrays and actual `response.reasoning_summary_text` SSE events.

When OpenAI genuinely omits a summary, a valid structured answer is retained as
`ok_without_provider_summary`, with `provider_summary_available: false` and
`provider_summary_origin: unavailable`. Final explanations, encrypted reasoning,
and model-generated substitutes are never relabelled as native summaries. No
regeneration is triggered merely because the summary is missing.

Without the explicit flag, missing native summaries still stop the run. Invalid
structured output, model mismatch, incomplete response, or exhausted transport
failures stop new submissions under either policy.

## Reuse the fixed streaming pilot

`--seed-stream-diagnostic artifacts/evaluations/gpt56-summary-diagnostic-20261010`
imports all four `stream_detailed` pilot cases, including the missing-summary
case. Prompts, schemas, model, generation controls, case identities, source
plan hash, and SSE/final-payload agreement are validated. Other diagnostic
profiles are not imported. There is no per-case selection for summary presence
or correctness. Provenance is preserved in each record and the plan.

The full subset has 424 decisions. Four are reused, so 420 new requests remain.
The remaining four smoke cases run first, then twelve concurrent requests.
PNG input tiles are reused and hash-verified. Token/HTTP usage for new attempts
is recorded separately from imported pilot-response usage.

## Reporting

Primary comparable source-recognition statements can be judged in structured
final responses, available for every accepted record. Native summaries provide
additional evidence when observed and should be reported with explicit per-task
coverage. Missing summaries are unobserved evidence, not negative source/rating
recognition. Full-subset accuracy is reported only after 212 records per task.
