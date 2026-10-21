# Completed explicit-recognition probes: reference and accuracy comparison

All four models attempted the same 212 outlets for factuality and bias. This is
the elicited recognition condition, not the original label-only benchmark or
the neutral-rationale/local-model condition.

| Model | Task | Rating references | Accuracy among reference cases | Overall accuracy | Valid-report coverage |
|---|---|---:|---:|---:|---:|
| Claude Opus 5 | Factuality | 49.06% | 83.65% | 90.09% | 100.00% |
| Claude Opus 5 | Bias | 56.13% | 83.19% | 73.58% | 100.00% |
| GPT-5.6 Sol | Factuality | 45.75% | 88.66% | 78.77% | 100.00% |
| GPT-5.6 Sol | Bias | 20.28% | 88.37% | 58.96% | 100.00% |
| Kimi K3 | Factuality | 36.79% | 83.33% | 74.53% | 100.00% |
| Kimi K3 | Bias | 23.58% | 78.00% | 55.66% | 100.00% |
| Qwen3.8 Max Prime | Factuality | 16.57% | 76.67% | 37.26% | 85.38% |
| Qwen3.8 Max Prime | Bias | 15.73% | 75.00% | 32.55% | 83.96% |

## Denominators and interpretation

- Rating-reference rates are manually coded asserted + uncertain recollections
  among valid structured reports. Either a factuality or bias rating can qualify
  within either task, once per decision. Native-only reasoning mentions are excluded.
- Opus/GPT retain earlier V1 coding. Kimi/Qwen use the clarified conservative V2
  attribution rubric with exact evidence; older-model re-audits are pending. This
  is a provisional cross-model comparison, not a method-matched ranking.
- Accuracy among reference cases evaluates final classifications against dataset
  labels. It does not verify recalled ratings, training membership, or causal
  memory use, and the models' reference-case subsets differ.
- Qwen has 181 valid factuality and 178 valid bias reports. Its reference rates
  use those denominators, while overall accuracy uses all 212 attempted cases per
  task and counts failed outcomes as incorrect. Missing reports are unobserved
  disclosure cases, not no-reference evidence. Valid coverage is shown explicitly.
- Qwen's final outcomes include 57 filters, 7 token-limit failures, and one
  malformed schema-echo response. Two original token-limit cases were rerun at
  the user's request; both generations remain audited, with the later outcome
  used regardless of correctness. Total Qwen generation requests: 426.
- Reference claims and possible-exposure admissions are elicited self-reports,
  not verified contamination prevalence. One LLM reviewer; no independent
  inter-rater reliability is claimed.

## Source audits

- Opus/GPT: `artifacts/audits/current_probe_manual_review_complete_20261010/`
- Kimi: `artifacts/audits/kimi_probe_manual_review_v2_20261010/`
- Qwen: `artifacts/audits/qwen_probe_manual_review_v2_20261010/`

Each audit retains snapshots/source hashes and per-decision manual annotations.
Kimi/Qwen evidence excerpts and source hashes were validated on aggregation.
