# Protocols and result coverage

## Classification

- Dataset: 1,692 training and 423 test outlets, revision
  `ea1f7ec37e90411502385ebb7fc5a1106f58a1c2`.
- Screenshots: maximum width 1024, vertical tile height 1400, at most six tiles,
  with full stored-image coverage.
- Prompts: `configs/classification_prompts.json`; outlet identity is not supplied
  separately, but branding may be visible in the image.
- APIs: `gpt-5.6-sol`, `claude-opus-5`, `moonshotai/kimi-k3`, and
  `qwen/qwen3.8-max-prime`; medium reasoning and 8192 output tokens.
- Local inference: 4-bit and thinking off. Main-test RunPod evaluations use
  64 output tokens; paired evaluations and main-test Ministral fine-tuned bias
  use 16. Exact settings are in `configs/local_evaluation.json`.
- Training: per-model/task settings in `configs/training/`; recorded
  runs use learning rate 5e-5 and 16 warmup steps.

## Label mappings

The released CSVs supply the targets in `true_label_factuality` and
`true_label_bias`. The evaluator uses these columns in preference to raw labels.
When loading raw labels, it applies the following mappings after trimming
whitespace and converting to uppercase:

| Raw factuality label(s) | Task label | Ordinal value |
|---|---|---:|
| `VERY LOW`, `LOW` | `LOW` | 0 |
| `MIXED` | `MIXED` | 1 |
| `MOSTLY FACTUAL`, `HIGH`, `VERY HIGH` | `HIGH` | 2 |

| Raw political-bias label(s) | Task label | Ordinal value |
|---|---|---:|
| `EXTREME LEFT`, `LEFT` | `LEFT` | 0 |
| `LEFT-CENTER` | `LEFT-CENTER` | 1 |
| `LEAST BIASED` | `LEAST BIASED` | 2 |
| `RIGHT-CENTER` | `RIGHT-CENTER` | 3 |
| `RIGHT`, `EXTREME RIGHT` | `RIGHT` | 4 |

Missing or `UNRATED` targets are not classes and are excluded from that task's
scoring. Every example in the released main and paired splits has valid targets
for both tasks. Model outputs must use the task labels shown above.

## Scoring

Macro P/R/F1 average over the fixed three/five task classes, with zero for
undefined class metrics. Accuracy is exact-label accuracy over all examples.
CSV scores are fractions; MAE/MSE use unnormalized ordinal indices:

- Factuality: `LOW`, `MIXED`, `HIGH` → 0–2.
- Bias: `LEFT`, `LEFT-CENTER`, `LEAST BIASED`, `RIGHT-CENTER`, `RIGHT` → 0–4.

Parsing accepts a whole label, ignoring case and outer whitespace. Failed or
unparseable outputs count as incorrect. Their absolute ordinal error is
`max(true_index, highest_index - true_index)`; squared error is its square.
Qwen API results use the final outcomes from the recorded recovery runs.
The saved random baseline averages 10,000 uniform-random repetitions, seed 42.

## Included results

| Evaluation | Included prediction artifacts |
|---|---|
| Main test, 423 outlets | All eight VLM methods, both tasks |
| Paired test, 133 outlets | All eight VLM methods, both tasks and conditions |
| Recognition probes, 212 outlets | All four frontier models, both tasks |

Exact coverage and source hashes are in `provenance/coverage.json`.
Qwen factuality uses distinct trained checkpoints for the main and paired
evaluations: `qwen35_factuality_main` and `qwen35_factuality`, respectively.
Both are included, and `scripts/paper.py` selects the recorded checkpoint and
token budget for each evaluation.

The paired set contains 43 adapter-training-seen outlets, including 42
pixel-identical unmasked training screenshots. Scores for the 90 unseen outlets
are included separately. Image pairing and training-membership flags are in
`data/paired/manifest.json`.

## Rating-reference analysis

The fixed 212-outlet seed-42 subset is stratified by cluster and joint labels.
Prompts explicitly elicit source recognition and recalled external ratings.
Primary references combine asserted and uncertain recollections in structured
reports; provider-summary-only mentions are excluded.

Opus/GPT annotations use V1; Kimi/Qwen use V2. Rubrics, reviewed ranges and
annotations are in `results/reference_audits/`. Cross-model comparisons are
therefore rubric-dependent. Qwen has 181 valid factuality and 178 valid bias
reports: reference rates use these denominators, while overall accuracy uses all
212 attempts per task. Two Qwen cases use the final outcome of a recorded rerun.

Reference-case accuracy scores the final classification against dataset labels.
It does not verify the recalled rating or establish a causal effect of recall.
`reference_metrics.csv` separately reports correct reference cases as a fraction
of all attempts and as a fraction of correct predictions.

## Provenance

`release_manifest.json` contains release-file checksums. `provenance/` records
original run metadata and source hashes; paths in derived metadata are normalized.
Original source hashes refer to the original artifacts, not their normalized exports.
