# Kimi K3 / Qwen3.8 Max Prime explicit-recognition probes

Runner: `eval_openrouter_summary_parallel.py`.

## Fixed inference protocol

- Original benchmark model IDs: `moonshotai/kimi-k3` and
  `qwen/qwen3.8-max-prime`. Other Qwen aliases/snapshots are not substituted.
- Same 212 unmasked outlets, both tasks, 424 decisions per model, as GPT/Opus.
- Reuses the authoritative Opus plan and saved full-page tiles, validating case,
  subset, source-image, inventory, and submitted-tile hashes before any calls.
- Identical system/user recognition prompts and JSON fields. Outlet names and
  URLs are not supplied as text. No prompt change in response to coding concerns.
- High reasoning effort, `reasoning.exclude: false`, 8192 combined output-token
  cap, strict JSON schema, and `provider.require_parameters: true`.
- Default six concurrent requests per model, globally paced at one second within
  each model process. Eight-case smoke gate; no full sweep until structured
  responses have validated for both tasks. Smoke responses are part of the run.
- Raw responses, returned provider/model IDs, usage, finish reasons, readable
  reasoning, full final text, and parsed reports are retained. No response-healing
  plugin or model-generated replacement trace is used.

## Observation and failure policy

- `--record-missing-summaries` keeps otherwise valid reports with empty native
  reasoning, explicit availability/origin flags, and no re-generation.
- OpenRouter may expose reasoning text rather than a condensed summary; the
  returned channel kind is recorded. Encrypted/signature-only blocks do not count
  as readable reasoning. Final explanations never replace missing native text.
- `--record-provider-refusals` retains actual provider content filters/refusals as
  terminal case outcomes, without re-generation, and continues the fixed subset.
  These have no structured report and are unobserved reference disclosures, not
  evidence of no prior knowledge. Report valid-output coverage separately.
- Only classified transient transport/upstream failures are retried, at most
  three times. Every attempt is journaled, with shared Retry-After cooldown.
- Invalid/truncated structured reports, wrong model IDs, authentication, billing,
  and incompatible requests stop further submission. In-flight responses are saved.
- With the explicit `--record-output-limits` observation policy, classified
  output-budget exhaustion is retained as a terminal unobserved report and does
  not stop remaining cases. Other incomplete/invalid reports still stop the run.
  Token limits and inference requests remain unchanged.
- After inspecting a completed malformed assessment, `--record-invalid-reports`
  can retain it as a terminal invalid-output failure and continue the fixed
  subset. It requires an HTTP-200, correct-model, completed stop-finish response;
  model mismatches and transport/authentication failures still stop submission.
  It does not manufacture a label/report or rerun the case. Invalid reports are
  unobserved recall disclosures and incorrect classifications.
- A `--retry-plan` can authorize exactly one user-requested generation retry of
  each archived output-limit failure. Original responses and original source code
  are backed up and hashed; every rerun is journaled with generation-attempt and
  provenance fields. A second resume does not repeat completed reruns, even when
  a rerun also hits its budget. Successful/filtered cases are not included.
- Atomic per-case files are authoritative for resume. Fingerprinted plans reject
  changed prompts, model, inputs, token cap, observation policy, or code.
- Classification accuracy includes every attempted fixed-subset decision, with
  refusals treated as incorrect. Recall-reference rates are computed only after
  manual review, using valid structured reports and explicit coverage denominators.

## Conservative manual analysis

Freeze the coding policy before reviewing these responses, with targets and
correctness hidden. Read all common structured-report fields; do not use keyword
matches or non-null fields as the judge.

For every candidate, retain an exact saved-text excerpt and separate:

1. Asserted versus uncertain recollection.
2. Rating attributed to the specific outlet, a named relevant parent/network, or
   merely an unspecified similar outlet type/site template.
3. Confirmed-by-self-report versus tentative/inferred network affiliation.

A comparison with generic automated/"pink slime" sites is **generic**, even if
MBFC or a possible network name is mentioned elsewhere. It qualifies as family
rating recall only when the response clearly attributes an existing rating to
the named family. A rating claimed for a named parent can qualify even when
affiliation is tentative, but that uncertainty must be visible in the breakdown.
Null recall fields do not settle the code: explicit recollection elsewhere can
qualify, while an unanchored general analogy remains excluded.

Report asserted-only and asserted-plus-uncertain rates separately, and split
outlet-specific, named-family, and tentative-family references. Apply this same
analysis policy to GPT/Opus before making a revised cross-model ranking. These
remain elicited claims, not independently verified ratings or training membership.

## Commands

Run from the project root with `OPENROUTER_API_KEY` exported:

```bash
.venv/bin/python scripts/eval_openrouter_summary_parallel.py \
  --model kimi \
  --output-dir artifacts/evaluations/kimi-summary-probe-parallel-20261010 \
  --record-missing-summaries --record-provider-refusals

.venv/bin/python scripts/eval_openrouter_summary_parallel.py \
  --model qwen \
  --output-dir artifacts/evaluations/qwen-summary-probe-parallel-20261010 \
  --record-missing-summaries --record-provider-refusals
```

Use `--dry-run` to verify all inputs without requests or writes, `--smoke-only`
for the gate alone, and `--resume` with the same flags to continue a validated run.
