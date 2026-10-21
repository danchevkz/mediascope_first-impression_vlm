# Usage

Run commands from the package root with Python 3.10+. Replace `/path/to/...`
with local paths.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

For GPU training/local inference, install a CUDA-compatible PyTorch/torchvision
pair, then `python -m pip install -r requirements-training.txt`. Recorded versions
are in `provenance/environment.json` and `configs/training/`. Training used RTX
PRO 6000 Blackwell GPUs with approximately 96 GB VRAM each.

## Offline reproduction

```bash
python scripts/reproduce_release.py
python scripts/reproduce_release.py --verify-random
```

The first command reproduces metrics from saved predictions and annotations.
The second also regenerates the 10,000 random-baseline repetitions. Outputs are
`metrics.csv`, `reference_metrics.csv`, `figure3_data.csv`, and `REPORT.md` in
`results/reproduced/`.

## 1. Screenshot inputs

The main screenshot directory must contain `train/*.png` and `test/*.png`, using
the filenames in `data/main/`. The paired directory must contain `masked/*.png`
and `unmasked/*.png`, using `data/paired/manifest.json`.

Bind the supplied manifests to the images and verify every required SHA-256:

```bash
python scripts/paper.py prepare --dataset main \
  --images /path/to/main-screenshots --output work/inputs-main
python scripts/paper.py prepare --dataset masked \
  --images /path/to/paired-screenshots --output work/inputs-masked
python scripts/paper.py prepare --dataset unmasked \
  --images /path/to/paired-screenshots --output work/inputs-unmasked
```

These commands verify image hashes and write local input manifests.

## 2. API classification

Export the relevant credential in the shell where inference is run:

```bash
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
export OPENROUTER_API_KEY='your-key'
```

The allowed paper model selectors are `gpt56`, `opus5`, `kimi`, and `qwen`.

```bash
python scripts/paper.py api --model gpt56 --dataset main \
  --inputs work/inputs-main --output work/main/gpt56 --dry-run
python scripts/paper.py api --model gpt56 --dataset main \
  --inputs work/inputs-main --output work/main/gpt56

python scripts/paper.py api --model opus5 --dataset masked \
  --inputs work/inputs-masked --output work/masked/opus5
python scripts/paper.py api --model opus5 --dataset unmasked \
  --inputs work/inputs-unmasked --output work/unmasked/opus5
```

`--dry-run` prints the command; `--resume` continues an interrupted run.
API runs incur provider charges and record request attempts and failures.

## 3. Local inference

Extract the companion adapter archive into the package root. It produces
`assets/adapters/{qwen35,ministral}_{factuality,bias}` and
`assets/adapters/qwen35_factuality_main`. The latter is the distinct Qwen
factuality checkpoint used for the main test. The wrapper selects checkpoints
and generation settings from `configs/local_evaluation.json`.

```bash
python scripts/paper.py local --model qwen35 --variant finetuned --task factuality \
  --dataset unmasked --images /path/to/paired-screenshots \
  --output work/local/qwen35-ft-factuality.csv

python scripts/paper.py local --model ministral --variant base --task bias \
  --dataset masked --images /path/to/paired-screenshots \
  --output work/local/ministral-base-bias.csv
```

Use `--dataset main --images /path/to/main-screenshots` for the main test set.
Use `--adapters /path/to/adapters` to relocate adapter assets. Base runs use the
factuality adapter's saved processor assets to match the evaluation protocol,
without attaching adapter weights. `--limit 2` provides a small local check.

## 4. Training

Training settings are loaded from `configs/training/`:

```bash
python scripts/paper.py train --model qwen35 --task factuality \
  --images /path/to/main-screenshots --output work/training/qwen35-factuality \
  --dry-run
```

Remove `--dry-run` to run training. Select `qwen35` or `ministral` and
`factuality` or `bias` for the four configurations. Use a fresh output directory
for a new run; the original trainer automatically resumes matching checkpoints.
Adapter weights after a new training run need not be byte-identical across GPU
software/hardware environments.

## 5. Explicit recognition probes

Prepare full-page tiles once, without making model calls:

```bash
python scripts/paper.py prepare-api-inputs \
  --inputs work/inputs-main --output work/probe-inputs
```

Use the original Opus/GPT runners with explicit portable inputs:

```bash
python scripts/eval_opus_summary_parallel.py \
  --dataset-root work/inputs-main --subset-csv data/recognition/subset.csv \
  --input-run-dir work/probe-inputs --output-dir work/recognition-opus5

python scripts/eval_gpt56_summary_parallel.py \
  --dataset-root work/inputs-main --subset-csv data/recognition/subset.csv \
  --input-run-dir work/probe-inputs --output-dir work/recognition-gpt56 \
  --stream --record-missing-summaries
```

The OpenRouter runner shares the same prompts/schema and checks the Opus
reference plan created by the preceding command:

```bash
python scripts/eval_openrouter_summary_parallel.py --model kimi \
  --reference-plan work/recognition-opus5/plan.json \
  --subset-csv data/recognition/subset.csv --output-dir work/recognition-kimi \
  --record-missing-summaries --record-provider-refusals \
  --record-output-limits --record-invalid-reports
```

Select `--model qwen` and a different output directory for Qwen. These flags
record terminal unobserved/invalid reports rather than resampling them. The exact
original run plans are included in `provenance/recognition/`.
Snapshots and rubrics are in `results/reference_audits/`;
`reproduce_release.py` aggregates their reviewed annotations offline.

## 6. Naïve baselines

```bash
python scripts/run_naive_baselines.py --dataset-root data/main \
  --seed 42 --random-repetitions 10000 --out work/naive-baselines.json
```

## Tests and archive verification

To render the rating-reference figure from the reproduced counts:

```bash
python -m pip install -r requirements-figures.txt
python scripts/plot_reference_analysis.py --output-prefix work/figures/figure2
```

This writes PDF, SVG, PNG, and CSV files. Reference rates use valid reports;
both correct-prediction bars use all attempted cases.

```bash
python -m pip install -r requirements-tests.txt
PYTHONPATH=scripts python -m unittest discover -s tests -p 'test_*.py'
python scripts/verify_adapter_archive.py /path/to/mediascope-vlm-release-20261021-adapters.tar.gz
```

Tests use synthetic images and mocked API calls. Archive verification checks the
adapter checksum and each asset without extracting the weight files.
