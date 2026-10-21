# MediaScope / MeFI

VLM code for **A Picture Is Worth a Hundred Articles: Media Profiling from
First Impressions**: predicting news-outlet factuality and political bias from
landing-page screenshots.

Includes Qwen/Ministral QLoRA training, local and frontier-model evaluation,
naïve baselines, paired masking experiments, and rating-reference analysis.

## Quickstart

Recompute metrics from the included predictions and annotations with Python 3.10+:

```bash
python3 -m venv .venv
.venv/bin/python scripts/reproduce_release.py
```

This checks the saved artifacts and writes metrics and figure data to
`results/reproduced/`. No model downloads or API calls are required.
See [protocols and result coverage](docs/PROTOCOLS.md).

## Training and inference

```bash
.venv/bin/python -m pip install -r requirements-inference.txt
```

For GPU training, install a CUDA-compatible PyTorch/torchvision pair followed by
`requirements-training.txt`. Use [RUNNING.md](docs/RUNNING.md) for commands,
data setup, API credentials, and tests. `scripts/paper.py --help` lists entry points.

## Structure

```text
.
├── scripts/       # Training, inference, scoring, and reference analysis
├── configs/       # Prompts, model revisions, and training settings
├── data/          # Split/pair manifests and image hashes
├── results/       # Predictions, annotations, and reproduced metrics
├── provenance/    # Run metadata and artifact coverage
├── docs/          # Usage and evaluation protocols
└── tests/         # CPU tests
```

Screenshots are supplied separately. The companion adapter archive extracts into
`assets/adapters/`.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff).
Code: [AI Pubs OpenRAIL-S v0.1](LICENSE). See [LICENSING.md](LICENSING.md)
for adapter and upstream-component licenses. The dataset is released separately.
