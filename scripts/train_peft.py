#!/usr/bin/env python3
"""LoRA fine-tuning for the mediascope factuality & bias tasks.

Trains ONE adapter per (model, task) call — run it 4 times for the full grid
(2 models x {factuality,bias}). Checkpoints are written frequently and the
latest checkpoint is resumed automatically.

Reuses the eval-time prompts and tiling from qwen_predictor.py so training
input formatting matches inference-time formatting exactly, and reuses
mediascope_eval.load_samples so labels/collapse/image-path resolution stay
one source of truth with the rest of the project.

Checkpoints are saved every --checkpoint-steps to --output-dir, and training
auto-resumes from the latest checkpoint found there.

The shared defaults are --max-tiles 3 --max-width 1024. Larger inputs need a
separate VRAM smoke check before a full run.

Before a full run, ALWAYS smoke-test first:
    python scripts/train_peft.py --model qwen35 --task factuality \
        --limit 2 --max-steps 1 --output-dir artifacts/smoke/qwen35 \
        --max-tiles 1 --max-width 512 --grad-accum 1

If the model errors on quantization/target-module names, inspect the actual
module tree with:
    python scripts/train_peft.py --model qwen35 --list-modules

Usage (real run):
    python scripts/train_peft.py --model ministral --task factuality \
        --output-dir artifacts/adapters/ministral_lora_factuality \
        --max-tiles 3 --max-width 1024
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import traceback
import json
from pathlib import Path

from mediascope_eval import load_samples, TASKS
from qwen_predictor import PROMPTS, tile_image
from run_config import (IMAGE_DEFAULTS, THINKING_MODES, environment_info,
                        render_generation_prompt, sha256_file, source_info, write_json)

# Repo ids picked so bitsandbytes quantizes from a plain bf16 checkpoint
# (avoiding any fp8-checkpoint-to-4bit quantization ambiguity on very new
# releases). Adjust `skip_vision_modules` if --list-modules shows different
# names on the actual model tree — these are best-effort guesses based on
# each family's usual naming convention (Pixtral/LLaVA-style vision_tower for
# Mistral3, Qwen-VL-style `visual` for Qwen).
MODEL_REGISTRY = {
    "ministral": {
        "id": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        "skip_vision_modules": ["vision_tower", "multi_modal_projector"],
    },
    "qwen35": {
        "id": "Qwen/Qwen3.5-4B",
        "skip_vision_modules": ["visual", "vision_model", "vision_tower"],
    },
}

DEFAULT_LR = 2e-4

# A single bad sample (corrupt image, processor rejecting an extreme tile
# aspect ratio, etc.) shouldn't kill a multi-hour paid GPU run -- retry with
# a different random sample instead, up to this many times, before giving up.
MAX_SAMPLE_RETRIES = 10


def resolve_model_id(model_arg: str) -> tuple[str, list[str]]:
    if model_arg in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_arg]
        return entry["id"], entry["skip_vision_modules"]
    for entry in MODEL_REGISTRY.values():
        if entry["id"] == model_arg:
            return model_arg, entry["skip_vision_modules"]
    return model_arg, []  # raw HF id, no vision-skip guess


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda d: int(d.split("-")[1]))
    return os.path.join(output_dir, ckpts[-1])


def list_modules(model_id: str, trust_remote_code: bool = True):
    from transformers import AutoModelForImageTextToText

    print(f"Loading {model_id} (cpu, fp32, no quantization) just to list modules...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    seen = set()
    vision_paths = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        seen.add(cls)
        if name and "vision" in cls.lower():
            vision_paths.append((name, cls))
    print("\nTop-level submodules:")
    for name, _ in model.named_children():
        print(f"  {name}")
    print("\nAll leaf module CLASS names (unique):")
    for cls in sorted(seen):
        print(f"  {cls}")
    print("\nVision-related module paths (shortest = the actual vision-tower attribute path):")
    for name, cls in sorted(vision_paths, key=lambda p: p[0].count(".")):
        print(f"  {name}  ({cls})")
    if vision_paths:
        top = min(vision_paths, key=lambda p: p[0].count("."))[0]
        print(f"\n--quant-skip-vision-modules should include: {top}")
    else:
        print("\nNo module with 'vision' in its class name found -- inspect the "
              "class list above manually for the vision tower's name.")


def load_model_and_processor(model_id, quant, skip_vision_modules, trust_remote_code=True, grad_checkpointing=True, revision=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    processor_kwargs = {"trust_remote_code": trust_remote_code, "revision": revision}
    if "mistral" in model_id.lower():
        # Known tokenizer pretokenization-regex bug on Mistral-family checkpoints --
        # transformers itself warns this silently mis-tokenizes without this flag
        # (https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84).
        # Not a real kwarg for non-Mistral tokenizers, so only pass it here.
        processor_kwargs["fix_mistral_regex"] = True
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    quant_config = None
    if quant == "4bit":
        # Always skip lm_head: it's commonly weight-tied to the input
        # embeddings, and quantizing a tied Linear leaves its .weight
        # pointing at the still-unquantized embedding tensor, which fails
        # bitsandbytes' internal packed-shape assertion. Also standard QLoRA
        # practice for output-logit numerical stability regardless of tying.
        skip_modules = list(skip_vision_modules) + ["lm_head"]
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip_modules,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    if quant == "4bit":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=grad_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        if grad_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.enable_input_require_grads()

    return model, processor


def build_lora_config(target_modules: str, lora_r: int):
    from peft import LoraConfig

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )


class MediaScopeSFTDataset:
    """One (image-tiles, prompt, label) example per __getitem__.

    Loss is masked to the completion (the label tokens + eos) only — the
    prompt + image tokens contribute 0 loss, standard SFT masking.
    """

    def __init__(self, samples, task_key, processor, tile_height=IMAGE_DEFAULTS["tile_height"], max_tiles=IMAGE_DEFAULTS["max_tiles"], max_width=IMAGE_DEFAULTS["max_width"], thinking="off", require_full_image=False):
        self.samples = samples
        self.task_key = task_key
        self.processor = processor
        self.tile_height = tile_height
        self.max_tiles = max_tiles
        self.max_width = max_width
        self.thinking = thinking
        self.require_full_image = require_full_image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.require_full_image:
            # Strict full-page runs must not substitute another example when
            # decoding/processing fails or the image exceeds the declared cap.
            return self._build_item(idx)
        # Wrap the real work so one bad sample (corrupt image, a processor
        # rejecting an extreme tile aspect ratio, etc.) logs a warning and
        # gets swapped for a different random sample instead of crashing a
        # multi-hour paid training run.
        for _ in range(MAX_SAMPLE_RETRIES):
            try:
                return self._build_item(idx)
            except Exception as e:
                s = self.samples[idx]
                print(f"[WARN] skipping sample idx={idx} ({s.image_path}): "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                idx = random.randrange(len(self.samples))
        raise RuntimeError(
            f"{MAX_SAMPLE_RETRIES} consecutive sample failures -- dataset or "
            "environment is likely broken, not just a few bad rows."
        )

    def _build_item(self, idx):
        import torch

        s = self.samples[idx]
        tiles, truncated = tile_image(s.image_path, self.tile_height, self.max_tiles, self.max_width)
        if self.require_full_image and truncated:
            raise ValueError(f"Full-page coverage required: {s.image_path} would lose {truncated} resized pixels")
        label = getattr(s, self.task_key)
        prompt = PROMPTS[self.task_key]

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image"} for _ in tiles],
            }
        ]
        prompt_text = render_generation_prompt(self.processor, messages, self.thinking)
        eos = self.processor.tokenizer.eos_token or ""
        full_text = prompt_text + label + eos

        enc_prompt = self.processor(text=prompt_text, images=tiles, return_tensors="pt")
        enc_full = self.processor(text=full_text, images=tiles, return_tensors="pt")

        prompt_len = enc_prompt["input_ids"].shape[1]
        if not torch.equal(enc_prompt["input_ids"], enc_full["input_ids"][:, :prompt_len]):
            raise ValueError("Prompt/completion token boundary changed; refusing incorrect loss masking")
        labels = enc_full["input_ids"].clone()
        labels[:, :prompt_len] = -100

        # Only input_ids/attention_mask/labels carry a "batch of prompts" dim
        # (1, seq_len). Vision tensors (pixel_values, image_grid_thw, ...) are
        # already a flat concatenation across this sample's image tiles with
        # no such wrapper dim -- squeezing/unsqueezing them would corrupt the
        # per-tile grid bookkeeping the vision encoder relies on.
        item = {k: (v.squeeze(0) if k in _TEXT_BATCH_KEYS else v) for k, v in enc_full.items()}
        item["labels"] = labels.squeeze(0)
        return item


# Every key here is shaped (1, seq_len) -- one entry per text token -- and
# must be padded + stacked like input_ids, NOT concatenated like a flat
# vision tensor. mm_token_type_ids (Qwen3.5: marks text vs. image tokens per
# position) was discovered live via scripts/probe_vision_shapes.py -- it's
# NOT in {"input_ids", "attention_mask"} so the original two-key set silently
# mis-handled it as a "vision" key and torch.cat'd two different sequence
# lengths together. Kept as a set (not a hardcoded field list in the
# collator) so any other per-token key some other model's processor returns
# gets the same correct treatment automatically instead of falling through
# to vision-style concatenation by default.
_TEXT_BATCH_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids"}
_PAD_VALUES = {"attention_mask": 0, "mm_token_type_ids": 0, "labels": -100}


class MediaScopeCollator:
    """Collates N examples into a real batch (N=1 degrades to the old behavior).

    Per-token keys (input_ids/attention_mask/labels/mm_token_type_ids/...) are
    right-padded to the batch's max length (standard for training -- loss is
    masked via `labels`, and attention_mask blocks the model from attending
    to pad positions either way). Vision tensors (pixel_values,
    image_grid_thw, ...) need no padding: they're already a flat
    concatenation across one example's image tiles with no batch-of-prompts
    wrapper dim, and that same flat-concatenation convention extends
    naturally across multiple examples' tiles too.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        import torch.nn.functional as F

        max_len = max(f["input_ids"].shape[0] for f in features)
        batch = {}
        for key in _TEXT_BATCH_KEYS | {"labels"}:
            if key not in features[0]:
                continue
            pad_value = self.pad_token_id if key == "input_ids" else _PAD_VALUES.get(key, 0)
            padded = [F.pad(f[key], (0, max_len - f[key].shape[0]), value=pad_value) for f in features]
            batch[key] = torch.stack(padded)

        vision_keys = [k for k in features[0] if k not in _TEXT_BATCH_KEYS | {"labels"}]
        for k in vision_keys:
            vals = [f[k] for f in features]
            if k == "pixel_values" and vals[0].ndim == 4:
                # Pixtral returns (images, channels, height, width), padded only
                # within each sample. Pad across samples before concatenating.
                max_height = max(v.shape[-2] for v in vals)
                max_width = max(v.shape[-1] for v in vals)
                vals = [F.pad(v, (0, max_width - v.shape[-1], 0, max_height - v.shape[-2]))
                        for v in vals]
            batch[k] = torch.cat(vals, dim=0) if torch.is_tensor(vals[0]) else vals
        return batch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry key (ministral, qwen35) or a raw HF repo id")
    ap.add_argument("--revision", help="immutable base-model revision; recorded and reused on resume")
    ap.add_argument("--thinking", choices=THINKING_MODES, default="off",
                    help="off: label-only direct-answer template; template-default: historical behavior")
    ap.add_argument("--task", choices=["factuality", "bias"], help="required unless --list-modules")
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--image-root", default="data")
    ap.add_argument("--output-dir", help="checkpoint directory; "
                                          "default artifacts/adapters/<model>_lora_<task>")
    ap.add_argument("--quant", choices=["4bit", "16bit"], default="4bit")
    ap.add_argument("--quant-skip-vision-modules", help="comma-separated module-name substrings to "
                                                          "exclude from 4-bit quant; overrides the registry guess")
    ap.add_argument("--target-modules", default="all-linear")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, help=f"overrides the default learning rate ({DEFAULT_LR})")
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--num-epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N steps (smoke test); -1 = full epochs")
    ap.add_argument("--batch-size", type=int, default=1, help="per-device train batch size (real batching, not simulated)")
    ap.add_argument("--grad-accum", type=int, default=16, help="effective batch size = --batch-size x --grad-accum")
    ap.add_argument("--checkpoint-steps", type=int, default=25, help="save every N steps -- keep small under spot pricing")
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only train on the first N rows (smoke test)")
    ap.add_argument("--tile-height", type=int, default=IMAGE_DEFAULTS["tile_height"])
    ap.add_argument("--max-tiles", type=int, default=IMAGE_DEFAULTS["max_tiles"])
    ap.add_argument("--max-width", type=int, default=IMAGE_DEFAULTS["max_width"])
    ap.add_argument("--require-full-image", action="store_true", help="fail preflight/sample processing instead of truncating or substituting images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes for image tiling/preprocessing -- "
                         "overlaps CPU work with GPU compute instead of blocking between "
                         "steps (0 = synchronous, the old default)")
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                    help="disable gradient checkpointing -- trades spare VRAM for speed; "
                         "only safe with real headroom (risks OOM otherwise)")
    ap.add_argument("--list-modules", action="store_true",
                    help="load the model on CPU, print its module tree, and exit (no training)")
    args = ap.parse_args()

    model_id, default_skip = resolve_model_id(args.model)

    if args.list_modules:
        list_modules(model_id)
        return

    if not args.task:
        ap.error("--task is required unless --list-modules is given")
    if args.warmup_steps < 0:
        ap.error("--warmup-steps must be nonnegative")

    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback
    from peft import get_peft_model

    torch.manual_seed(args.seed)

    skip_vision_modules = (
        args.quant_skip_vision_modules.split(",") if args.quant_skip_vision_modules else default_skip
    )

    model_key = args.model if args.model in MODEL_REGISTRY else model_id.split("/")[-1]
    output_dir = args.output_dir or f"artifacts/adapters/{model_key}_lora_{args.task}"
    run_path = Path(output_dir) / "run_config.json"
    settings = {
        "model_id": model_id, "task": args.task, "quant": args.quant,
        "skip_vision_modules": skip_vision_modules, "thinking": args.thinking,
        "require_full_image": args.require_full_image,
        "preprocessing": {key: getattr(args, key) for key in IMAGE_DEFAULTS},
        "prompts": {args.task: PROMPTS[args.task]},
        "csv_path": str(Path(args.csv).resolve()), "csv_sha256": sha256_file(args.csv),
        "arguments": {k: v for k, v in vars(args).items() if k not in ("output_dir", "list_modules")},
    }
    if any(v <= 0 for v in settings["preprocessing"].values()):
        ap.error("Image dimensions and tile count must be positive")
    saved_run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
    if saved_run and saved_run["settings"] != settings:
        ap.error("Output directory belongs to different run settings/data/prompts; use a new directory")
    if not saved_run and Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        ap.error("Nonempty output directory has no run_config.json; use a new directory for this protocol")
    os.makedirs(output_dir, exist_ok=True)

    found, missing = load_samples(args.csv, args.image_root, tasks=(args.task,))
    if missing or not found:
        ap.error(f"Expected a nonempty dataset with all images present; found={len(found)}, missing={len(missing)}")
    print(f"Loaded {len(found) + len(missing)} rows: {len(found)} with images, "
          f"{len(missing)} DROPPED (no image on disk)")
    if args.limit:
        found = found[: args.limit]
        print(f"--limit: training on first {len(found)} samples only")
    assert args.task in TASKS
    if args.require_full_image:
        from audit_image_sizes import geometry
        from PIL import Image
        for sample in found:
            with Image.open(sample.image_path) as image:
                layout = geometry(image.width, image.height, args.max_width, args.tile_height, args.max_tiles)
            if layout["truncated_px"]:
                ap.error(f"Full-page budget exceeded by {sample.image_path}")
        print(f"Full-image preflight passed for {len(found)} examples")
    training_samples = [{"image_path": str(Path(s.image_path).resolve()),
                         "image_sha256": sha256_file(s.image_path),
                         "label": getattr(s, args.task)} for s in found]
    if saved_run and saved_run.get("training_samples") != training_samples:
        ap.error("Training images or eligible examples changed since this run was created")

    print(f"model={model_id} task={args.task} quant={args.quant} "
          f"skip_vision_modules={skip_vision_modules}")
    model, processor = load_model_and_processor(
        model_id, args.quant, skip_vision_modules, grad_checkpointing=not args.no_grad_checkpointing,
        revision=saved_run.get("model_revision") if saved_run else args.revision,
    )
    run_record = saved_run or {
        "schema_version": 1, "settings": settings,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "environment": environment_info(), "source": source_info(),
        "training_samples": training_samples,
    }
    write_json(run_path, run_record)

    class SaveRunAssets(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            processor.save_pretrained(checkpoint)
            write_json(checkpoint / "run_config.json", run_record)

    peft_config = build_lora_config(args.target_modules, args.lora_r)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = MediaScopeSFTDataset(
        found, args.task, processor,
        tile_height=args.tile_height, max_tiles=args.max_tiles, max_width=args.max_width,
        thinking=args.thinking,
        require_full_image=args.require_full_image,
    )

    lr = args.lr or DEFAULT_LR
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=lr,
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.checkpoint_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.num_workers > 0,
    )
    run_record["effective_training_arguments"] = training_args.to_dict()
    run_record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(run_path, run_record)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MediaScopeCollator(pad_token_id),
        callbacks=[SaveRunAssets()],
    )

    resume = find_latest_checkpoint(output_dir)
    if resume:
        print(f"Resuming from {resume}")
    try:
        trainer.train(resume_from_checkpoint=resume)
    except Exception:
        latest = find_latest_checkpoint(output_dir)
        print("\n" + "=" * 70, file=sys.stderr)
        print("TRAINING CRASHED -- traceback below.", file=sys.stderr)
        print(f"Last checkpoint on disk: {latest or '(none saved yet)'}", file=sys.stderr)
        print(f"Resume with the same command plus --output-dir {output_dir}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)

    final_dir = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    write_json(Path(final_dir) / "run_config.json", run_record)
    print(f"\nSaved final adapter to {final_dir}")
    print("Next: serve the base model + this adapter via vLLM (or merge it), then run "
          "scripts/mediascope_eval.py --predictor openrouter --base-url <server> --model <served-name> "
          f"--task {args.task} against data/test.csv.")


if __name__ == "__main__":
    main()
