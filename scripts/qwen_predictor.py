#!/usr/bin/env python3
"""Qwen3.5-4B predictor over an OpenAI-compatible (OpenRouter) vision endpoint.

Tall full-page screenshots are sliced into readable vertical tiles and sent as
multiple images in a single message. Factuality and bias are asked as two
SEPARATE calls (per the task design), reusing the same tiles for both.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image

from run_config import IMAGE_DEFAULTS, adapter_fingerprint, render_generation_prompt

# Full-page screenshots are legitimately huge (some >180M px); these are
# trusted local files, so disable PIL's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None

# --- Shared prompts: outlet identity must not be supplied as text metadata ---

FACTUALITY_PROMPT = """You are an expert media analyst applying Media Bias/Fact Check (MBFC) methodology.

Below are vertical slices (top to bottom) of a single full-page screenshot of the
homepage of a media outlet. Judge its FACTUAL REPORTING level
from what you can see — headline wording, sourcing cues, sensationalism, ads,
conspiracy/pseudoscience signals, corrections.

Choose EXACTLY ONE label:
- LOW: poor sourcing, multiple failed fact checks, propaganda or conspiracy/pseudoscience
- MIXED: some failed fact checks or loaded/unreliable sourcing
- HIGH: generally-to-highly reliable, strong sourcing, few or no failed fact checks

Output ONLY the label text (LOW, MIXED, or HIGH), nothing else."""

BIAS_PROMPT = """You are an expert media analyst applying Media Bias/Fact Check (MBFC) methodology.

Below are vertical slices (top to bottom) of a single full-page screenshot of the
homepage of a media outlet. Judge its POLITICAL BIAS from what
you can see — story selection, framing, loaded language, which side is favored.

Choose EXACTLY ONE label:
- LEFT
- LEFT-CENTER
- LEAST BIASED
- RIGHT-CENTER
- RIGHT

Output ONLY the label text, nothing else."""

PROMPTS = {"factuality": FACTUALITY_PROMPT, "bias": BIAS_PROMPT}


# --- Tiling ------------------------------------------------------------------


def tile_image(path, tile_height=IMAGE_DEFAULTS["tile_height"], max_tiles=IMAGE_DEFAULTS["max_tiles"], max_width=IMAGE_DEFAULTS["max_width"]):
    """Slice a tall screenshot into <= max_tiles readable vertical tiles.

    Returns (tiles, truncated_px): PIL images top-to-bottom and how many pixels
    of page height were dropped past the tile cap (0 if the whole page fit).
    """
    tiles, info = tile_image_with_info(path, tile_height, max_tiles, max_width)
    return tiles, info["truncated_px"]


def tile_image_with_info(path, tile_height=IMAGE_DEFAULTS["tile_height"], max_tiles=IMAGE_DEFAULTS["max_tiles"], max_width=IMAGE_DEFAULTS["max_width"]):
    """Return tiles and the exact resizing/cropping information used."""
    if min(tile_height, max_tiles, max_width) <= 0:
        raise ValueError("Image sizing settings must be positive")
    with Image.open(path) as source:
        original_size = list(source.size)
        img = source.convert("RGB")
    if img.width > max_width:  # keep text crisp; only downscale over-wide shots
        scale = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * scale))))
    w, h = img.size
    # A trailing remainder (h % tile_height) can be a sliver only a few px
    # tall, giving that one tile an extreme aspect ratio -- Qwen's image
    # processor hard-rejects anything over 200:1. Fold slivers below this
    # into the previous tile instead of emitting them on their own.
    min_tile_height = max(50, w // 100)
    tiles = []
    y = 0
    while y < h and len(tiles) < max_tiles:
        end = min(y + tile_height, h)
        if 0 < h - end < min_tile_height:
            end = h  # absorb the trailing sliver into this tile
        tiles.append(img.crop((0, y, w, end)))
        y = end
    truncated_px = max(0, h - y)
    return tiles, {
        "original_size": original_size, "resized_size": [w, h],
        "tile_sizes": [list(t.size) for t in tiles], "tile_count": len(tiles),
        "truncated_px": truncated_px, "retained_height_fraction": y / h,
    }


def _vstack(tiles):
    """Paste tiles top-to-bottom into a single tall image (for 1-image models)."""
    w = max(t.width for t in tiles)
    h = sum(t.height for t in tiles)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for t in tiles:
        canvas.paste(t, (0, y))
        y += t.height
    return canvas


def _data_url(tile: Image.Image) -> str:
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# --- Predictor ---------------------------------------------------------------


class OpenRouterQwenPredictor:
    """Predictor over any OpenAI-compatible (OpenRouter) vision model.

    Named for Qwen historically, but model-agnostic: pass any OpenRouter slug.
    Two per-model knobs matter across families:
      * disable_reasoning: send `reasoning:{enabled:false}` (Qwen3-VL etc. are
        reasoning models that otherwise burn max_tokens "thinking"). Harmless /
        ignored for non-reasoning models, but skippable.
      * single_image: some models (notably Llama-3.2-Vision) accept only ONE
        image per request; stitch the tiles into one tall image instead of
        sending them as a multi-image message.
    """

    def __init__(
        self,
        model="qwen/qwen3-vl-8b-instruct",
        api_key=None,
        base_url="https://openrouter.ai/api/v1",
        tile_height=IMAGE_DEFAULTS["tile_height"],
        max_tiles=IMAGE_DEFAULTS["max_tiles"],
        max_width=IMAGE_DEFAULTS["max_width"],
        single_image=False,
        disable_reasoning=True,
        timeout=120,
        max_new_tokens=512,
        prompts=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            if "openrouter" in self.base_url:
                raise RuntimeError("Set OPENROUTER_API_KEY (or pass api_key=).")
            self.api_key = "local"  # local servers (Ollama/vLLM) ignore the token
        self.tile_height = tile_height
        self.max_tiles = max_tiles
        self.max_width = max_width
        self.single_image = single_image
        self.disable_reasoning = disable_reasoning
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.prompts = dict(PROMPTS if prompts is None else prompts)
        self._tile_cache = {}  # path -> list[data_url]; reused across both tasks
        self._image_details = {}
        self.truncation_log = []
        self.last_details = {}

    def run_metadata(self):
        url = urlsplit(self.base_url)
        endpoint = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
        return {
            "backend": "openai-compatible", "model_id": self.model, "base_url": endpoint,
            "prompts": self.prompts,
            "preprocessing": {k: getattr(self, k) for k in IMAGE_DEFAULTS},
            "single_image": self.single_image, "disable_reasoning": self.disable_reasoning,
            "generation": {"max_new_tokens": self.max_new_tokens, "temperature": 0},
            "timeout_seconds": self.timeout,
        }

    def _tiles_for(self, sample):
        if sample.image_path not in self._tile_cache:
            tiles, info = tile_image_with_info(
                sample.image_path, self.tile_height, self.max_tiles, self.max_width
            )
            trunc = info["truncated_px"]
            if trunc > 0:
                self.truncation_log.append((sample.media_name, trunc))
            if self.single_image and tiles:  # stitch for 1-image-only models
                tiles = [_vstack(tiles)]
            info["submitted_image_sizes"] = [list(t.size) for t in tiles]
            self._tile_cache.clear()
            self._image_details = info
            self._tile_cache[sample.image_path] = [_data_url(t) for t in tiles]
        return self._tile_cache[sample.image_path]

    def predict(self, sample, task_key):
        self.last_details = {}
        prompt = self.prompts[task_key]
        content = [{"type": "text", "text": prompt}]
        for durl in self._tiles_for(sample):
            content.append({"type": "image_url", "image_url": {"url": durl}})
        self.last_details = {"image": self._image_details, "prompt": prompt}

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
        }
        if self.disable_reasoning:
            # Reasoning models (Qwen3-VL etc.) otherwise burn max_tokens
            # "thinking" and return empty content; ignored by non-reasoning models.
            body["reasoning"] = {"enabled": False}
            if "openrouter" not in self.base_url and "qwen" in self.model.lower():
                body["chat_template_kwargs"] = {"enable_thinking": False}

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        self.last_details["http_status"] = resp.status_code
        if resp.status_code >= 400:
            self.last_details["response_body"] = resp.text
        resp.raise_for_status()
        response = resp.json()
        choice = response["choices"][0]
        self.last_details.update({
            "response_id": response.get("id"), "response_model": response.get("model"),
            "usage": response.get("usage"), "choice": choice,
        })
        # Reasoning is retained in the audit record, but is not a final answer.
        return choice["message"].get("content")


class TransformersLoraPredictor:
    """Local inference: base model (optionally 4-bit) + a trained LoRA adapter,
    generated directly via transformers/peft -- no vLLM/API hop needed for a
    quick sanity check of an adapter fresh off training.
    """

    def __init__(
        self,
        model_id,
        adapter_dir=None,
        processor_dir=None,
        quant="4bit",
        skip_vision_modules=(),
        tile_height=IMAGE_DEFAULTS["tile_height"],
        max_tiles=IMAGE_DEFAULTS["max_tiles"],
        max_width=IMAGE_DEFAULTS["max_width"],
        max_new_tokens=16,
        thinking="template-default",
        prompts=None,
        revision=None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.model_id = model_id
        self.adapter_dir = str(Path(adapter_dir).resolve()) if adapter_dir else None
        self.thinking = thinking
        self.prompts = dict(PROMPTS if prompts is None else prompts)
        # Intermediate/legacy checkpoints may not contain a processor bundle.
        processor_source = processor_dir or (
            adapter_dir if adapter_dir and (Path(adapter_dir) / "tokenizer_config.json").exists() else model_id
        )
        processor_kwargs = {"trust_remote_code": True}
        if processor_source == model_id:
            processor_kwargs["revision"] = revision
        if "mistral" in model_id.lower():
            processor_kwargs["fix_mistral_regex"] = True
        self.processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
        self.processor_source = str(processor_source)

        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available()  # Apple Silicon GPU
        if quant == "4bit" and not has_cuda:
            # bitsandbytes 4-bit quantization is CUDA-only -- no GPU to quantize
            # onto on a CPU-only pod or Apple Silicon, so fall back to
            # full-precision instead of hard-failing (slower on plain CPU,
            # still reasonably fast on MPS).
            print("[warn] no CUDA device found -- skipping 4-bit quant, "
                  "running full-precision instead.")
            quant = "16bit"

        quant_config = None
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=list(skip_vision_modules) + ["lm_head"],
            )
        if has_cuda:
            device_map = {"": 0}
        elif has_mps:
            device_map = {"": "mps"}
        else:
            device_map = "cpu"
        base = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quant_config,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            revision=revision,
        )
        self.model_revision = getattr(base.config, "_commit_hash", None)
        self.quant = quant
        if adapter_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, adapter_dir)
        else:
            self.model = base
        self.model.eval()
        self.tile_height, self.max_tiles, self.max_width = tile_height, max_tiles, max_width
        self.max_new_tokens = max_new_tokens
        self._tile_cache = {}
        self._image_details = {}
        self.truncation_log = []
        self.last_details = {}

    def run_metadata(self):
        import torch

        device = self.model.device
        hardware = {"device_type": device.type, "torch_cuda_runtime": torch.version.cuda}
        if device.type == "cuda":
            hardware.update(name=torch.cuda.get_device_name(device),
                            capability=list(torch.cuda.get_device_capability(device)))
        return {
            "backend": "transformers-peft", "model_id": self.model_id,
            "hardware": hardware,
            "model_revision": self.model_revision, "adapter_dir": self.adapter_dir,
            "adapter_files_sha256": adapter_fingerprint(self.adapter_dir) if self.adapter_dir else {},
            "processor_source": self.processor_source,
            "processor_class": type(self.processor).__name__,
            "chat_template": self.processor.chat_template,
            "quant": self.quant, "thinking": self.thinking, "prompts": self.prompts,
            "preprocessing": {k: getattr(self, k) for k in IMAGE_DEFAULTS},
            "generation": {"max_new_tokens": self.max_new_tokens, "do_sample": False},
            "base_generation_config": self.model.generation_config.to_dict(),
        }

    def _tiles_for(self, sample):
        if sample.image_path not in self._tile_cache:
            tiles, info = tile_image_with_info(sample.image_path, self.tile_height, self.max_tiles, self.max_width)
            trunc = info["truncated_px"]
            if trunc > 0:
                self.truncation_log.append((sample.media_name, trunc))
            self._tile_cache.clear()
            self._image_details = info
            self._tile_cache[sample.image_path] = tiles
        return self._tile_cache[sample.image_path]

    def predict(self, sample, task_key):
        import torch

        self.last_details = {}
        tiles = self._tiles_for(sample)
        prompt = self.prompts[task_key]
        self.last_details = {"image": self._image_details, "prompt": prompt}
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in tiles],
        }]
        prompt_text = render_generation_prompt(self.processor, messages, self.thinking)
        self.last_details["rendered_prompt"] = prompt_text
        enc = self.processor(text=prompt_text, images=tiles, return_tensors="pt").to(self.model.device)
        self.last_details["input_tokens"] = enc["input_ids"].shape[1]
        self.last_details["tensor_shapes"] = {k: list(v.shape) for k, v in enc.items()}
        for key in ("image_grid_thw", "image_sizes"):
            if key in enc:
                self.last_details[key] = enc[key].tolist()

        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, enc["input_ids"].shape[1]:]
        self.last_details.update({
            "generated_token_ids": gen.tolist(), "generated_tokens": len(gen),
            "reached_token_budget": len(gen) >= self.max_new_tokens,
            "decoded_with_special_tokens": self.processor.tokenizer.decode(gen, skip_special_tokens=False),
        })
        return self.processor.tokenizer.decode(gen, skip_special_tokens=True)
