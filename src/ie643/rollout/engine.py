"""Decoder-level attention rollout for a Qwen VL model on its own stored output.

One teacher-forced forward pass over ``chat prompt + image + raw output``. With
greedy decoding this reproduces the attention the model had while it generated
the answer, without generating again. vLLM cannot return attention weights, so
this runs in HF transformers with eager attention.

Rollout (Abnar & Zuidema, 2020): per decoder layer, average heads, mix in the
residual as ``a*A + (1-a)*I``, renormalise rows and multiply up the stack. Row q
of the product is the relevance of every input position for the hidden state at
q, which predicts token q+1. Hooks fold each layer in as it runs, so no layer's
full attention tensor is kept.

Causal decoders need changes to the vision-transformer recipe:

* Attention sinks are excluded: each layer's attention to them is zeroed and
  rows are renormalised (their raw share is still recorded). A sink is
  - any position before the first image token (``<|im_start|>``, chat template,
    Qwen2.5's default system prompt). Position 0 takes ~30-50% of raw attention
    and, attending only to itself, is an absorbing state that swallows all
    rollout mass after 36-64 layers; ``You are`` is a second sink for Qwen2.5.
  - any image token with massive activations (Sun et al., 2024; Kang et al.,
    2025 "visual attention sink"): max |hidden| over layers above
    ``sink_ratio`` x the image-token median. Qwen3-VL has 1-4 such patches per
    page (ratio 11-221; ordinary patches <= 6) taking 9-41% of its image attention;
    Qwen2.5-VL has none (max ratio 9.6), so ``sink_ratio=10`` sits in the gap.
* ``rollout`` (main): relevance propagates only through the generated tokens;
  every prompt position (image tokens, instruction text) is a leaf. Otherwise
  the earliest prompt tokens are near-absorbing too, and image-to-image mixing
  under a causal mask biases maps toward the top of the page.
  ``rollout_full`` keeps the classic full-matrix product (sinks removed) for
  comparison; it still collapses onto the earliest kept prompt tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ie643.schema import teacher_prompt


@dataclass
class Loaded:
    model_id: str
    model: Any
    processor: Any
    layers: list[torch.nn.Module]


def load_model(model_id: str, *, device_map: str = "auto", adapter: str | None = None) -> Loaded:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",  # sdpa / flash do not return weights
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    # Qwen2_5_VLDecoderLayer / Qwen3VLTextDecoderLayer; vision blocks are *VisionBlock
    layers = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer") and hasattr(m, "self_attn")]
    if not layers:
        raise RuntimeError(f"no decoder layers found in {model_id}")
    return Loaded(model_id, model, processor, layers)


@dataclass
class _Capture:
    """Per-layer tensors captured during one forward pass (rollout is computed afterwards,
    because visual sinks are only known once every layer's activations have been seen)."""

    q_rows: torch.Tensor  # positions whose hidden state predicts each output token (contiguous)
    full: bool
    q_attn: list[torch.Tensor] = field(default_factory=list)  # [T, N] head-mean, fp32
    attn: list[torch.Tensor] = field(default_factory=list)  # [N, N] head-mean, bf16 (full only)
    act_max: list[torch.Tensor] = field(default_factory=list)  # [N] max |hidden| entering the layer


def _attn_hook(cap: _Capture):
    def hook(_module, _args, output):
        weights = output[1]
        if weights is None:
            raise RuntimeError("attention weights missing; is attn_implementation='eager'?")
        attn = weights[0].to(cap.q_rows.device, torch.float32).mean(0)
        cap.q_attn.append(attn[cap.q_rows])
        if cap.full:
            cap.attn.append(attn.to(torch.bfloat16))

    return hook


def _act_hook(cap: _Capture):
    def hook(_module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        cap.act_max.append(h[0].abs().amax(-1).to(cap.q_rows.device, torch.float32))

    return hook


def _mix(attn: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor, alpha: float) -> torch.Tensor:
    """attn [len(rows), N] for query positions ``rows`` -> sink-free, residual-mixed, row-stochastic."""
    attn = attn * keep
    attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-12)
    mixed = alpha * attn
    mixed[torch.arange(len(rows), device=attn.device), rows] += 1 - alpha
    return mixed / mixed.sum(-1, keepdim=True)


def output_rollout(q_attn: list[torch.Tensor], q_rows: torch.Tensor, keep: torch.Tensor, alpha: float) -> torch.Tensor:
    """[T, N]: relevance flows bottom-up through generated tokens; prompt positions are leaves."""
    s0, s1 = int(q_rows[0]), int(q_rows[-1]) + 1
    roll = None
    for a in q_attn:
        mixed = _mix(a, q_rows, keep, alpha)
        if roll is None:
            roll = mixed
        else:
            new = mixed[:, s0:s1] @ roll  # paths through earlier generated tokens
            new[:, :s0] += mixed[:, :s0]  # direct hits on prompt leaves
            roll = new
    return roll


def full_rollout(attn: list[torch.Tensor], keep: torch.Tensor, alpha: float) -> torch.Tensor:
    """Classic [N, N] product over every position."""
    n = attn[0].shape[0]
    rows = torch.arange(n, device=keep.device)
    roll = None
    for a in attn:
        m = _mix(a.float(), rows, keep, alpha)
        roll = m if roll is None else m @ roll
    return roll


def build_inputs(loaded: Loaded, image_path: Path, prompt: str, output_text: str) -> tuple[dict, list[tuple[int, int]], int]:
    """Same message layout as ie643.teacher.client (user: image then text)."""
    proc = loaded.processor
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    enc = proc(text=[chat], images=[image], return_tensors="pt")
    out = proc.tokenizer(output_text, add_special_tokens=False, return_offsets_mapping=True)
    n_prompt = enc["input_ids"].shape[1]
    out_ids = torch.tensor([out["input_ids"]], dtype=enc["input_ids"].dtype)
    for key, val in list(enc.items()):
        # per-token side inputs (mm_token_type_ids, ...): output tokens are plain text = 0
        if key != "input_ids" and torch.is_tensor(val) and val.shape[:2] == (1, n_prompt):
            enc[key] = torch.cat([val, torch.zeros_like(out_ids, dtype=val.dtype)], dim=1)
    enc["input_ids"] = torch.cat([enc["input_ids"], out_ids], dim=1)
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    return dict(enc), [tuple(o) for o in out["offset_mapping"]], n_prompt


@torch.no_grad()
def rollout_record(
    loaded: Loaded,
    rec: dict,
    root: Path,
    *,
    raw_key: str,
    alpha: float = 0.5,
    sink_ratio: float = 10.0,
    full: bool = True,
) -> dict[str, Any]:
    # the prompt the output was generated with: "<stem>_prompt" (student_e3_raw -> student_e3_prompt). Only the two
    # legacy hinted runs (teacher_raw / student_raw) fall back to the stored hinted teacher_prompt; anything else
    # without its own prompt field (e.g. teacher_nohint_raw) was generated with the current hint-free prompt.
    stem = raw_key[: -len("_raw")] if raw_key.endswith("_raw") else raw_key
    legacy = rec.get("teacher_prompt") if stem in ("teacher", "student") else None
    prompt = rec.get(f"{stem}_prompt") or legacy or teacher_prompt()
    raw = rec[raw_key]
    inputs, offsets, n_prompt = build_inputs(loaded, root / rec["image"], prompt, raw)
    ids = inputs["input_ids"][0]
    n_out = ids.shape[0] - n_prompt

    img_pos = torch.nonzero(ids == loaded.model.config.image_token_id).flatten()
    merge = loaded.processor.image_processor.merge_size
    patch = loaded.processor.image_processor.patch_size
    _, gh, gw = (int(x) for x in inputs["image_grid_thw"][0])
    grid = (gh // merge, gw // merge)
    if img_pos.numel() != grid[0] * grid[1]:
        raise RuntimeError(f"{img_pos.numel()} image tokens vs grid {grid}")

    dev = next(loaded.model.parameters()).device
    cap = _Capture(q_rows=torch.arange(n_prompt - 1, n_prompt - 1 + n_out, device=dev), full=full)
    handles = [layer.self_attn.register_forward_hook(_attn_hook(cap)) for layer in loaded.layers]
    handles += [layer.register_forward_pre_hook(_act_hook(cap), with_kwargs=True) for layer in loaded.layers]
    try:
        inputs = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}
        out = loaded.model(**inputs, use_cache=False, logits_to_keep=n_out + 1)
    finally:
        for h in handles:
            h.remove()
    pred = out.logits[0, :n_out].argmax(-1).cpu()
    del out

    img = img_pos.to(dev)
    act = torch.stack(cap.act_max)[:, img]  # [L, n_img]
    ratio = (act / act.median(-1, keepdim=True).values).amax(0)
    keep = torch.ones(ids.shape[0], device=dev)
    keep[: int(img_pos[0])] = 0.0  # chat template / system prompt before the image
    keep[img[ratio > sink_ratio]] = 0.0  # visual attention sinks

    q_stack = torch.stack(cap.q_attn)  # [L, T, N]
    res: dict[str, Any] = {
        "uid": rec["uid"],
        "model": loaded.model_id,
        "raw": raw,
        "offsets": np.asarray(offsets, dtype=np.int32),
        "n_prompt": n_prompt,
        "n_out": n_out,
        "img_pos": img_pos.numpy().astype(np.int32),
        "grid": np.asarray(grid, dtype=np.int32),
        "token_px": merge * patch,
        "proc_size": np.asarray((gh * patch, gw * patch), dtype=np.int32),  # model-side H, W
        "keep": keep.cpu().numpy().astype(bool),
        "img_act_ratio": ratio.cpu().numpy(),
        "rollout": output_rollout(cap.q_attn, cap.q_rows, keep, alpha).cpu().numpy(),  # [n_out, N]
        "last_attn": cap.q_attn[-1].cpu().numpy(),  # raw, sinks included
        "layer_img": q_stack[:, :, img].sum(-1).mean(-1).cpu().numpy(),
        "layer_img_sink": q_stack[:, :, img[ratio > sink_ratio]].sum(-1).mean(-1).cpu().numpy(),
        "layer_sink": q_stack[:, :, 0].mean(-1).cpu().numpy(),
        "layer_prefix": q_stack[:, :, : int(img_pos[0])].sum(-1).mean(-1).cpu().numpy(),
        "argmax_match": float((pred == ids[n_prompt:]).float().mean()),
    }
    if full:
        res["rollout_full"] = full_rollout(cap.attn, keep, alpha)[cap.q_rows].cpu().numpy()
    return res
