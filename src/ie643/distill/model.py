"""Student LoRA and the frozen teacher (docs/distillation_plan.md, C.2 and E5.1)."""

from __future__ import annotations

import torch
from torch import nn


def lora_target_names(model: nn.Module, suffixes: tuple[str, ...] | list[str], *, merger: bool) -> list[str]:
    """Language-model linears only. Vision stays frozen unless ``merger`` adds the merger MLP."""
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    has_language_model = any("language_model" in name for name, _ in modules)
    names = []
    for name, _module in modules:
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in suffixes:
            continue
        parts = name.split(".")
        if "visual" in parts:
            if merger and "merger" in parts:
                names.append(name)
            continue
        in_text = "language_model" in parts if has_language_model else ".layers." in f".{name}."
        if in_text and ("self_attn" in parts or "mlp" in parts or "merger" in parts):
            names.append(name)
    if not names:
        raise RuntimeError("no language-model Linear modules matched the LoRA targets")
    return names


def load_student(model_id: str, cfg, device: str):
    """BF16 student on ``device`` with a language-model LoRA. Vision encoder stays frozen."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from ie643.distill.data import configure_processor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    processor = configure_processor(AutoProcessor.from_pretrained(model_id))
    targets = lora_target_names(model, cfg.lora.target_modules, merger=cfg.lora.merger)
    peft_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=targets,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, peft_cfg)
    if cfg.rel.enabled:
        # Relation hooks need the Q/K/V activations to carry grad. Checkpointing drops
        # those tensors, so E5 trades it for micro-batch 1 (see configs/e5_minilm.yaml).
        model.gradient_checkpointing_disable()
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)
    return model, processor


def load_teacher(model_id: str, device: str):
    """Frozen teacher, full weights, no grad. The training step stops it at ``rel.teacher_layer``."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from ie643.distill.data import configure_processor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    model.to(device)
    processor = configure_processor(AutoProcessor.from_pretrained(model_id))
    return model, processor


def trainable_groups(model: nn.Module, lora_lr: float, new_lr: float) -> list[dict]:
    lora, other = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (lora if "lora_" in name else other).append(param)
    groups = []
    if lora:
        groups.append({"params": lora, "lr": lora_lr})
    if other:
        groups.append({"params": other, "lr": new_lr})
    if not groups:
        raise RuntimeError("student has no trainable parameters")
    return groups
