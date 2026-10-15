"""Training config. Every field has a default; YAML only needs the overrides.

  python -m ie643.distill train --config configs/e3_sft.yaml
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class KDConfig:
    """E4 logit KD. Disabled for E3 (CE only) and for a pure E5 run."""

    enabled: bool = False
    alpha: float = 0.5
    tau: float = 1.0
    div: str = "fkl"  # fkl | rkl | skl
    skew: float = 0.1
    weight: str = "entropy"  # none | entropy | value


@dataclass
class RelConfig:
    """E5 MiniLMv2 relations. Layer numbers are 1-based (36 = the last student layer)."""

    enabled: bool = False
    beta: float = 1.0
    relation_heads: int = 32
    student_layer: int = 36
    teacher_layer: int = 48
    pre_rope: bool = True
    mask_sinks: bool = True
    sink_ratio: float = 10.0
    causal: bool = True
    image_columns: bool = True  # False -> S = output tokens only (E5-O)
    resize: str = "B"  # B: teacher image * 8/7; A: packed image + resample columns
    student_heads: int = 16
    student_kv_heads: int = 2
    teacher_heads: int = 64
    teacher_kv_heads: int = 8


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    merger: bool = False  # E3c: also LoRA the vision-merger MLP


@dataclass
class OptimConfig:
    lr: float = 1e-4
    new_lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    epochs: int = 2
    micro_batch: int = 4
    grad_accum: int = 8
    grad_clip: float = 1.0
    seed: int = 643


@dataclass
class DataConfig:
    manifest: str = "Data/processed/train/manifest.jsonl"
    label_filter: str = "Data/processed/train/label_filter.json"
    soft_key: str = "teacher_soft"
    raw_key: str = "teacher_raw"
    max_length: int = 4096
    epoch_size: int | None = None  # None = one draw per kept row
    dataset_weighting: bool = True
    weight_power: float = 0.5
    drop_flagged: bool = False  # E3d
    pad_id: int = 151643
    eos_id: int = 151645


@dataclass
class TrainConfig:
    exp: str = "e3"
    variant: str = "main"
    student: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    teacher: str = "Qwen/Qwen3-VL-32B-Instruct"
    runs_root: str = "runs"
    log_every: int = 20
    save_every: int = 1000
    keep_ckpts: int = 3
    sanity_n: int = 32
    sanity_max_tokens: int = 1024
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"
    kd: KDConfig = None  # type: ignore[assignment]
    rel: RelConfig = None  # type: ignore[assignment]
    lora: LoRAConfig = None  # type: ignore[assignment]
    optim: OptimConfig = None  # type: ignore[assignment]
    data: DataConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kd is None:
            self.kd = KDConfig()
        if self.rel is None:
            self.rel = RelConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.optim is None:
            self.optim = OptimConfig()
        if self.data is None:
            self.data = DataConfig()

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.exp / self.variant / f"seed{self.optim.seed}"


def _coerce(hint: Any, value: Any) -> Any:
    origin = getattr(hint, "__origin__", None)
    if hint is tuple or origin is tuple:
        return tuple(value)
    if is_dataclass(hint) and isinstance(value, dict):
        return _fill(hint, value)
    return value


def _fill(cls: type, data: dict[str, Any] | None) -> Any:
    if data is None:
        return cls()
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints.get(name), value)
    return cls(**kwargs)


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    return _fill(TrainConfig, raw)


def dump_config(cfg: TrainConfig) -> str:
    return yaml.safe_dump(asdict(cfg), sort_keys=False)
