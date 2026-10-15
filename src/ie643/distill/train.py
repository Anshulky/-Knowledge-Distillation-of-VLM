"""Plain PyTorch loop. No validation split: the last checkpoint is the one that is evaluated."""

from __future__ import annotations

import importlib.metadata
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ie643.distill.config import TrainConfig, dump_config
from ie643.distill.data import DistillDataset, LengthGroupedBatchSampler, collate, image_token_id
from ie643.distill.evaluate import sanity_report, sanity_uids
from ie643.distill.losses import sequence_loss
from ie643.distill.model import load_student, load_teacher, trainable_groups
from ie643.distill.relations import (
    QKVCapture,
    StopForward,
    attach_acts,
    attach_qkv,
    attach_stop,
    decoder_layers,
    install_post_rope,
    layer_index,
    relation_loss_from_qkv,
    sink_columns,
)

_STUDENT_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}


def _versions() -> dict[str, str | None]:
    out = {}
    for pkg in ("torch", "transformers", "peft", "vllm", "accelerate"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine(total: int, warmup: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def _student_batch(batch: dict, device: str, dtype: torch.dtype) -> dict:
    out: dict = {}
    for key, value in batch.items():
        if key == "uid" or key.startswith("teacher_") or not torch.is_tensor(value):
            out[key] = value
            continue
        value = value.to(device)
        if key in {"pixel_values"}:
            value = value.to(dtype)
        out[key] = value
    return out


def _model_kwargs(batch: dict, prefix: str = "") -> dict:
    keys = {prefix + k for k in _STUDENT_KEYS}
    kw = {}
    for key, value in batch.items():
        if key in keys or key.startswith(prefix + "mm_") or key.endswith("token_type_ids") and key.startswith(prefix):
            name = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            kw[name] = value
    return kw


def _span(ids: torch.Tensor, image_id: int, n_prompt: int, n_out: int, image_columns: bool) -> tuple[torch.Tensor, torch.Tensor]:
    real = ids[: n_prompt + n_out]
    img = (real == image_id).nonzero(as_tuple=False).flatten()
    outputs = torch.arange(n_prompt, n_prompt + n_out, device=ids.device)
    cols = torch.cat([img, outputs]) if image_columns else outputs
    return cols, img


def _sink_vector(acts: list[torch.Tensor], img: torch.Tensor, n_cols: int, n_img_cols: int, ratio: float, enabled: bool) -> torch.Tensor:
    sink = torch.zeros(n_cols, dtype=torch.bool, device=img.device)
    if not enabled or img.numel() == 0:
        return sink
    flagged = sink_columns(acts, img, ratio)
    sink[:n_img_cols] = flagged.to(sink.device)
    return sink


def _qkv(cap: QKVCapture, index: int, length: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cap.q is None or cap.k is None or cap.v is None:
        raise RuntimeError("Q/K/V hooks did not fire")
    return (
        cap.q[index, :length].to(device),
        cap.k[index, :length].to(device),
        cap.v[index, :length].to(device),
    )


def _relation(cfg: TrainConfig, student_cap, teacher_cap, batch, s_image: int, t_image: int, device: str) -> torch.Tensor:
    total = torch.zeros((), device=device)
    n = batch["input_ids"].shape[0]
    for b in range(n):
        n_prompt = int(batch["n_prompt"][b])
        n_out = int(batch["n_out"][b])
        t_prompt = int(batch["teacher_n_prompt"][b])
        s_cols, s_img = _span(batch["input_ids"][b], s_image, n_prompt, n_out, cfg.rel.image_columns)
        t_cols, t_img = _span(batch["teacher_input_ids"][b], t_image, t_prompt, n_out, cfg.rel.image_columns)
        s_acts = [a[b] for a in student_cap.acts]
        t_acts = [a[b].to(device) for a in teacher_cap.acts]
        s_sink = _sink_vector(s_acts, s_img, int(s_cols.shape[0]), int(s_img.shape[0]) if cfg.rel.image_columns else 0, cfg.rel.sink_ratio, cfg.rel.mask_sinks)
        t_sink = _sink_vector(t_acts, t_img, int(t_cols.shape[0]), int(t_img.shape[0]) if cfg.rel.image_columns else 0, cfg.rel.sink_ratio, cfg.rel.mask_sinks)
        if cfg.rel.image_columns and s_img.numel() == t_img.numel():
            union = s_sink[: s_img.numel()] | t_sink[: t_img.numel()].to(device)
            s_sink = s_sink.clone()
            t_sink = t_sink.clone()
            s_sink[: s_img.numel()] = union
            t_sink[: t_img.numel()] = union.to(t_sink.device)
        resample = None
        if cfg.rel.resize == "A" and cfg.rel.image_columns:
            src = tuple(int(x) for x in batch["teacher_grid_hw"][b].tolist())
            dst = tuple(int(x) for x in batch["student_grid_hw"][b].tolist())
            if src != dst:
                resample = (src, dst)
        total = total + relation_loss_from_qkv(
            _qkv(student_cap, b, n_prompt + n_out, device),
            tuple(t.detach() for t in _qkv(teacher_cap, b, t_prompt + n_out, device)),
            s_cols, t_cols.to(device), n_out, s_sink, t_sink.to(device),
            student_heads=cfg.rel.student_heads,
            student_kv_heads=cfg.rel.student_kv_heads,
            teacher_heads=cfg.rel.teacher_heads,
            teacher_kv_heads=cfg.rel.teacher_kv_heads,
            n_rel=cfg.rel.relation_heads,
            causal=cfg.rel.causal,
            resample_hw=resample,
        )
    return total / n


def _save(model, optimizer, scheduler, step: int, epoch: int, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    torch.save(
        {"step": step, "epoch": epoch, "optim": optimizer.state_dict(), "sched": scheduler.state_dict()},
        path / "trainer.pt",
    )


def _load_trainer(model, optimizer, scheduler, path: Path) -> tuple[int, int]:
    from safetensors.torch import load_file

    weights = path / "adapter_model.safetensors"
    if weights.exists():
        model.load_state_dict(load_file(weights), strict=False)
    state = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optim"])
    scheduler.load_state_dict(state["sched"])
    return int(state["step"]), int(state["epoch"])


def _prune(run_dir: Path, keep: int) -> None:
    ckpts = sorted(run_dir.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in ckpts[:-keep]:
        for child in old.glob("*"):
            child.unlink()
        old.rmdir()


def train(
    cfg: TrainConfig,
    *,
    resume: bool = False,
    limit: int | None = None,
    stop_after_steps: int | None = None,
    run_sanity: bool = True,
) -> dict:
    _seed(cfg.optim.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.jsonl").exists() and not resume and stop_after_steps is None:
        raise SystemExit(f"{run_dir} already has a run; pass --resume or use a new seed/variant")
    (run_dir / "config.yaml").write_text(dump_config(cfg), encoding="utf-8")
    (run_dir / "env.json").write_text(
        json.dumps({"git": _git_hash(), "seed": cfg.optim.seed, "versions": _versions()}, indent=2),
        encoding="utf-8",
    )

    device = cfg.student_device
    model, processor = load_student(cfg.student, cfg, device)
    teacher = teacher_processor = None
    if cfg.rel.enabled:
        teacher, teacher_processor = load_teacher(cfg.teacher, cfg.teacher_device)

    manifest = Path(cfg.data.manifest)
    dataset = DistillDataset(
        manifest,
        Path(cfg.data.label_filter),
        processor,
        soft_key=cfg.data.soft_key,
        max_length=cfg.data.max_length,
        drop_flagged=cfg.data.drop_flagged,
        eos_id=cfg.data.eos_id,
        teacher_processor=teacher_processor,
        teacher_resize=cfg.rel.resize,
        limit=limit,
    )
    if dataset.dropped_long:
        (run_dir / "dropped_long.json").write_text(json.dumps(dataset.dropped_long, indent=1), encoding="utf-8")
        print(f"dropped {len(dataset.dropped_long)} rows longer than {cfg.data.max_length}")
    if len(dataset) == 0:
        raise SystemExit("no training rows after the label filter and length cap")
    epoch_size = cfg.data.epoch_size or len(dataset)
    sampler = LengthGroupedBatchSampler(
        [row.length for row in dataset.rows],
        dataset.datasets,
        cfg.optim.micro_batch,
        epoch_size,
        power=cfg.data.weight_power,
        weighting=cfg.data.dataset_weighting,
        seed=cfg.optim.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda rows: collate(rows, cfg.data.pad_id),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        trainable_groups(model, cfg.optim.lr, cfg.optim.new_lr),
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    opt_per_epoch = math.ceil(len(loader) / cfg.optim.grad_accum)
    total_steps = opt_per_epoch * cfg.optim.epochs
    warmup = max(1, int(total_steps * cfg.optim.warmup_ratio)) if total_steps else 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine(total_steps, warmup))

    step, start_epoch = 0, 0
    if resume:
        ckpts = sorted(run_dir.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            step, start_epoch = _load_trainer(model, optimizer, scheduler, ckpts[-1])
            print(f"resumed {ckpts[-1].name} at step {step}")

    s_image = image_token_id(processor)
    t_image = image_token_id(teacher_processor) if teacher_processor is not None else s_image
    student_layers = decoder_layers(model)
    s_layer = layer_index(len(student_layers), cfg.rel.student_layer) if cfg.rel.enabled else None
    teacher_layers = decoder_layers(teacher) if teacher is not None else []
    t_layer = layer_index(len(teacher_layers), cfg.rel.teacher_layer) if teacher is not None else None
    if cfg.rel.enabled and not cfg.rel.pre_rope:
        install_post_rope(student_layers[s_layer])
        install_post_rope(teacher_layers[t_layer])

    metrics_path = run_dir / "metrics.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window_tokens = 0
    window_start = time.time()
    last = {"ce": None, "kd": None, "rel": None, "loss": None}
    micro_in_accum = 0
    dtype = next(model.parameters()).dtype

    for epoch in range(start_epoch, cfg.optim.epochs):
        sampler.epoch = epoch
        for batch in loader:
            if stop_after_steps is not None and step >= stop_after_steps:
                break
            batch = _student_batch(batch, device, dtype)
            if teacher is not None:
                for key, value in list(batch.items()):
                    if key.startswith("teacher_") and torch.is_tensor(value):
                        moved = value.to(cfg.teacher_device)
                        if key == "teacher_pixel_values":
                            moved = moved.to(dtype)
                        batch[key] = moved
            student_cap, teacher_cap = QKVCapture(), QKVCapture()
            handles: list = []
            try:
                if cfg.rel.enabled:
                    handles += attach_qkv(student_layers[s_layer].self_attn, student_cap, pre_rope=cfg.rel.pre_rope)
                    handles += attach_acts(student_layers[: s_layer + 1], student_cap)
                    if not cfg.rel.pre_rope:
                        import sys

                        holder = sys.modules[type(student_layers[s_layer]).__module__]._ie643_rope_holder
                        holder[0] = student_cap
                        handles.append(student_layers[s_layer].register_forward_pre_hook(lambda *_a, **_k: student_cap.rope_on.__setitem__(0, True), with_kwargs=True))
                        handles.append(student_layers[s_layer].register_forward_hook(lambda *_a: student_cap.rope_on.__setitem__(0, False)))
                autocast_device = device.split(":")[0]
                with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=autocast_device == "cuda"):
                    outputs = model(**_model_kwargs(batch), use_cache=False)
                    parts = sequence_loss(
                        outputs.logits, batch["labels"], batch["value_mask"], batch["n_prompt"], batch["n_out"],
                        batch["topk_ids"], batch["topk_logprobs"],
                        kd_enabled=cfg.kd.enabled, alpha=cfg.kd.alpha, tau=cfg.kd.tau,
                        div=cfg.kd.div, skew=cfg.kd.skew, weight=cfg.kd.weight,
                    )
                    rel = outputs.logits.new_zeros(())
                    if cfg.rel.enabled:
                        if not cfg.rel.pre_rope:
                            import sys

                            sys.modules[type(teacher_layers[t_layer]).__module__]._ie643_rope_holder[0] = teacher_cap
                            teacher_cap.rope_on[0] = True
                        handles += attach_qkv(teacher_layers[t_layer].self_attn, teacher_cap, pre_rope=cfg.rel.pre_rope)
                        handles += attach_acts(teacher_layers[: t_layer + 1], teacher_cap)
                        handles += attach_stop(teacher_layers[t_layer])
                        try:
                            with torch.no_grad():
                                teacher(**_model_kwargs(batch, "teacher_"), use_cache=False)
                        except StopForward:
                            pass
                        rel = _relation(cfg, student_cap, teacher_cap, batch, s_image, t_image, device)
                    loss = parts["loss"] + cfg.rel.beta * rel
                (loss / cfg.optim.grad_accum).backward()
            finally:
                for handle in handles:
                    handle.remove()
                student_cap.clear()
                teacher_cap.clear()
            micro_in_accum += 1
            window_tokens += int(batch["n_out"].sum())
            last = {
                "ce": float(parts["ce"].detach()),
                "kd": float(parts["kd"].detach()),
                "rel": float(rel.detach()),
                "loss": float(loss.detach()),
            }
            if micro_in_accum < cfg.optim.grad_accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                elapsed = max(time.time() - window_start, 1e-6)
                peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if torch.cuda.is_available() else 0.0
                row = {
                    "step": step,
                    "epoch": epoch,
                    **last,
                    "grad_norm": float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                    "tokens_per_s": window_tokens / elapsed,
                    "peak_mem_gb": peak,
                }
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(
                    f"step {step} epoch {epoch} loss {row['loss']:.4f} ce {row['ce']:.4f} "
                    f"kd {row['kd']:.4f} rel {row['rel']:.4f} lr {row['lr']:.2e} "
                    f"{row['tokens_per_s']:.0f} tok/s peak {peak:.1f} GB"
                )
                window_tokens = 0
                window_start = time.time()
            if cfg.save_every and step % cfg.save_every == 0:
                _save(model, optimizer, scheduler, step, epoch, run_dir / f"ckpt-{step}")
                _prune(run_dir, cfg.keep_ckpts)
            if stop_after_steps is not None and step >= stop_after_steps:
                break
        if micro_in_accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_in_accum = 0
            step += 1
        if stop_after_steps is not None and step >= stop_after_steps:
            break
        _save(model, optimizer, scheduler, step, epoch, run_dir / f"ckpt-{step}")
        _prune(run_dir, cfg.keep_ckpts)
        if run_sanity:
            try:
                chosen = sanity_uids([row.rec["uid"] for row in dataset.rows], cfg.sanity_n)
                by_uid = {row.rec["uid"]: row.rec for row in dataset.rows}
                report = sanity_report(
                    model, processor, [by_uid[u] for u in chosen], manifest.parent,
                    raw_key=cfg.data.raw_key, max_new_tokens=cfg.sanity_max_tokens, device=device,
                )
                report["epoch"] = epoch
                (run_dir / "sanity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"sanity epoch {epoch}: parse {report['parse_rate']:.2f} exact {report['exact_match']:.2f}")
            except Exception as exc:
                print(f"sanity epoch {epoch} failed: {exc}")
            model.train()

    _save(model, optimizer, scheduler, step, cfg.optim.epochs, run_dir / "final")
    print(f"wrote {run_dir / 'final'}")
    return {"step": step, **last, "run_dir": str(run_dir)}
