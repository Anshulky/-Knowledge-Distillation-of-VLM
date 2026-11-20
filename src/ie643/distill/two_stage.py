"""E5-TS (proposed): two-stage relation distillation in MiniLMv2's distil-then-fine-tune order.

Stage 1 (``configs/e5ts_stage1_relation.yaml``, 2 epochs, plain ``ie643.distill train``)
trains the LoRA mainly on the MiniLMv2 relation loss: beta 1, where the relation gradient
is about 27x the CE gradient. Stage 2 (this module, ``configs/e5ts_stage2_kd.yaml``,
2 epochs) continues from that adapter with the E4 KD objective (hard + soft labels) and
no relation loss. The optimizer and LR schedule start fresh, as in a fine-tuning phase.

Nothing in ``train.py`` changes: this module replaces ``load_student`` for this process
only, so the stage-1 adapter is loaded right after the LoRA is built.

  python -m ie643.distill.two_stage --config configs/e5ts_stage2_kd.yaml \
      --init-adapter runs/e5ts/stage1/seed643/final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ie643.distill import train as train_mod
from ie643.distill.config import load_config


def _with_init_adapter(init: Path):
    original = train_mod.load_student
    saved = json.loads((init / "adapter_config.json").read_text(encoding="utf-8"))

    def load_student(model_id: str, cfg, device: str):
        model, processor = original(model_id, cfg, device)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        live = model.peft_config["default"]
        if int(saved["r"]) != int(live.r) or int(saved["lora_alpha"]) != int(live.lora_alpha):
            raise SystemExit(f"LoRA shape differs: stage 1 r={saved['r']} alpha={saved['lora_alpha']}, "
                             f"stage 2 r={live.r} alpha={live.lora_alpha}")
        if set(saved["target_modules"]) != set(live.target_modules):
            raise SystemExit("LoRA target modules differ between stage 1 and stage 2")

        state = load_file(init / "adapter_model.safetensors")
        result = set_peft_model_state_dict(model, state)
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            raise SystemExit(f"{len(unexpected)} stage-1 tensors did not match the model, e.g. {unexpected[:3]}")
        # lora_B starts at zero in a fresh LoRA, so non-zero lora_B proves the load happened
        lora_b = [p for name, p in model.named_parameters() if "lora_B" in name]
        nonzero = sum(int(p.detach().abs().sum().item() > 0) for p in lora_b)
        if nonzero == 0:
            raise SystemExit("stage-1 adapter did not load: every lora_B is still zero")
        print(f"loaded stage-1 adapter {init}: {len(state)} tensors, {nonzero}/{len(lora_b)} lora_B non-zero")
        return model, processor

    return load_student


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 2 of E5-TS: KD from a stage-1 relation adapter")
    parser.add_argument("--config", type=Path, required=True, help="stage-2 config (KD on, relation off)")
    parser.add_argument("--init-adapter", type=Path, required=True, help="stage-1 adapter folder (…/final)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    init = args.init_adapter
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (init / name).is_file():
            raise SystemExit(f"missing {init / name}")
    cfg = load_config(args.config)
    if cfg.rel.enabled:
        raise SystemExit("stage 2 must not use the relation loss (rel.enabled: false)")
    if not cfg.kd.enabled:
        raise SystemExit("stage 2 is defined as KD (kd.enabled: true)")

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "init_adapter.json").write_text(
        json.dumps({"init_adapter": str(init.resolve()), "config": str(args.config)}, indent=2),
        encoding="utf-8",
    )
    train_mod.load_student = _with_init_adapter(init)
    train_mod.train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
