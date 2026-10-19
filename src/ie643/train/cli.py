"""E3: fine-tune the student with cross-entropy on source_gt only (no teacher, no distillation).

Defaults match the teacher-label SFT / KD runs (configs/e3_sft.yaml, e4_kd_wnone.yaml) so that
only the targets differ: LoRA r16/a32 on the LLM, vision encoder and merger frozen, hint-free
prompt, 2 epochs, batch 4 x 8, lr 1e-4 cosine (3% warmup), grad clip 1.0, examples over 4096
tokens dropped, datasets sampled with P ∝ n^0.5, no validation split. Training images are the
current split's train uids (splits/*.json, content duplicates of test removed); nothing derived
from the teacher is used.

Values in source_gt are always trained; an empty label is trained as null only where the
dataset fills that field in >= --null-frac (1%) of its rows, i.e. annotates it at all; fields
the dataset never labels are masked.

  python -m ie643.train --manifest Data/processed/train/manifest.jsonl --out checkpoints/e3_v3
  (E3-v1 was: --train-merger --epochs 1 --batch-size 2 --val-frac 0.02 --balance-power 1)
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import WeightedRandomSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from ie643.labeled import labeled_fields
from ie643.pack.build import load_split_records
from ie643.teacher.cli import load_jsonl
from ie643.train.data import MAX_PIXELS, MIN_PIXELS, SFTDataset, balance_weights, collate, is_val_uid

# Attention + MLP projections of the LLM only; the vision tower reuses some of these names.
LLM_LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


class BalancedTrainer(Trainer):
    """Draws each epoch's examples with replacement from per-row weights (dataset balancing)."""

    def __init__(self, *args, sample_weights: list[float] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def _get_train_sampler(self, train_dataset=None):
        if self.sample_weights is None:
            return super()._get_train_sampler(train_dataset)
        generator = torch.Generator().manual_seed(self.args.seed)
        return WeightedRandomSampler(self.sample_weights, len(self.sample_weights), replacement=True, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: CE-only LoRA fine-tuning of the student on source_gt")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/train/manifest.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("splits"), help="train only on this split's train uids")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/e3_v3"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-merger", action="store_true", help="also train the vision merger (E3-v1 did)")
    parser.add_argument("--null-frac", type=float, default=0.01,
                        help="train an empty label as null only if the dataset fills that field in this share of rows")
    parser.add_argument("--balance-power", type=float, default=0.5, help="P(dataset) ∝ n^power; 1 = no balancing")
    parser.add_argument("--val-frac", type=float, default=0.0, help="slice of train held out for eval loss (0 = none)")
    parser.add_argument("--max-val", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=4096, help="longer examples are dropped")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=3, help="dataloader processes (image decode + processor)")
    parser.add_argument("--limit", type=int, default=None, help="train on the first N rows (smoke test)")
    parser.add_argument("--seed", type=int, default=643)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    not_train = [r["uid"] for r in rows if r.get("split") != "train"]
    if not_train:
        raise SystemExit(f"{len(not_train)} non-train rows in {args.manifest} (e.g. {not_train[0]}); refusing to train")
    split_uids = {r["uid"] for r in load_split_records(args.splits, "train")}
    n_manifest = len(rows)
    rows = [r for r in rows if r["uid"] in split_uids]
    print(f"manifest {n_manifest} rows -> {len(rows)} in {args.splits} train (content duplicates removed)", flush=True)
    null_fields = labeled_fields(rows, min_frac=args.null_frac)
    train_rows = [r for r in rows if not is_val_uid(r["uid"], args.val_frac)]
    val_rows = [r for r in rows if is_val_uid(r["uid"], args.val_frac)][: args.max_val]
    if args.limit:
        train_rows = train_rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "null_fields.json").write_text(json.dumps(null_fields, indent=2) + "\n")
    (args.out / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    print(f"empty labels trained as null (field filled in >= {args.null_frac:.0%} of the dataset's rows):", flush=True)
    for ds, keys in null_fields.items():
        print(f"  {ds}: {', '.join(keys)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LLM_LORA_TARGETS,
        modules_to_save=["merger"] if args.train_merger else None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    root = args.manifest.parent
    train_ds = SFTDataset(train_rows, root, processor, null_fields=null_fields, max_len=args.max_len)
    val_ds = SFTDataset(val_rows, root, processor, null_fields=null_fields, max_len=args.max_len) if val_rows else None
    dropped = train_ds.dropped_long + (val_ds.dropped_long if val_ds else [])
    if dropped:
        (args.out / "dropped_long.json").write_text(json.dumps([r["uid"] for r in dropped], indent=1) + "\n")
    print(f"train {len(train_ds)}  val {len(val_ds) if val_ds else 0}  dropped (> {args.max_len} tokens) {len(dropped)}", flush=True)
    weights = balance_weights([r["dataset"] for r in train_ds.rows], args.balance_power) if args.balance_power != 1 else None

    steps = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)) * args.epochs
    targs = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * steps)),
        weight_decay=0.0,
        max_grad_norm=args.grad_clip,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=args.workers,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        seed=args.seed,
    )
    trainer = BalancedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate, pad_id=processor.tokenizer.pad_token_id),
        sample_weights=weights,
    )
    last = get_last_checkpoint(str(args.out)) if any(args.out.glob("checkpoint-*")) else None
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)

    final = args.out / "adapter"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    print(f"saved LoRA adapter -> {final}", flush=True)


if __name__ == "__main__":
    main()
