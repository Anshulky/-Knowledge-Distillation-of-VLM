"""E3: fine-tune the student with cross-entropy on source_gt only (no teacher, no distillation).

LoRA on the language model, vision encoder frozen, patch merger trained in full
(slide 8, stage 1). Only answer tokens are supervised; fields a dataset does not
annotate are masked out of the loss unless --no-mask-unlabeled.

  python -m ie643.train --manifest Data/processed/train/manifest.jsonl --out checkpoints/e3
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from ie643.labeled import labeled_fields
from ie643.teacher.cli import load_jsonl
from ie643.train.data import MAX_PIXELS, MIN_PIXELS, SFTDataset, collate, is_val_uid

# Attention + MLP projections of the LLM only; the vision tower reuses some of these names.
LLM_LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: CE-only LoRA fine-tuning of the student on source_gt")
    parser.add_argument("--manifest", type=Path, default=Path("Data/processed/train/manifest.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/e3"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-train-merger", dest="train_merger", action="store_false")
    parser.add_argument("--no-mask-unlabeled", dest="mask_unlabeled", action="store_false")
    parser.add_argument("--val-frac", type=float, default=0.02, help="slice of train held out for eval loss")
    parser.add_argument("--max-val", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=6144)
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
    labeled = labeled_fields(rows)
    train_rows = [r for r in rows if not is_val_uid(r["uid"], args.val_frac)]
    val_rows = [r for r in rows if is_val_uid(r["uid"], args.val_frac)][: args.max_val]
    if args.limit:
        train_rows = train_rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labeled_fields.json").write_text(json.dumps(labeled, indent=2) + "\n")
    (args.out / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    print(f"train {len(train_rows)}  val {len(val_rows)}  mask_unlabeled={args.mask_unlabeled}", flush=True)
    for ds, keys in labeled.items():
        print(f"  labeled[{ds}] = {', '.join(keys)}", flush=True)

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
    mask = labeled if args.mask_unlabeled else None
    train_ds = SFTDataset(train_rows, root, processor, labeled=mask, max_len=args.max_len)
    val_ds = SFTDataset(val_rows, root, processor, labeled=mask, max_len=args.max_len)

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
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
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
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate, pad_id=processor.tokenizer.pad_token_id),
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
