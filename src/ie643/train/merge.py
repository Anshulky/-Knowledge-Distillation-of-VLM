"""Fold a trained LoRA adapter into the base student so vLLM serves it like any model.

  python -m ie643.train.merge --adapter checkpoints/e3/adapter --out checkpoints/e3/merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base student")
    parser.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    merged.save_pretrained(str(args.out))
    # Adapter dir holds the processor saved at train time (same min/max pixels as ie643.pack);
    # adapters trained elsewhere may ship without one, then the base model's is used.
    has_processor = (args.adapter / "preprocessor_config.json").exists()
    AutoProcessor.from_pretrained(str(args.adapter) if has_processor else args.base).save_pretrained(str(args.out))
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
