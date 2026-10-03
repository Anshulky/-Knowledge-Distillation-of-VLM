"""Gradio demo: upload a receipt, show the Qwen student extraction.

The checkpoint is STUDENT_MODEL. Point that at the distilled model when it is
on the Hub; the prompt and parser stay the same.

  STUDENT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct python app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import parse_output, teacher_prompt
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(1280 * 28 * 28)))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))

SUMMARY_FIELDS = (
    ("Vendor", "vendor"),
    ("Address", "vendor_address"),
    ("Tax id", "vendor_tax_id"),
    ("Buyer", "buyer"),
    ("Date", "date"),
    ("Time", "time"),
    ("Currency", "currency"),
    ("Receipt no.", "receipt_number"),
    ("Invoice no.", "invoice_number"),
    ("Subtotal", "subtotal"),
    ("Tax", "tax"),
    ("Total", "total"),
    ("Payment", "payment_method"),
)

_model = None
_processor = None


def _dtype_kw() -> dict[str, torch.dtype]:
    import transformers

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    major = int(transformers.__version__.split(".", 1)[0])
    key = "dtype" if major >= 5 else "torch_dtype"
    return {key: dtype}


def get_model():
    global _model, _processor
    if _model is None:
        kwargs = _dtype_kw()
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        _model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
        _model.eval()
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        image_processor = getattr(_processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict) and "longest_edge" in size:
            size["longest_edge"] = MAX_PIXELS
    return _model, _processor


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _summary(parsed: dict[str, Any]) -> str:
    lines = [f"**Model:** `{MODEL_ID}`"]
    doc = parsed.get("document_type")
    lang = parsed.get("language")
    if doc or lang:
        lines.append(f"**Document:** {doc or '—'} · **Language:** {lang or '—'}")
    for label, key in SUMMARY_FIELDS:
        value = parsed.get(key)
        if value not in (None, "", [], {}):
            lines.append(f"**{label}:** {value}")
    return "\n\n".join(lines)


def _items(parsed: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for item in parsed.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                item.get("name") or "",
                item.get("quantity") if item.get("quantity") is not None else "",
                item.get("unit_price") if item.get("unit_price") is not None else "",
                item.get("amount") if item.get("amount") is not None else "",
            ]
        )
    return rows


def extract(image: Image.Image | None, document_type: str | None):
    if image is None:
        raise gr.Error("Upload a receipt or invoice image.")
    hint = document_type or None
    if hint == "auto":
        hint = None
    prompt = teacher_prompt(hint)
    model, processor = get_model()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    batch = _move(dict(batch), device)
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.15,
        )
    prompt_len = int(batch["input_ids"].shape[1])
    text = processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    try:
        parsed = parse_output(text)
    except (ValueError, json.JSONDecodeError) as exc:
        return (
            f"**Model:** `{MODEL_ID}`\n\nCould not parse JSON ({exc}). The raw text is below.",
            "",
            [],
            text,
        )
    return (
        _summary(parsed),
        json.dumps(parsed, ensure_ascii=False, indent=2),
        _items(parsed),
        text,
    )


try:
    import spaces

    extract = spaces.GPU(duration=180)(extract)
except ImportError:
    pass


def build() -> gr.Blocks:
    with gr.Blocks(title="IE643 receipt extractor") as demo:
        gr.Markdown(
            f"""# Receipt and invoice extraction

Upload a photo of a receipt or invoice. The student model reads it into the shared IE643 JSON schema.
"""
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Receipt or invoice", sources=["upload", "clipboard"])
                document_type = gr.Dropdown(
                    choices=["auto", "receipt", "invoice", "other"],
                    value="auto",
                    label="Document type hint",
                )
                button = gr.Button("Extract", variant="primary")
            with gr.Column():
                summary = gr.Markdown(label="Fields")
                items = gr.Dataframe(
                    headers=["name", "quantity", "unit_price", "amount"],
                    datatype=["str", "str", "str", "str"],
                    label="Line items",
                    interactive=False,
                )
        with gr.Accordion("JSON", open=True):
            payload = gr.Code(language="json", label="Canonical JSON")
        with gr.Accordion("Raw model text", open=False):
            raw = gr.Textbox(label="Raw output", lines=8)
        button.click(extract, [image, document_type], [summary, payload, items, raw])
    return demo


if __name__ == "__main__":
    build().launch(share=True)
