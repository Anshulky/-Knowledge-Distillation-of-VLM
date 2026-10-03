---
title: IE643 Receipt Extractor
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# IE643 receipt extractor

Upload a receipt or invoice. A Qwen-VL student returns the shared extraction JSON (vendor, line items, tax, total, and the rest of the IE643 fields).

The default checkpoint is `Qwen/Qwen2.5-VL-3B-Instruct`. Set the Space variable **STUDENT_MODEL** to the distilled Hub repo when that model is ready. The prompt and parser do not change.

Hardware: a GPU Space (T4 or ZeroGPU). The base student is about 7GB in bf16.

```bash
# from the training repo, publish this folder as the Space root
huggingface-cli upload <user>/ie643-student space . --repo-type space
```

Locally:

```bash
pip install -r requirements.txt
STUDENT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct python app.py
```
