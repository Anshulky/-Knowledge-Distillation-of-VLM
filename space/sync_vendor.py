"""Copy the training-repo code the Space needs into space/vendor/ (run before every Space upload).

  python space/sync_vendor.py

The Space must use exactly the prompt, JSON parser and image resize the student was trained with:
  ie643/schema.py              teacher_prompt() (hint-free) + canonical schema
  ie643/teacher/parse.py       parse_teacher_output (JSON repair, canonical coercion)
  ie643/preprocess/image.py    canonicalize_image (EXIF, RGB, Qwen smart_resize <= 1,003,520 px, LANCZOS)
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
VENDOR = HERE / "vendor"
FILES = ("ie643/schema.py", "ie643/teacher/parse.py", "ie643/preprocess/image.py")


def main() -> None:
    if VENDOR.exists():
        shutil.rmtree(VENDOR)
    manifest = {}
    for rel in FILES:
        src, dst = SRC / rel, VENDOR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[rel] = hashlib.sha1(src.read_bytes()).hexdigest()
    # empty package markers: the repo's real __init__ files import heavier modules the Space does not need
    for pkg in ("ie643", "ie643/teacher", "ie643/preprocess"):
        (VENDOR / pkg / "__init__.py").write_text('"""Vendored from the IE643 training repo by sync_vendor.py."""\n')
    (VENDOR / "VENDORED.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"vendored {len(FILES)} files -> {VENDOR}")


if __name__ == "__main__":
    main()
