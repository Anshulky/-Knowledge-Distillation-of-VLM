"""Local review UI for the 100-case teacher pack."""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ie643.preprocess.pipeline import write_jsonl
from ie643.schema import coerce_canonical
from ie643.teacher.cli import load_jsonl

PAGE = Path(__file__).with_name("page.html")


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, root: Path, review_path: Path, **kwargs):
        self.root = root
        self.review_path = review_path
        self.manifest_path = root / "manifest.jsonl"
        super().__init__(*args, directory=str(root), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/files/") or self.path.startswith("/api/"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            data = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/samples":
            rows = load_jsonl(self.review_path)
            payload = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path.startswith("/files/"):
            rel = parsed.path[len("/files/") :]
            self.path = "/" + rel
            return SimpleHTTPRequestHandler.do_GET(self)
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        uid = body["uid"]
        review_rows = load_jsonl(self.review_path)
        updated = None
        for rec in review_rows:
            if rec["uid"] != uid:
                continue
            rec["review_status"] = body.get("review_status") or "corrected"
            rec["review_gt"] = coerce_canonical(body.get("review_gt") or {})
            rec["review_notes"] = body.get("review_notes") or ""
            updated = rec
            break
        if updated is None:
            self.send_error(404, "unknown uid")
            return
        write_jsonl(self.review_path, review_rows)
        if self.manifest_path.exists():
            manifest = load_jsonl(self.manifest_path)
            for rec in manifest:
                if rec["uid"] == uid:
                    rec["review_status"] = updated["review_status"]
                    rec["review_gt"] = updated["review_gt"]
                    rec["review_notes"] = updated["review_notes"]
            write_jsonl(self.manifest_path, manifest)
        payload = json.dumps(updated, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review teacher JSON in the browser")
    parser.add_argument("--root", type=Path, default=Path("Data/processed/test"))
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    review = args.review or (args.root / "review_100.jsonl")
    handler = partial(ReviewHandler, root=args.root.resolve(), review_path=review.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"review UI http://{args.host}:{args.port}  ({review})")
    server.serve_forever()


if __name__ == "__main__":
    main()
