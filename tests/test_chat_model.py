"""The chat GGUF download used by a full export.

    python tests/test_chat_model.py
"""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.chat_model import download_gguf, ensure_chat_model, is_usable_gguf  # noqa: E402

PAYLOAD = b"GGUF" + b"x" * 28


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        start = 0
        status = 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            start = int(range_header.split("=", 1)[1].split("-", 1)[0] or 0)
            status = 206
        body = PAYLOAD[start:]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def check(name: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/model.gguf"
    root = Path(__file__).resolve().parent / "_chat_model_tmp"
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                path.unlink()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "models" / "chat.gguf"
    settings = {
        "Chat": {
            "ModelFile": "models/chat.gguf",
            "SourceUrl": url,
            "Bytes": len(PAYLOAD),
            "MinBytes": 8,
            "DownloadTimeoutSeconds": 10,
        },
        "Site": {"UserAgent": "test"},
    }
    ok = True
    try:
        download_gguf(url, target, 8, len(PAYLOAD), 10, "test")
        ok &= check("download writes a gguf", is_usable_gguf(target, 8) and target.stat().st_size == len(PAYLOAD))

        part = target.with_name(target.name + ".part")
        target.unlink()
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(PAYLOAD[:10])
        download_gguf(url, target, 8, len(PAYLOAD), 10, "test")
        ok &= check("resume finishes the file", target.is_file() and target.read_bytes() == PAYLOAD)
        ok &= check("part file is removed", not part.exists())

        found = ensure_chat_model(settings, root)
        ok &= check("existing file is kept", found == target.resolve() and target.read_bytes() == PAYLOAD)
    finally:
        server.shutdown()
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
