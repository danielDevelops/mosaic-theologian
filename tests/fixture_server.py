"""Serve captured pages over loopback so the crawler can be tested offline.

The point is fidelity: the real crawler code runs unchanged against this, so
robots handling, link extraction, canonicalisation, priority ordering and
audio download are all genuinely exercised. Only the origin changes.

Absolute links inside the fixtures are rewritten to this server on the way
out, which is what lets the sermon audio URLs resolve locally too.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pages"

REAL_SITE = "https://thisismosaic.org"
REAL_PODCAST = "https://podcast.thisismosaic.org"

ROBOTS = b"""User-agent: *
Disallow: /calendar/*?
Disallow: /community-groups/join-a-group/*?
"""

# Large enough to clear the Audio.MinBytes sanity check.
FAKE_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + (b"\x00" * 65536)


def slug_for(path: str) -> str:
    cleaned = path.strip("/")
    return (cleaned.replace("/", "__") or "index") + ".html"


class Handler(BaseHTTPRequestHandler):
    base_url = ""
    hits: dict[str, int] = {}

    def log_message(self, *args) -> None:  # keep test output readable
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        Handler.hits[path] = Handler.hits.get(path, 0) + 1

        if path == "/robots.txt":
            self._send(200, ROBOTS, "text/plain")
            return

        if path.startswith("/podcast/") and path.endswith(".mp3"):
            self._send(200, FAKE_MP3, "audio/mpeg")
            return

        fixture = FIXTURES / slug_for(path)
        # /core-beliefs and /about/core-beliefs are the same page.
        if not fixture.is_file() and path.rstrip("/") == "/core-beliefs":
            fixture = FIXTURES / slug_for("/about/core-beliefs/")
        if not fixture.is_file():
            self._send(404, b"not found", "text/plain")
            return

        html = fixture.read_text(encoding="utf-8")
        html = html.replace(REAL_PODCAST, f"{self.base_url}/podcast")
        html = html.replace(REAL_SITE, self.base_url)
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")


class FixtureSite:
    """Context manager that runs the fixture server on a free port."""

    def __init__(self) -> None:
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.base_url = ""

    def __enter__(self) -> "FixtureSite":
        Handler.hits = {}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}"
        Handler.base_url = self.base_url

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    @property
    def hits(self) -> dict[str, int]:
        return dict(Handler.hits)

    def hit_count(self, path: str) -> int:
        return Handler.hits.get(path, 0)


def available_fixtures() -> list[str]:
    if not FIXTURES.is_dir():
        return []
    return sorted(p.name for p in FIXTURES.glob("*.html"))
