"""Capture a small set of real pages once, for offline testing.

Run this rarely. Everything else in tests/ works from what it saves, so the
pipeline can be exercised end to end in seconds without touching the church's
server. Fixtures are gitignored: they are a local cache, not source.

    python tests/capture_fixtures.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pages"

BASE = "https://thisismosaic.org"
UA = "MosaicTheologian/1.0 (personal offline study index)"

# Chosen to cover every branch the crawler has:
#   - the archive, which enumerates all series
#   - two series index pages
#   - two messages that are the SAME sermon under different series paths
#   - a message from a different series
#   - a non-message content page
PATHS = [
    "/messages/archive/",
    "/messages/",
    "/messages/1st-john/",
    "/messages/hebrews/",
    "/messages/1st-john/1-john-1.3-4/",
    "/messages/the-letters-of-john/1-john-1.3-4-2/",
    "/messages/hebrews/hebrews-4.14-16/",
    "/about/core-beliefs/",
    "/core-beliefs/",
    "/about/",
    "/mission-vision-values/",
]


def slug_for(path: str) -> str:
    cleaned = path.strip("/")
    return (cleaned.replace("/", "__") or "index") + ".html"


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    saved = 0
    for path in PATHS:
        target = FIXTURES / slug_for(path)
        if target.exists():
            print(f"  have  {path}")
            continue

        url = BASE + path
        try:
            response = session.get(url, timeout=45)
        except requests.RequestException as exc:
            print(f"  FAIL  {path}: {exc}")
            continue

        if response.status_code != 200:
            print(f"  FAIL  {path}: http {response.status_code}")
            continue

        target.write_text(response.text, encoding="utf-8")
        saved += 1
        print(f"  saved {path}  ({len(response.text):,} bytes)")
        time.sleep(3)

    print(f"\n{saved} new fixture(s) in {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
