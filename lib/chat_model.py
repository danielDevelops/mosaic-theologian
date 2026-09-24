"""Make sure the chat GGUF is on disk before a full export copies it.

The night job does not train this file. A full export downloads the
configured Llama GGUF when models/chat.gguf is missing or incomplete.
A file already there is kept, including one you downloaded yourself.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"
_CHUNK = 1 << 20
_LOG_EVERY = 100 * 1024 * 1024


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _log(message: str) -> None:
    # Imported lazily so a unit test can call the downloader without the
    # worker logging setup.
    from workers.common import log
    log(message)


def chat_model_path(settings: dict[str, Any], root: Path) -> Path:
    relative = settings["Chat"].get("ModelFile", "models/chat.gguf")
    return (root / relative).resolve()


def is_usable_gguf(path: Path, min_bytes: int) -> bool:
    """True when the file is large enough and starts with the GGUF magic.

    The magic sits at byte 0, so a partial download looks valid unless the
    size check rejects it.
    """
    if not path.is_file():
        return False
    if path.stat().st_size < min_bytes:
        return False
    with path.open("rb") as handle:
        return handle.read(4) == GGUF_MAGIC


def _announced_total(response, already: int) -> int | None:
    content_range = response.headers.get("Content-Range") or ""
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total.isdigit():
            return int(total)
    length = response.headers.get("Content-Length")
    if length and str(length).isdigit():
        status = getattr(response, "status", 200) or 200
        extra = already if status == 206 else 0
        return int(length) + extra
    return None


def download_gguf(url: str, target: Path, min_bytes: int, expected_bytes: int,
                  timeout: int, user_agent: str) -> None:
    """Stream the GGUF to a .part file and rename it only when complete."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    already = part.stat().st_size if part.is_file() else 0

    headers = {"User-Agent": user_agent}
    if already:
        headers["Range"] = f"bytes={already}-"
        _log(f"  resuming chat model at {_human(already)}")

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and is_usable_gguf(part, min_bytes):
            os.replace(part, target)
            return
        raise RuntimeError(
            f"Chat model download failed: HTTP {exc.code} for {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Chat model download failed: {exc.reason}") from exc

    with response:
        status = getattr(response, "status", 200) or 200
        if status == 200:
            already = 0
        elif status not in (200, 206):
            raise RuntimeError(f"Chat model download failed: HTTP {status}")

        total = _announced_total(response, already if status == 206 else 0)
        if expected_bytes and total and total != expected_bytes:
            _log(f"  server size {_human(total)} differs from "
                 f"configured {_human(expected_bytes)}; using the server size")
        goal = total or expected_bytes or 0

        written = already
        last_logged = written
        with open(part, "ab" if status == 206 else "wb") as handle:
            while True:
                block = response.read(_CHUNK)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                if written - last_logged >= _LOG_EVERY:
                    if goal:
                        _log(f"  chat model {_human(written)} / {_human(goal)}")
                    else:
                        _log(f"  chat model {_human(written)}")
                    last_logged = written
            handle.flush()
            os.fsync(handle.fileno())

    if goal and written != goal:
        raise RuntimeError(
            f"Chat model download stopped at {_human(written)}"
            + (f" of {_human(goal)}" if goal else "")
            + ". The partial file is kept and the next export resumes it."
        )
    if not is_usable_gguf(part, min_bytes):
        raise RuntimeError(
            "Downloaded file is not a complete GGUF. "
            "Check Chat.SourceUrl and Chat.MinBytes."
        )
    os.replace(part, target)


def ensure_chat_model(settings: dict[str, Any], root: Path) -> Path:
    """Return the local GGUF, downloading it when the file is not usable."""
    cfg = settings["Chat"]
    target = chat_model_path(settings, root)
    min_bytes = int(cfg.get("MinBytes", 5_600_000_000))
    expected = int(cfg.get("Bytes", 0) or 0)
    if is_usable_gguf(target, min_bytes):
        _log(f"Chat model present: {target.name} ({_human(target.stat().st_size)})")
        return target

    url = str(cfg.get("SourceUrl") or "").strip()
    if not url:
        raise RuntimeError(
            f"Chat model missing at {target} and Chat.SourceUrl is empty."
        )

    timeout = int(cfg.get("DownloadTimeoutSeconds", 180))
    user_agent = str(settings.get("Site", {}).get("UserAgent") or "MosaicTheologian/1.0")
    _log(f"Chat model missing. Downloading {_human(expected) if expected else 'GGUF'}")
    _log(f"  {url}")
    download_gguf(url, target, min_bytes, expected, timeout, user_agent)
    _log(f"Chat model saved: {target} ({_human(target.stat().st_size)})")
    return target
