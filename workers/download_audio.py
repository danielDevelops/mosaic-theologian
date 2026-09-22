"""Download sermon MP3s for queued messages.

Downloads are streamed to a .part file and renamed on success, so a kill
mid-transfer never leaves a truncated MP3 that would later be transcribed as
if it were complete.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import requests

from lib.state import STATUS_FAILED, STATUS_PENDING
from workers.common import WorkerContext, base_parser, log


def download(session: requests.Session, url: str, target: Path,
             timeout: int, min_bytes: int, max_bytes: int) -> tuple[bool, str]:
    tmp = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                return False, f"http {response.status_code}"

            declared = int(response.headers.get("Content-Length") or 0)
            if declared and declared > max_bytes:
                return False, f"too large ({declared} bytes)"

            written = 0
            with open(tmp, "wb") as handle:
                for block in response.iter_content(chunk_size=1 << 16):
                    if not block:
                        continue
                    written += len(block)
                    if written > max_bytes:
                        handle.close()
                        tmp.unlink(missing_ok=True)
                        return False, "exceeded max bytes mid-stream"
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())

        if written < min_bytes:
            tmp.unlink(missing_ok=True)
            return False, f"suspiciously small ({written} bytes)"

        os.replace(tmp, target)
        return True, ""

    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        return False, str(exc)[:200]


def main() -> int:
    parser = base_parser("Download sermon audio")
    args = parser.parse_args()

    with WorkerContext(args) as ctx:
        site = ctx.settings["Site"]
        audio_cfg = ctx.settings["Audio"]
        min_bytes = int(audio_cfg.get("MinBytes", 20480))
        max_bytes = int(audio_cfg.get("MaxBytes", 300 * 1024 * 1024))
        timeout = int(site.get("TimeoutSeconds", 45))
        delay_min = float(site.get("DelaySecondsMin", 2.0))
        delay_max = float(site.get("DelaySecondsMax", 5.0))

        session = requests.Session()
        session.headers.update({"User-Agent": site["UserAgent"]})

        candidates = [
            job for job in ctx.jobs
            if job.audio_url
            and (ctx.force or not job.at_least("audio_downloaded"))
            and not job.at_least("transcribed")
        ]

        if not candidates:
            log("No audio to download.")
            return 0

        log(f"NOW: download {len(candidates)} audio file(s).")
        processed = 0
        total = len(candidates)

        for job in candidates:
            if ctx.should_stop(processed):
                log(f"Stopping with {total - processed} download(s) still queued.")
                break

            target = ctx.paths.audio / f"{job.id}.mp3"
            label = f"{job.date or '????-??-??'}  {job.title or job.id}"

            if target.exists() and target.stat().st_size >= min_bytes and not ctx.force:
                job.audio_path = ctx.relative(target)
                job.advance("audio_downloaded")
                ctx.jobs.put(job)
                log(f"  [{processed + 1}/{total}] already on disk  {label}")
                processed += 1
                continue

            log(f"  [{processed + 1}/{total}] downloading  {label}")

            ok, error = download(session, job.audio_url, target,
                                 timeout, min_bytes, max_bytes)
            processed += 1

            if ok:
                size_mb = target.stat().st_size / (1024 * 1024)
                job.audio_path = ctx.relative(target)
                job.status = STATUS_PENDING
                job.error = ""
                job.advance("audio_downloaded")
                ctx.jobs.put(job)
                log(f"  ok   {job.id} ({size_mb:.1f} MB)")
            else:
                # A missing MP3 is normal for older years. Keep the page text
                # and let the item proceed instead of blocking the pipeline.
                job.error = f"audio: {error}"
                job.status = STATUS_FAILED if "http 4" not in error else STATUS_PENDING
                if "http 404" in error:
                    job.advance("transcribed")
                    job.status = STATUS_PENDING
                    job.error = "no audio published; indexing page text only"
                ctx.jobs.put(job)
                log(f"  skip {job.id}: {error}")

            time.sleep(random.uniform(delay_min, delay_max))

        log(f"Audio pass complete: {processed} attempted.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
