"""Transcribe downloaded sermon audio with faster-whisper on the GPU.

CUDA is pinned from .env before the model loads, so a second toolkit on
PATH cannot supply the wrong cublas/cudnn DLL. GPU failure stops the
run instead of silently falling back to CPU for the rest of the night.

Audio is deleted after a successful transcript. The transcript JSON is kept.
"""

from __future__ import annotations

import time
from pathlib import Path

from lib.cuda_env import apply_cuda_env
from lib.state import STATUS_FAILED, STATUS_PENDING, atomic_write_json
from workers.common import WorkerContext, base_parser, log

_model = None
_device_used = ""


def load_model(settings: dict):
    """Build the model once per process. CUDA is required unless opted out."""
    global _model, _device_used
    if _model is not None:
        return _model

    pinned = apply_cuda_env()
    if pinned.get("cuda_path"):
        log(f"CUDA pinned: {pinned['cuda_version']}  {pinned['cuda_path']}")
    else:
        log(f"CUDA version pin: {pinned.get('cuda_version', '?')} "
            "(no toolkit folder; using venv NVIDIA DLLs)")
    if pinned.get("prepended"):
        log("DLL search: " + " | ".join(pinned["prepended"]))

    from faster_whisper import WhisperModel

    cfg = settings["Whisper"]
    name = cfg["Model"]
    want = cfg.get("Device", "cuda")
    fail_if_no_gpu = bool(cfg.get("FailIfNoGpu", True))

    if want == "cuda":
        try:
            _model = WhisperModel(name, device="cuda",
                                  compute_type=cfg.get("ComputeType", "float16"))
            _device_used = "cuda"
            log(f"Whisper model '{name}' loaded on GPU.")
            return _model
        except Exception as exc:
            log(f"ERROR: GPU unavailable for faster-whisper "
                f"({type(exc).__name__}: {str(exc)[:240]})")
            log("ERROR: Check .env CUDA_VERSION / CUDA_PATH. "
                "Two CUDA installs on PATH is the usual cause.")
            if fail_if_no_gpu:
                raise RuntimeError(
                    "faster-whisper could not load on CUDA. "
                    "Fix .env and re-run; will not fall back to CPU."
                ) from exc
            log("WARNING: Falling back to CPU (FailIfNoGpu is false).")

    _model = WhisperModel(name, device="cpu",
                          compute_type=cfg.get("CpuComputeType", "int8"))
    _device_used = "cpu"
    log(f"Whisper model '{name}' loaded on CPU.")
    return _model


def transcribe_one(model, audio_path: Path, settings: dict,
                   label: str = "") -> dict:
    cfg = settings["Whisper"]
    segments, info = model.transcribe(
        str(audio_path),
        language=cfg.get("Language", "en"),
        beam_size=int(cfg.get("BeamSize", 5)),
        vad_filter=bool(cfg.get("VadFilter", True)),
    )

    rows = []
    pieces = []
    last_beat = time.time()
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        rows.append({
            "start": round(float(segment.start), 2),
            "end": round(float(segment.end), 2),
            "text": text,
        })
        pieces.append(text)
        now = time.time()
        if now - last_beat >= 30:
            log(f"    ... still transcribing {label or audio_path.name}  "
                f"{segment.end:.0f}s in")
            last_beat = now

    return {
        "text": " ".join(pieces).strip(),
        "segments": rows,
        "duration": round(float(getattr(info, "duration", 0.0)), 2),
        "language": getattr(info, "language", cfg.get("Language", "en")),
        "device": _device_used,
        "model": cfg["Model"],
    }


def main() -> int:
    parser = base_parser("Transcribe sermon audio")
    args = parser.parse_args()

    apply_cuda_env()

    with WorkerContext(args) as ctx:
        keep_audio = bool(ctx.settings["Audio"].get("KeepAfterTranscribe", False))

        candidates = [
            job for job in ctx.jobs
            if job.audio_path
            and (ctx.force or not job.at_least("transcribed"))
        ]

        # Files on disk whose job never got audio_path written.
        if ctx.paths.audio.is_dir():
            known = {job.id for job in candidates}
            for mp3 in ctx.paths.audio.glob("*.mp3"):
                job = ctx.jobs.get(mp3.stem)
                if job is None or job.id in known:
                    continue
                if job.at_least("transcribed") and not ctx.force:
                    continue
                job.audio_path = ctx.relative(mp3)
                if not job.at_least("audio_downloaded"):
                    job.advance("audio_downloaded")
                    ctx.jobs.put(job)
                candidates.append(job)
                known.add(job.id)

        if not candidates:
            log("Nothing to transcribe.")
            return 0

        log(f"NOW: transcribe {len(candidates)} file(s). "
            "Crawl and new downloads wait until this queue is empty.")
        log("Transcripts are kept. Audio is deleted after each success."
            if not keep_audio else
            "Transcripts and audio are both kept.")

        try:
            model = load_model(ctx.settings)
        except Exception as exc:
            log(f"Transcription cannot start: {exc}")
            return 1

        processed = 0
        total = len(candidates)

        for job in candidates:
            if ctx.should_stop(processed):
                log(f"Stopping with {total - processed} file(s) still queued.")
                break

            audio_path = ctx.root / job.audio_path
            if not audio_path.exists():
                job.audio_path = ""
                job.step = "page_saved"
                job.error = "audio missing at transcribe time; will re-download"
                ctx.jobs.put(job)
                log(f"  [{processed + 1}/{total}] missing audio  {job.id}")
                continue

            label = f"{job.date or '????-??-??'}  {job.title or job.id}"
            log(f"  [{processed + 1}/{total}] START  {label}")
            target = ctx.paths.transcripts / f"{job.id}.json"
            started = time.time()

            try:
                payload = transcribe_one(model, audio_path, ctx.settings,
                                        label=job.id)
            except Exception as exc:
                job.status = STATUS_FAILED
                job.error = f"transcribe: {type(exc).__name__}: {str(exc)[:200]}"
                ctx.jobs.put(job)
                log(f"  [{processed + 1}/{total}] FAIL   {job.id}: {exc}")
                processed += 1
                continue

            payload.update({
                "id": job.id,
                "title": job.title,
                "date": job.date,
                "speaker": job.speaker,
                "series": job.series,
                "campus": job.campus,
                "url": job.url,
                "audio_url": job.audio_url,
            })

            # Artifact first, then state.
            atomic_write_json(target, payload)

            job.transcript_path = ctx.relative(target)
            job.status = STATUS_PENDING
            job.error = ""
            job.advance("transcribed")
            ctx.jobs.put(job)

            elapsed = time.time() - started
            minutes = payload.get("duration", 0) / 60.0
            ratio = (payload.get("duration", 0) / elapsed) if elapsed > 0 else 0
            log(f"  [{processed + 1}/{total}] DONE   {label}  "
                f"{minutes:.0f} min audio in {elapsed / 60:.1f} min "
                f"({ratio:.0f}x realtime, {_device_used})")
            log(f"           transcript kept: {job.transcript_path}")

            if not keep_audio:
                audio_path.unlink(missing_ok=True)
                job.audio_path = ""
                ctx.jobs.put(job)
                log(f"           audio deleted: {audio_path.name}")

            processed += 1

        remaining = max(0, total - processed)
        log(f"Transcription pass complete: {processed} file(s) on "
            f"{_device_used or 'n/a'}. {remaining} still queued.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
