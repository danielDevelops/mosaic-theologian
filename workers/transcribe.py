"""Transcribe downloaded sermon audio with faster-whisper on the GPU.

Reports whether CUDA was actually used. faster-whisper falls back to CPU
silently when cuBLAS or cuDNN are missing, which turns a week of nights into
a couple of months, so the fallback is logged loudly rather than absorbed.
"""

from __future__ import annotations

import time
from pathlib import Path

from lib.state import STATUS_FAILED, STATUS_PENDING, atomic_write_json
from workers.common import WorkerContext, base_parser, log

_model = None
_device_used = ""


def load_model(settings: dict):
    """Build the model once per process, downgrading to CPU only if forced."""
    global _model, _device_used
    if _model is not None:
        return _model

    from faster_whisper import WhisperModel

    cfg = settings["Whisper"]
    name = cfg["Model"]
    want = cfg.get("Device", "cuda")

    if want == "cuda":
        try:
            _model = WhisperModel(name, device="cuda",
                                  compute_type=cfg.get("ComputeType", "float16"))
            _device_used = "cuda"
            log(f"Whisper model '{name}' loaded on GPU.")
            return _model
        except Exception as exc:
            log(f"WARNING: GPU unavailable for faster-whisper ({type(exc).__name__}: "
                f"{str(exc)[:160]}).")
            log("WARNING: Falling back to CPU. This is roughly ten times slower.")
            log("WARNING: Usual cause is missing cuDNN/cuBLAS DLLs on PATH.")

    _model = WhisperModel(name, device="cpu",
                          compute_type=cfg.get("CpuComputeType", "int8"))
    _device_used = "cpu"
    log(f"Whisper model '{name}' loaded on CPU.")
    return _model


def transcribe_one(model, audio_path: Path, settings: dict) -> dict:
    cfg = settings["Whisper"]
    segments, info = model.transcribe(
        str(audio_path),
        language=cfg.get("Language", "en"),
        beam_size=int(cfg.get("BeamSize", 5)),
        vad_filter=bool(cfg.get("VadFilter", True)),
    )

    rows = []
    pieces = []
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

    with WorkerContext(args) as ctx:
        keep_audio = bool(ctx.settings["Audio"].get("KeepAfterTranscribe", False))

        candidates = [
            job for job in ctx.jobs
            if job.audio_path
            and (ctx.force or not job.at_least("transcribed"))
        ]

        if not candidates:
            log("Nothing to transcribe.")
            return 0

        log(f"{len(candidates)} file(s) queued for transcription.")
        model = load_model(ctx.settings)
        processed = 0

        for job in candidates:
            if ctx.should_stop(processed):
                break

            audio_path = ctx.root / job.audio_path
            if not audio_path.exists():
                job.audio_path = ""
                job.step = "page_saved"
                job.error = "audio missing at transcribe time; will re-download"
                ctx.jobs.put(job)
                continue

            target = ctx.paths.transcripts / f"{job.id}.json"
            started = time.time()

            try:
                payload = transcribe_one(model, audio_path, ctx.settings)
            except Exception as exc:
                job.status = STATUS_FAILED
                job.error = f"transcribe: {type(exc).__name__}: {str(exc)[:200]}"
                ctx.jobs.put(job)
                log(f"  fail {job.id}: {exc}")
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
            log(f"  ok   {job.id}  {minutes:.0f} min audio in {elapsed/60:.1f} min "
                f"({ratio:.0f}x realtime, {_device_used})")

            if not keep_audio:
                audio_path.unlink(missing_ok=True)
                job.audio_path = ""
                ctx.jobs.put(job)

            processed += 1

        log(f"Transcription pass complete: {processed} file(s) on {_device_used or 'n/a'}.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
