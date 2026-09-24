#!/usr/bin/env bash
#
# Read the Windows build machine over the mounted share and report the
# night job: whether a run is active, the latest log lines, and counts of
# transcripts, audio, queued message pages, and the vector index.
#
#   ./remote-status.sh
#   MOSAIC_SHARE="/Volumes/Mosiac LLM" ./remote-status.sh
#
# The share name is "Mosiac LLM" (the folder on w5ffl3pc). If it is not
# mounted, this opens the smb URL and waits for Finder to attach it.

set -euo pipefail

SHARE="${MOSAIC_SHARE:-/Volumes/Mosiac LLM}"
SMB_URL="${MOSAIC_SMB:-smb://w5ffl3pc/Mosiac%20LLM}"
REPO="$SHARE/mosaic-theologian"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR %s\n' "$*" >&2; exit 1; }

if [ ! -d "$REPO" ]; then
  say "Share not mounted. Opening $SMB_URL"
  open "$SMB_URL" || die "Could not open $SMB_URL"
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    [ -d "$REPO" ] && break
    sleep 1
  done
fi

[ -d "$REPO" ] || die "Repo not found at $REPO"

say "Mosaic remote status"
say "Repo: $REPO"
say ""

/usr/bin/python3 - "$REPO" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

repo = Path(sys.argv[1])


def is_message_url(url: str) -> bool:
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if not parts or parts[0] != "messages" or len(parts) < 3:
        return False
    return parts[1] not in {"archive", "series"}


def is_listing_url(url: str) -> bool:
    path = urlparse(url).path.rstrip("/")
    if path in ("", "/"):
        return True
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts or parts[0] != "messages":
        return False
    return not is_message_url(url)


def replay_jsonl(path: Path, key: str) -> dict:
    latest = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ident = row.get(key)
            if ident:
                latest[ident] = row
    return latest


def count_files(folder: Path, suffix: str) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix == suffix)


def age(ts: float) -> str:
    seconds = max(0, int(datetime.now().timestamp() - ts))
    if seconds < 90:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m ago"
    return f"{minutes // 60}h {minutes % 60}m ago"


lock_path = repo / "state" / "night.lock"
if lock_path.is_file():
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        print(f"Run: in progress  PID {lock.get('pid', '?')}  started {lock.get('started', '?')}")
    except json.JSONDecodeError:
        print("Run: lock file is present but unreadable")
else:
    print("Run: not in progress")

log_path = repo / "state" / "run.log"
cycle = ""
error = ""
recent = []
if log_path.is_file():
    lines = [ln.rstrip() for ln in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    for line in lines:
        if "--- cycle " in line:
            cycle = line
            error = ""
        elif "[ERROR]" in line:
            error = line
    recent = lines[-8:]
    print(f"Log updated: {datetime.fromtimestamp(log_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}  ({age(log_path.stat().st_mtime)})")
else:
    print("Log: missing")

if cycle:
    print(f"Latest cycle: {cycle.split('] ', 1)[-1]}")
if error:
    print(f"Latest error: {error.split('] ', 1)[-1]}")

manifest_path = repo / "index" / "index-manifest.json"
if manifest_path.is_file():
    rag_ts = manifest_path.stat().st_mtime
    rag_when = datetime.fromtimestamp(rag_ts).strftime("%Y-%m-%d %H:%M:%S")
    print(f"RAG last updated: {rag_when}  ({age(rag_ts)})")
else:
    print("RAG last updated: never")

print()
print("On disk")
transcripts = repo / "data" / "transcripts"
audio = repo / "data" / "audio"
pages = repo / "data" / "pages"
print(f"  transcripts     {count_files(transcripts, '.json'):6d}")
print(f"  audio           {count_files(audio, '.mp3'):6d}")
print(f"  pages           {count_files(pages, '.json'):6d}")
if transcripts.is_dir():
    files = [p for p in transcripts.iterdir() if p.is_file() and p.suffix == ".json"]
    if files:
        newest = max(files, key=lambda p: p.stat().st_mtime)
        stamp = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%H:%M:%S")
        print(f"  newest          {stamp}  {age(newest.stat().st_mtime)}  {newest.name}")

jobs = replay_jsonl(repo / "state" / "jobs.jsonl", "id")
print()
print(f"Jobs  {len(jobs)}")
steps = Counter(j.get("step") or "?" for j in jobs.values())
for step, count in steps.most_common():
    print(f"  {step:<18} {count:6d}")
failed = sum(1 for j in jobs.values() if j.get("status") == "failed")
if failed:
    print(f"  status failed       {failed:6d}")

catalog = replay_jsonl(repo / "state" / "catalog.jsonl", "url")
queued_messages = fetched_messages = queued_listings = 0
for url, row in catalog.items():
    state = row.get("state", "")
    if is_message_url(url):
        if state == "queued":
            queued_messages += 1
        else:
            fetched_messages += 1
    elif is_listing_url(url) and state == "queued":
        queued_listings += 1
print()
print("Catalog")
print(f"  message pages   {queued_messages + fetched_messages:6d}")
print(f"  fetched         {fetched_messages:6d}")
print(f"  still queued    {queued_messages:6d}")
print(f"  listings queued {queued_listings:6d}")

print()
print("Index")
if manifest_path.is_file():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        manifest = {}
    counts = manifest.get("counts") or {}
    if counts:
        for name, count in counts.items():
            print(f"  {name:<16} {count:6d}")
    else:
        print("  manifest has no counts")
else:
    print("  no manifest")

if recent:
    print()
    print("Recent log")
    for line in recent:
        print(f"  {line}")
PY
