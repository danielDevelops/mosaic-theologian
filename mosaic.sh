#!/usr/bin/env bash
#
# Mosaic study assistant - Intel MacBook Pro query runtime.
#
# This machine only asks questions. It never crawls, downloads sermons,
# transcribes, or contacts the Windows build machine. Everything it needs
# arrived in the exported bundle.
#
#   ./mosaic.sh install    one time setup
#   ./mosaic.sh verify     check the copied bundle before trusting it
#   ./mosaic.sh start      run the local model server on 127.0.0.1
#   ./mosaic.sh ask        chat in the terminal
#   ./mosaic.sh web        chat in the browser
#   ./mosaic.sh stop       stop the model server
#   ./mosaic.sh status     what is running and what is indexed

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="$VENV/bin/python"
STATE="$HERE/state"
PIDFILE="$STATE/llama-server.pid"
LOGFILE="$STATE/llama-server.log"
DEPS="$STATE/deps.json"

# The bundle is meant to run without network access. Without these the
# embedding library tries to reach the model hub on load and stalls.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$STATE"

say()  { printf '%s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*" >&2; }
die()  { printf 'ERROR %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Intel PyTorch wheels stop at Python 3.12. Homebrew's python@3.12 is
# keg-only, so it is not the python3 already on PATH.
resolve_python312() {
  if have python3.12; then
    command -v python3.12
    return 0
  fi
  local prefix
  prefix="$(brew --prefix python@3.12 2>/dev/null || true)"
  if [ -n "$prefix" ] && [ -x "$prefix/bin/python3.12" ]; then
    printf '%s' "$prefix/bin/python3.12"
    return 0
  fi
  return 1
}

venv_is_python312() {
  [ -x "$PY" ] || return 1
  local ver
  ver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  [ "$ver" = "3.12" ]
}

# Append a dependency record. Mirrors the Windows deps.json so both machines
# leave the same kind of audit trail. Nothing is ever uninstalled.
record_dep() {
  local name="$1" status="$2" detail="${3:-}"
  local stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local tmp="$DEPS.part"

  if [ -f "$DEPS" ]; then
    /usr/bin/python3 - "$DEPS" "$name" "$status" "$detail" "$stamp" "$tmp" <<'PYEOF'
import json, sys
path, name, status, detail, stamp, tmp = sys.argv[1:7]
try:
    with open(path) as fh:
        data = json.load(fh)
except Exception:
    data = {"items": []}
items = [i for i in data.get("items", []) if i.get("name") != name]
items.append({"name": name, "status": status, "detail": detail, "recorded": stamp})
with open(tmp, "w") as fh:
    json.dump({"items": items}, fh, indent=2)
PYEOF
  else
    printf '{\n  "items": [\n    {"name": "%s", "status": "%s", "detail": "%s", "recorded": "%s"}\n  ]\n}\n' \
      "$name" "$status" "$detail" "$stamp" > "$tmp"
  fi
  mv "$tmp" "$DEPS"
}

read_setting() {
  # read_setting <dotted.path> <default>
  local path="$1" fallback="${2:-}"
  "$PY" - "$HERE" "$path" "$fallback" <<'PYEOF' 2>/dev/null || printf '%s' "$fallback"
import json, sys, os
root, dotted, fallback = sys.argv[1:4]
value = None
try:
    with open(os.path.join(root, "config", "settings.json")) as fh:
        value = json.load(fh)
    local = os.path.join(root, "config", "settings.local.json")
    if os.path.exists(local):
        with open(local) as fh:
            override = json.load(fh)
    else:
        override = {}
    for key in dotted.split("."):
        nxt = override.get(key) if isinstance(override, dict) else None
        value = value[key]
        override = nxt if isinstance(nxt, dict) else {}
        if nxt is not None and not isinstance(nxt, dict):
            value = nxt
    sys.stdout.write(str(value))
except Exception:
    sys.stdout.write(fallback)
PYEOF
}

# ------------------------------------------------------------------ install --

cmd_install() {
  say "== Install =="

  if [ "$(uname -s)" != "Darwin" ]; then
    warn "This script targets macOS. On Windows use Mosaic-NightJob.ps1."
  fi

  if [ "$(uname -m)" = "x86_64" ]; then
    say "Intel Mac detected: CPU inference only (Metal is Apple Silicon only)."
  fi

  if have brew; then
    say "Homebrew present."
    record_dep homebrew present "$(brew --version | head -n1)"
  else
    die "Homebrew is required. Install it from https://brew.sh then re-run."
  fi

  local py312
  if py312="$(resolve_python312)"; then
    say "Python 3.12 present: $("$py312" --version)"
    record_dep python3.12 present "$("$py312" --version)"
  else
    say "Installing python@3.12"
    NONINTERACTIVE=1 brew install python@3.12
    py312="$(resolve_python312)" || die "python@3.12 installed but the interpreter was not found."
    record_dep python3.12 installed "brew python@3.12"
  fi

  if have llama-server; then
    say "llama.cpp present."
    record_dep llama.cpp present "$(command -v llama-server)"
  else
    say "Installing llama.cpp"
    NONINTERACTIVE=1 brew install llama.cpp
    record_dep llama.cpp installed "brew llama.cpp"
  fi

  if venv_is_python312; then
    say "Virtual environment present (Python 3.12)."
  else
    if [ -d "$VENV" ]; then
      say "Replacing virtual environment (need Python 3.12)"
      rm -rf "$VENV"
    else
      say "Creating virtual environment"
    fi
    "$py312" -m venv "$VENV"
    record_dep venv installed ".venv python3.12"
  fi

  say "Installing Python requirements (CPU wheels)"
  "$PY" -m pip install --upgrade pip --quiet
  "$PY" -m pip install -r "$HERE/requirements-mac.txt" --quiet
  record_dep python-requirements installed requirements-mac.txt

  if [ -d "$HERE/models/embedding" ]; then
    say "Embedding model found in the bundle."
    record_dep embedding-model present "bundled"
  else
    warn "models/embedding is missing from this bundle."
    warn "Re-export from Windows with -Full. Fetching a different revision"
    warn "here would silently mismatch the index."
    record_dep embedding-model missing "not bundled"
  fi

  local gguf
  gguf="$(find "$HERE/models" -maxdepth 1 -name '*.gguf' 2>/dev/null | head -n1 || true)"
  if [ -n "$gguf" ]; then
    say "Chat model found: $(basename "$gguf")"
    record_dep chat-model present "$(basename "$gguf")"
  else
    warn "No .gguf chat model in models/. Copy one over before ./mosaic.sh start."
    record_dep chat-model missing "no gguf"
  fi

  say ""
  say "Next: ./mosaic.sh verify"
}

# ------------------------------------------------------------------- verify --

cmd_verify() {
  [ -x "$PY" ] || die "Run ./mosaic.sh install first."
  say "== Verify bundle =="
  "$PY" -m query.verify_bundle "$@"
}

# -------------------------------------------------------------------- start --

find_gguf() {
  local configured
  configured="$(read_setting Chat.ModelFile "models/chat.gguf")"
  if [ -f "$HERE/$configured" ]; then
    printf '%s' "$HERE/$configured"
    return 0
  fi
  find "$HERE/models" -maxdepth 1 -name '*.gguf' 2>/dev/null | head -n1
}

server_running() {
  [ -f "$PIDFILE" ] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || echo '')"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

cmd_start() {
  [ -x "$PY" ] || die "Run ./mosaic.sh install first."

  if server_running; then
    say "Model server already running (PID $(cat "$PIDFILE"))."
    return 0
  fi

  have llama-server || die "llama-server not found. Run ./mosaic.sh install."

  local model
  model="$(find_gguf)"
  [ -n "$model" ] || die "No .gguf model found in models/."

  local ctx threads
  ctx="$(read_setting Chat.ContextTokens 8192)"
  threads="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)"

  say "Starting llama-server"
  say "  model:   $(basename "$model")"
  say "  context: $ctx tokens"
  say "  threads: $threads"
  say "  bound:   127.0.0.1:8080  (loopback only)"

  # Loopback only, never 0.0.0.0. This index is personal.
  nohup llama-server \
    --model "$model" \
    --host 127.0.0.1 \
    --port 8080 \
    --ctx-size "$ctx" \
    --threads "$threads" \
    >"$LOGFILE" 2>&1 &

  echo $! > "$PIDFILE"

  local waited=0
  printf 'Waiting for the server'
  while [ "$waited" -lt 90 ]; do
    if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
      printf '\nReady.\n'
      say ""
      say "Ask a question:  ./mosaic.sh ask"
      say "Or open the UI:  ./mosaic.sh web"
      return 0
    fi
    if ! server_running; then
      printf '\n'
      warn "Server exited. Last lines of $LOGFILE:"
      tail -n 20 "$LOGFILE" >&2 || true
      return 1
    fi
    printf '.'
    sleep 2
    waited=$((waited + 2))
  done

  printf '\n'
  warn "Server did not become healthy within 90s. Check $LOGFILE."
  warn "On an Intel Mac a large model can take a while to load."
  return 1
}

cmd_stop() {
  if ! server_running; then
    say "Model server is not running."
    rm -f "$PIDFILE"
    return 0
  fi
  local pid
  pid="$(cat "$PIDFILE")"
  say "Stopping model server (PID $pid)"
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  say "Stopped."
}

# ---------------------------------------------------------------------- ask --

cmd_ask() {
  [ -x "$PY" ] || die "Run ./mosaic.sh install first."
  server_running || {
    warn "Model server is not running. Starting it now."
    cmd_start || die "Could not start the model server."
  }
  "$PY" -m query.cli --device cpu "$@"
}

cmd_web() {
  [ -x "$PY" ] || die "Run ./mosaic.sh install first."
  server_running || {
    warn "Model server is not running. Starting it now."
    cmd_start || die "Could not start the model server."
  }
  local port
  port="$(read_setting Web.Port 8787)"
  say "Opening http://127.0.0.1:$port"
  ( sleep 2; open "http://127.0.0.1:$port" >/dev/null 2>&1 || true ) &
  "$PY" -m query.web --host 127.0.0.1 --port "$port"
}

cmd_status() {
  say "== Status =="

  if server_running; then
    say "Model server: running (PID $(cat "$PIDFILE"))"
  else
    say "Model server: stopped"
  fi

  if [ -f "$HERE/manifest.json" ] && [ -x "$PY" ]; then
    "$PY" - "$HERE" <<'PYEOF'
import json, sys, os
root = sys.argv[1]
with open(os.path.join(root, "manifest.json")) as fh:
    manifest = json.load(fh)
counts = manifest.get("counts", {})
embedding = manifest.get("embedding", {})
print("Bundle:      %s (%s)" % (manifest.get("created"), manifest.get("mode")))
print("Index rows:  " + ", ".join("%s=%s" % (k, format(v, ",")) for k, v in counts.items()))
print("Embedding:   %s@%s dim=%s" % (
    embedding.get("model"), embedding.get("revision"), embedding.get("dimension")))
PYEOF
  else
    say "No bundle manifest found in this folder."
  fi
}

usage() {
  sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
  install) shift; cmd_install "$@" ;;
  verify)  shift; cmd_verify  "$@" ;;
  start)   shift; cmd_start   "$@" ;;
  stop)    shift; cmd_stop    "$@" ;;
  ask)     shift; cmd_ask     "$@" ;;
  web)     shift; cmd_web     "$@" ;;
  status)  shift; cmd_status  "$@" ;;
  ""|-h|--help|help) usage ;;
  *) die "Unknown command: $1  (try: install verify start ask web stop status)" ;;
esac
