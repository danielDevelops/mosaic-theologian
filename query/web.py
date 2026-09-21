"""Small local web chat, bound to loopback only.

Serves one page and one streaming endpoint. The host defaults to 127.0.0.1
and is validated on startup, because this index is personal and should not be
reachable from the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI                                      # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse    # noqa: E402
from pydantic import BaseModel                                   # noqa: E402

from lib.config import Paths, apply_offline_env, load_settings   # noqa: E402
from lib.llm import ChatUnavailable, LocalChat                   # noqa: E402
from lib.prompt import build_messages, format_citations          # noqa: E402
from lib.retrieval import Retriever                              # noqa: E402

apply_offline_env()

app = FastAPI(title="Mosaic study assistant", docs_url=None, redoc_url=None)

_settings = load_settings()
_paths = Paths()
_retriever: Retriever | None = None
_chat = LocalChat(_settings)


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(_settings, _paths.index)
    return _retriever


class AskRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mosaic study assistant</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 16px/1.6 -apple-system, Segoe UI, system-ui, sans-serif;
         max-width: 820px; margin: 0 auto; padding: 24px 20px 80px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #888; font-size: 13px; margin-bottom: 24px; }
  #log { display: flex; flex-direction: column; gap: 20px; }
  .turn { border-left: 3px solid #d0d0d0; padding-left: 14px; }
  .turn.you { border-color: #7aa2c8; }
  .who { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
         color: #888; margin-bottom: 6px; }
  .body { white-space: pre-wrap; }
  .cites { margin-top: 12px; font-size: 13px; color: #777; }
  .cites div { padding: 2px 0; }
  form { position: fixed; bottom: 0; left: 0; right: 0; padding: 12px;
         background: Canvas; border-top: 1px solid #ccc; }
  .row { display: flex; gap: 8px; max-width: 820px; margin: 0 auto; }
  input { flex: 1; padding: 11px 13px; font-size: 15px;
          border: 1px solid #bbb; border-radius: 7px; background: Canvas;
          color: CanvasText; }
  button { padding: 11px 18px; font-size: 15px; border-radius: 7px;
           border: 1px solid #888; cursor: pointer; background: Canvas;
           color: CanvasText; }
</style>
</head>
<body>
  <h1>Mosaic study assistant</h1>
  <div class="sub">Local and offline. Scripture first, then this church's
    teaching, then wider Christian thought.</div>
  <div id="log"></div>

  <form id="form">
    <div class="row">
      <input id="q" autocomplete="off" placeholder="Ask anything..." autofocus>
      <button type="submit">Ask</button>
    </div>
  </form>

<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const box = document.getElementById('q');
let history = [];

function turn(who, cls) {
  const wrap = document.createElement('div');
  wrap.className = 'turn ' + (cls || '');
  const label = document.createElement('div');
  label.className = 'who';
  label.textContent = who;
  const body = document.createElement('div');
  body.className = 'body';
  wrap.append(label, body);
  log.append(wrap);
  window.scrollTo(0, document.body.scrollHeight);
  return { wrap, body };
}

form.onsubmit = async (event) => {
  event.preventDefault();
  const question = box.value.trim();
  if (!question) return;
  box.value = '';
  turn('You', 'you').body.textContent = question;

  const answer = turn('Assistant');
  let text = '';

  const response = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of decoder.decode(value).split('\\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = JSON.parse(line.slice(5));
      if (payload.token) {
        text += payload.token;
        answer.body.textContent = text;
        window.scrollTo(0, document.body.scrollHeight);
      }
      if (payload.citations) {
        const cites = document.createElement('div');
        cites.className = 'cites';
        cites.innerHTML = '<strong>Sources</strong>';
        for (const citation of payload.citations) {
          const row = document.createElement('div');
          row.textContent = '- ' + citation;
          cites.append(row);
        }
        answer.wrap.append(cites);
      }
      if (payload.error) {
        answer.body.textContent = text + '\\n\\n[' + payload.error + ']';
      }
    }
  }

  history.push({ role: 'user', content: question });
  history.push({ role: 'assistant', content: text });
  history = history.slice(-6);
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/health")
def health() -> dict:
    try:
        counts = retriever().store.counts()
        identity = retriever().identity
        return {
            "ok": True,
            "counts": counts,
            "embedding": identity.to_dict(),
            "chat_server": _chat.available(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/ask")
def ask(request: AskRequest) -> StreamingResponse:
    def emit(payload: dict) -> str:
        return "data:" + json.dumps(payload) + "\n"

    def generate():
        try:
            result = retriever().search(request.question)
            messages = build_messages(result, request.history)
            for piece in _chat.stream(messages):
                yield emit({"token": piece})
            yield emit({"citations": format_citations(result)})
        except ChatUnavailable as exc:
            yield emit({"error": str(exc)})
        except Exception as exc:  # surfaced in the UI rather than a blank reply
            yield emit({"error": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(generate(), media_type="text/event-stream")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local web chat")
    parser.add_argument("--host", default=_settings["Web"]["Host"])
    parser.add_argument("--port", type=int, default=_settings["Web"]["Port"])
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # This index is personal. Refuse to expose it on the network by accident.
        print(f"Refusing to bind {args.host}. This server is loopback-only; "
              f"use 127.0.0.1.")
        return 2

    import uvicorn

    print(f"Mosaic study assistant on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
