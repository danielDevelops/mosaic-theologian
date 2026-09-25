# Intel MacBook Pro query machine

This machine only asks questions. It never crawls the site, downloads sermons,
transcribes audio, or contacts the Windows machine. Everything it needs arrived
in the exported bundle, and it runs fully offline.

## Setup

Copy the whole `mosaic-portable` folder over, then:

```bash
cd mosaic-portable
chmod +x mosaic.sh
./mosaic.sh install
./mosaic.sh verify
./mosaic.sh start
./mosaic.sh ask
```

`install` needs Homebrew. It adds Python 3.12 and `llama.cpp` if they are
missing, creates a `.venv` with that Python, and records what it installed
in `state/deps.json`. It is safe to re-run. An existing `.venv` built with
another Python is replaced: Intel PyTorch wheels stop at 3.12.

If the shell refuses to run the script with a `bad interpreter` error, the copy
converted the line endings. Fix with:

```bash
sed -i '' 's/\r$//' mosaic.sh
```

## Always verify before trusting a copy

```bash
./mosaic.sh verify
```

This replays every checksum in `manifest.json`, confirms the index row counts
match what the Windows machine exported, and loads the embedding model to
confirm it is the same one that built the index.

It is worth the minute it takes. Both failures it catches are silent:

- A truncated copy leaves an index that looks healthy and simply returns
  nothing for whatever was lost. Nothing errors.
- An embedding model that differs from the one that built the index produces
  vectors in a different space. Retrieval then returns irrelevant passages
  with no error at all, and the answers just quietly get worse.

`verify` exits non-zero on either, so it refuses to let you serve from a bad
bundle rather than letting you find out weeks later.

Exit codes: `0` verified, `1` bundle problem, `2` embedding mismatch.

## Asking questions

```bash
./mosaic.sh ask     # terminal
./mosaic.sh web     # browser at http://127.0.0.1:8787
```

Type anything. Nothing is pre-seeded: your question is embedded at the moment
you hit enter and matched against stored source text, so there is no list of
supported questions to maintain.

Answers follow a source hierarchy - Scripture first, then the church's belief
statements, then its sermons and articles, woven into one explanation. General
theological knowledge is marked in the sentence when it is not in the supplied
sources. If the church has not addressed something, the answer says so rather
than inventing a position for it. Citations are printed after each answer.

In the CLI, `:sources` reprints the citations, `:clear` drops the conversation
history, and `:quit` exits.

## Speed, honestly

This is CPU inference. Metal acceleration is Apple-Silicon-only, so an Intel
Mac has no GPU path at all. Expect roughly 2-4 tokens per second on an 8B
model, and expect a pause before the first token while the retrieved passages
are processed. A full cited answer can take a couple of minutes.

The levers, in the order worth trying:

1. Use a **Q4_K_M** build rather than Q5.
2. Drop to a 3B-4B model for quick lookups and keep the larger one for
   questions worth waiting on. Point `Chat.ModelFile` at whichever you want.
3. Lower `Retrieval.TopK` in `config/settings.json`. Prompt length drives most
   of the latency, so fewer passages helps more than anything else.
4. Lower `Chat.MaxTokens` if you want shorter answers sooner.

Editing `config/settings.local.json` keeps these changes local to this machine
so a re-sync does not overwrite them.

## Getting new sermons

New messages are always ingested on the Windows machine. To bring this one up
to date, re-export there and recopy:

```powershell
.\Mosaic-NightJob.ps1 -Action Export -Destination E:\mosaic-portable
```

Then here:

```bash
./mosaic.sh verify
./mosaic.sh stop && ./mosaic.sh start
```

Re-syncs are small. A new sermon adds rows to the index but does not touch
either model, and the models are nearly all of the bundle's size, so an
index-only export is usually tens to a few hundred MB rather than several GB.
The fresh `manifest.json` is how `verify` confirms you actually replaced the
old bundle.

## Privacy

`llama-server` binds to `127.0.0.1` only, and the web UI refuses to start on
any non-loopback address. Nothing is sent anywhere. The Python runtime sets
`HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` so the embedding library never
reaches for the network either, which also stops it stalling when there is no
connection.

## Commands

| Command | What it does |
|---|---|
| `./mosaic.sh install` | One-time setup; re-runnable |
| `./mosaic.sh verify` | Check the copied bundle; do this after every copy |
| `./mosaic.sh start` | Start the local model server |
| `./mosaic.sh ask` | Chat in the terminal |
| `./mosaic.sh web` | Chat in the browser |
| `./mosaic.sh stop` | Stop the model server |
| `./mosaic.sh status` | What is running and what is indexed |

## If something is wrong

`./mosaic.sh status` shows whether the server is up and what the bundle
contains. Server startup problems are logged to `state/llama-server.log`; a
large model on this hardware can legitimately take a while to load.

If `verify` reports an embedding mismatch, do not use `--no-verify` to get
past it. Re-export from Windows with `-Full` so the bundled model and the
index match again.

If a question fails with `Index read failed`, the index was written by a
different LanceDB than the one pinned in `requirements-mac.txt`. On Windows,
install the pinned requirements, run `-Action Reindex`, then export with
`-Full` and copy the bundle again. Do not keep asking: a failed read is an
error, and it is not a finding that the church has no position.
