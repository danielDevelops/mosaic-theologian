# Mosaic theologian

A personal, offline study assistant over Scripture, one church's belief
statements, and twenty years of its published sermons.

## What this is

A general local chat model plus a local search index. It is **not** a custom
trained or fine-tuned model, and nothing is pre-seeded: you type any question,
that question is embedded at ask time, the closest source passages are
retrieved, and the model answers from them with citations.

Two pieces work together:

1. **A general chat model** (Llama 3.1 8B or similar, trained by someone else).
   It supplies reasoning and background knowledge, so it can explain a debate
   like Arminianism versus Calvinism even if nobody at the church ever used
   those words.
2. **A local index** of Scripture, belief statements, and sermon transcripts.
   It supplies what was actually said, so claims are grounded and checkable.

Answers follow a source hierarchy: Scripture first, then the church's belief
statements, then its sermons and articles, woven into one explanation.
General knowledge is marked in the sentence when it is not in the supplied
sources. When the church has not addressed a question, the answer says so
instead of inventing a position. Citations are printed after the answer.

## Two machines

```
Windows 11 + RTX 4070 Ti Super          Intel MacBook Pro
  crawl, transcribe, index, export        verify, then ask questions
  Mosaic-NightJob.ps1                     mosaic.sh
                     |                          ^
                     +------ copy bundle -------+
```

The Mac is standalone. After the copy it never contacts the Windows machine,
and it runs with no network access at all.

- [docs/README-Windows.md](docs/README-Windows.md) - building and updating the library
- [docs/README-Mac.md](docs/README-Mac.md) - installing, verifying, and asking

## Quick start

On Windows:

```powershell
.\Mosaic-NightJob.ps1 -Action EnsureDeps
.\Mosaic-NightJob.ps1 -Action Run -MaxPages 15 -Until 06:00
.\Mosaic-NightJob.ps1 -Action Export -Destination E:\mosaic-portable -Full
```

On the Mac:

```bash
./mosaic.sh install && ./mosaic.sh verify && ./mosaic.sh start && ./mosaic.sh ask
```

You do not need the full backfill before asking anything. Beliefs, Scripture,
and a handful of recent messages are enough to be useful on night one; the
remaining twenty years fills in over roughly four to eleven nights.

## Design decisions worth knowing

**Stopping is always safe.** Resume safety is not built on catching Ctrl+C,
because Windows PowerShell frequently cannot catch it and takes the child
Python process down too. Instead state advances only after an artifact is on
disk, artifacts are renamed into place atomically, and startup re-queues
anything whose recorded step has no artifact behind it. A kill costs the one
item in flight.

**Sermons are deduplicated by content, not URL.** The same message is
published under more than one series path, so items are keyed on date, campus,
and audio URL. Keying on page URL would index some sermons twice and let them
dominate retrieval.

**The Bible source is validated by book count.** A "does Genesis exist" check
accepts a file that stops partway through, which would leave an index that
looks healthy and answers nothing for everything after the cut. Anything under
66 books is rejected.

**The embedding model is pinned to the index.** A different model, or even a
different revision, produces vectors in a different space. Nothing errors;
retrieval just quietly returns irrelevant passages. So the model identity is
recorded in the index manifest, the model files travel inside the bundle, and
both machines refuse to serve on a mismatch.

**Everything stays local.** The model server binds to loopback only, the web
UI refuses any non-loopback address, and the runtime disables the model hub so
nothing reaches for the network.

## Scripture text

The Bible JSON is downloaded at setup time from the URL in
`config/settings.json` and stored in `data/bible/`, which is gitignored. Verse
text is never committed to this repository.

Keep this personal and offline. If it ever becomes a public server or a
published app, switch the scripture collection to a public-domain translation
or a licensed API, since translations are generally copyrighted.

## Layout

```
Mosaic-NightJob.ps1    Windows orchestrator
mosaic.sh              Mac runtime
config/settings.json   every tunable
lib/                   shared: chunking, embedding, retrieval, prompt, state
workers/               crawl, download, transcribe, ingest, index, export
query/                 CLI, web chat, bundle verifier
```

`lib/` and `query/` are copied into the bundle, so retrieval and prompting are
literally the same code on both machines and answers do not drift between them.

## If answers are not good enough

Tune retrieval before considering anything more exotic. In rough order of
value: add BM25 keyword scoring alongside the vectors, rerank the top hits
with a small cross-encoder, adjust chunking, and expand queries into several
retrieval phrasings. Fine-tuning is the wrong tool here: it moves style rather
than facts, it cannot be cited, and it would make the model fluent enough to
assert church positions that were never taught.
