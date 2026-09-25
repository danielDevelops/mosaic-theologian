---
name: mosaic-theologian
description: >-
  Describes and rebuilds the Mosaic theologian offline study assistant: the
  Windows crawl/transcribe/index pipeline, the Intel Mac query runtime, the
  Scripture-beliefs-sermons source hierarchy, LanceDB retrieval, and the
  resume-safety and embedding-identity invariants. Use when explaining the
  project, writing a README or architecture summary, reconstructing or porting
  the system, or changing crawl, transcription, indexing, retrieval, prompting,
  export, or the Windows-to-Mac bundle.
---

# Mosaic theologian

Personal offline study assistant over Scripture, one church's belief
statements, and its published sermons. A general local chat model plus a
local search index. It is not a fine-tuned model, and nothing is pre-seeded.

Read [architecture.md](architecture.md) before rebuilding, porting, or writing
a description that names files, stages, or tunables. Quote current code for
numbers and prompt wording. This skill holds the contracts; the files hold
the current values.

## How to describe it

Write for someone who will run it, not for someone reviewing a diff.

1. Open with what it is: local chat plus a local index of Scripture, belief
   statements, and sermon transcripts. Say explicitly that the model is
   general (Llama 3.1 8B Instruct or the GGUF named in `config/settings.json`)
   and that questions are embedded at ask time.
2. State the source hierarchy in this order: Scripture, then this church's
   belief statements, then its sermons and articles, woven into one
   explanation. General knowledge is marked in the sentence when it is not in
   the supplied sources. When the church has not addressed the question, the
   answer says so. Citations follow the answer.
3. Draw the two machines. Windows 11 with the RTX builds the library
   (`Mosaic-NightJob.ps1`). The Intel Mac only asks (`mosaic.sh`). The handoff
   is a file copy of the bundle. After the copy the Mac never contacts
   Windows and runs with the model hub disabled.
4. Name the decisions that prevent silent failure: resume safety by write
   order, sermon identity by audio URL, Bible validation by 66 books,
   embedding identity pinned in the index manifest, loopback-only serving.
5. Point at `docs/README-Windows.md` and `docs/README-Mac.md` for procedures.
   Do not paste verse text. Do not claim the system was trained on the church.

Use this shape:

```markdown
# Mosaic theologian

[One paragraph: what it is, and what it is not.]

## What answers from what
[Source hierarchy, citations, and the "church has not addressed this" rule.]

## Two machines
[Build machine, query machine, bundle copy, offline after the copy.]

## Design decisions worth knowing
[The silent-failure invariants, each in one short paragraph.]

## Layout
[Orchestrators, config, lib, workers, query.]
```

After drafting, check the draft against `lib/prompt.py` (`SYSTEM_PROMPT`),
`config/settings.json`, and the priority comment in `Invoke-Run` inside
`Mosaic-NightJob.ps1`. If those disagree with the draft, the draft is wrong.

## How to rebuild it

Preserve behavior. Do not replace the design with a hosted vector database,
a fine-tune, or a crawler that keeps going after HTTP 429 or 403.

1. **Settings and paths.** One `config/settings.json`. Optional
   `config/settings.local.json` deep-merges over it. Every path resolves from
   the directory that contains `config/settings.json`. `MOSAIC_ROOT` overrides
   that for tests.
2. **Shared library (`lib/`).** This code is copied into the Mac bundle.
   Retrieval and prompting must stay here so both machines run the same
   logic. Build in this order: `bookmap`, `config`, `state`, `pagetext`,
   `chunking`, `embedding`, `store`, `retrieval`, `prompt`, then `llm` and
   `chat_model`.
3. **Workers.** Each worker is a `python -m workers.<name>` process. The
   PowerShell orchestrator sequences them. Do not fold the night loop into
   Python; the kill behavior that matters is PowerShell taking the child
   down with it.
4. **Query runtime.** `query/cli.py`, `query/web.py`, and
   `query/verify_bundle.py`, driven by `mosaic.sh`. CPU wheels only. No
   crawler and no Whisper on the Mac.
5. **Prove the contracts** in [architecture.md](architecture.md) with the
   existing tests before calling a port done:
   `python tests/test_pipeline.py`, `python tests/test_cross_links.py`,
   `python tests/test_chat_model.py`.

## Invariants

Break one of these and the system looks healthy while answering wrong, or it
cannot resume after a kill.

- **Write order is the resume mechanism.** Advance a job step only after its
  artifact is on disk. Write artifacts to `*.part` and rename into place.
  On startup, delete stray `*.part` files and re-queue any job whose recorded
  step has no artifact. A kill costs the in-flight item. Do not depend on
  catching Ctrl+C.
- **Jobs are append-only JSONL.** `state/jobs.jsonl`: last record for an id
  wins. Steps only move forward: `discovered`, `page_saved`,
  `audio_downloaded`, `transcribed`, `chunked`, `indexed`.
- **Sermon identity is the audio URL**, after stripping the cache-busting
  query (`lib/pagetext.py` `dedup_key`). The same message is published under
  more than one series path. Keying on page URL double-indexes it.
- **Bible ingest requires 66 books.** A file that merely contains Genesis is
  rejected. Verse text stays in `data/bible/` and is never committed.
- **One embedding model owns the index.** Record model, revision, and
  dimension in `index/index-manifest.json`. Refuse to query or to finish
  `verify` on a mismatch. Swapping the embedding model requires a full
  reindex. Swapping the chat GGUF does not.
- **Three collections, weighted.** `scripture`, `beliefs`, `mosaic`. Weights
  express the source hierarchy. Do not merge them into one flat table.
- **Cross-links are the sermon's links.** A `Connected in` passage is
  attributed to the sermon. It is not Scripture citing Scripture, and it is
  not what the sermon's main text itself says.
- **Reindex is atomic.** Build into a side directory and replace the live
  index only when scripture and every transcript finish. A deadline or a kill
  leaves the live index in place.
- **Index never blocks the frontier.** Night-job order after the on-disk
  audio drain: download, crawl, discover, then index. A stuck index must not
  stop crawl or download. Two cycles with no progress stop the run.
- **Be a polite guest.** Honor `robots.txt`, identify the crawler, wait
  between requests (`Site.DelaySecondsMin` / `Max`), and abort the run on
  429 or 403.
- **Offline after the copy.** Chat server and web UI bind to loopback only.
  The query process sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`.

## Where to change things

| Intent | Start here |
|---|---|
| What an answer is allowed to claim | `lib/prompt.py` |
| What gets retrieved and boosted | `lib/retrieval.py`, `Retrieval` in settings |
| Chunk shape and sermon primary passage | `lib/chunking.py` |
| Verse reference parsing | `lib/bookmap.py` |
| Crawl scope, campuses, audio hosts | `Site` in settings, `workers/crawl.py`, `lib/pagetext.py` |
| Night order, stop time, GPU | `Mosaic-NightJob.ps1` |
| Mac install, verify, ask | `mosaic.sh`, `query/` |
| What the bundle contains | `workers/export_bundle.py` |

## Additional resources

- [architecture.md](architecture.md) — pipeline, index schema, retrieval, bundle, and tests
