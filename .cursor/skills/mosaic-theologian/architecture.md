# Architecture reference

Contracts for a rebuild. Tunable numbers live in `config/settings.json`;
this file names the fields, not a snapshot of every value.

## Machines

| | Windows build | Intel Mac query |
|---|---|---|
| Entry | `Mosaic-NightJob.ps1` | `mosaic.sh` |
| Hardware assumption | RTX 4070 Ti Super, CUDA | CPU only. No Metal path |
| Does | crawl, download, transcribe, index, export | verify, then ask |
| Does not | serve as the study UI | crawl, download, transcribe, or call Windows |

`remote-status.sh` only reads a mounted SMB share of the Windows repo and
prints queue counts. It is not part of asking or building.

`lib/` and `query/` are copied into the bundle so both machines execute the
same retrieval and prompt code.

## Repository layout

```
Mosaic-NightJob.ps1     Windows orchestrator
mosaic.sh               Mac runtime
remote-status.sh        read-only status over the Windows share
config/settings.json    every tunable
lib/                    shared library (shipped in the bundle)
workers/                build-machine stages
query/                  CLI, web chat, bundle verifier
tests/                  pipeline, cross-link, and chat-model checks
docs/                   README-Windows.md, README-Mac.md
```

Gitignored and produced at runtime: `data/bible/`, `data/pages/`,
`data/audio/`, `data/transcripts/`, `index/`, `models/`, `state/*`,
`.env`, `tests/fixtures/`. Scripture text is never committed.

## Settings

`lib/config.py` loads `config/settings.json`, then deep-merges
`config/settings.local.json` when that file exists. Export writes a local
override that sets `Embedding.Device` to `cpu` on the Mac. Machine-local
edits belong in the local file so a re-sync does not clobber them.

Sections: `Bible`, `Site`, `Audio`, `Whisper`, `Embedding`, `Chat`,
`Chunking`, `Retrieval`, `Web`, `Export`.

## Pipeline

`lib/state.py` is the source of the step list. A job (`kind` message, page,
belief, or bible) moves only forward.

```
discovered → page_saved → audio_downloaded → transcribed → chunked → indexed
```

`state/jobs.jsonl` is append-only. `JobStore.put` fsyncs each record.
`requeue_missing_artifacts` pulls a job back to `discovered` when `page_saved`
or `transcribed` is recorded but the file is gone. `RunLock` stores a PID and
treats a dead PID as stale.

Artifacts use `atomic_write_text` / `atomic_write_json`: write `name.part`,
then `os.replace`.

### Night-job cycle

`Invoke-Run` in `Mosaic-NightJob.ps1`:

1. Reconcile (stray `.part` files, missing artifacts, MP3s on disk with no job).
2. Ingest the Bible if `verses.jsonl` is missing or forced.
3. Drain on-disk audio: transcribe all of it before any crawl or new download.
   Delete the MP3 after a successful transcript unless
   `Audio.KeepAfterTranscribe` is true.
4. Then, repeatedly: download, else crawl message pages, else discover
   listings, else index. `-DiscoverFirst` refreshes listings after the
   on-disk drain and before new downloads. `-SkipTranscribe` leaves audio
   on disk and does not let that queue block other work.
5. Stop at `-Until`, `-MaxMinutes`, `-MaxCycles`, idle, or two cycles with
   an unchanged work signature. A stuck index does not count as a stall
   while download or crawl work remains.

`-Action Reindex` rebuilds into a side index and calls `replace_index` only
when scripture and transcripts both finish. It does not crawl, download, or
transcribe.

### Stage modules

| Module | Contract |
|---|---|
| `workers/reconcile.py` | Attach orphan MP3s; repair state before work |
| `workers/ingest_bible.py` | Normalize the Bible JSON to `data/bible/verses.jsonl`. Reject under `Bible.ExpectedBookCount` |
| `workers/crawl.py` | `robots.txt`, same-site, seed paths, deny patterns, `FollowScope`, belief paths. Abort the run on 429 or 403 |
| `workers/download_audio.py` | Fetch MP3s from `Site.AudioHosts` within `Audio.MinBytes` / `MaxBytes` |
| `workers/transcribe.py` | faster-whisper. GPU required when `Whisper.FailIfNoGpu` is set. CUDA is pinned by `lib/cuda_env.py` and `.env` so one toolkit is on `PATH` |
| `workers/index_build.py` | Chunk, embed, upsert. `--rebuild` publishes only a finished side index |
| `workers/export_bundle.py` | Copy the bundle and write `manifest.json` checksums |
| `workers/worklist.py` | Counts the orchestrator uses to pick the next stage |
| `workers/status.py` | Stage counts, disk counts, failures |
| `lib/chat_model.py` | Download the chat GGUF on full export if missing; reject a short file |

Page identity for a sermon is `dedup_key`: normalized audio URL, else
`date|campus`. Alternate series URLs are stored on the job, not as new jobs.
`Site.ExcludeCampuses` drops campuses such as `wdw`.

`lib/pagetext.py` extracts title, date, speaker, campus, series, audio URL,
heading sections, and scripture references from saved HTML.

## Index

LanceDB directory `index/`, three tables named by collection. Schema is
`schema_for` in `lib/store.py`: citation metadata, `book_id` / `chapter` /
`verse_start` / `verse_end`, packed `scripture_refs`, packed `primary_refs`,
`source_id`, and a fixed-width float vector.

`source_id` is the job id. Re-index deletes that source's rows before adding
new ones. Belief page chunks use `{job_id}:page`.

Search returns L2 distance on normalized vectors, mapped to similarity
`1 - distance / 2`. `filter_rows` is the metadata path used for cross-links
and exact passage lookup.

`index-manifest.json` stores the embedding identity and row counts.
`assert_embedding_matches` refuses to serve on model, revision, or dimension
disagreement. An index with no manifest is not treated as a mismatch.

### Chunking (`lib/chunking.py`)

Token counts are `word_count * 1.33`. No tokenizer dependency.

| Source | Function | Shape |
|---|---|---|
| Sermon transcript | `chunk_sermon` | Sentence windows, `Chunking.SermonTargetTokens` / `SermonOverlapTokens` |
| Belief statement | `chunk_beliefs` | One chunk per heading. Never split |
| Article or page with no transcript | `chunk_page` | One chunk per heading; split only past `PageMaxTokens` |
| Scripture | `chunk_bible_chapter` | Whole chapter plus verse windows of `VerseWindowSize` stepping by `VerseWindowStride` |

A sermon with a transcript does not also index its page prose. Page prose
would duplicate the sermon. Beliefs are the exception: they have no transcript
and are their own collection.

### Scripture references (`lib/bookmap.py`)

English names and aliases map to OSIS ids. Unknown book names raise; dropping
one would hide a hole in the Bible index. References pack to a single string
column and unpack on read. Overlap treats `Genesis 6` and `Genesis 6:4` as
the same link.

`primary_refs` is the passage the sermon is about: title, page refs, page
title, then job refs. If those are empty, the transcript opening decides
(`primary_from_opening`). Every chunk of that sermon stores the same list.

`scripture_refs` on a chunk are the references inside that window, including
asides.

### Embeddings (`lib/embedding.py`)

`Embedding.Model` and `Revision` are the identity. Files are cached under
`Embedding.LocalDir` and travel inside a full bundle. Queries are prefixed
with `Embedding.QueryPrefix`; documents are not. The Mac loads only the local
snapshot, with the hub disabled.

## Retrieval and the prompt

`lib/retrieval.py` embeds the question, then searches each collection for
`TopK * CandidateMultiplier` hits. Score is similarity times `Retrieval.Weights`,
plus a reference boost:

- Scripture whose citation overlaps a reference in the question, or a sermon
  whose `primary_refs` overlap it: `PrimaryRefBoost`.
- A sermon that only mentions the reference: `MentionedRefBoost`.
- Older rows with empty `primary_refs` still boost when the citation string
  itself names the passage.

Hits are deduped on a text prefix and sorted. After the act-restatement
search below, they are trimmed to `Retrieval.MaxContextChars`. Scripture
windows from that search are kept first. At least one passage from each
other collection is kept when it fits.

### Cross-links

Before the budget trim, every question asks the local chat model for up to
`ExpansionMaxQueries` short search phrases. Each phrase restates the concrete
act in the question, in plain wording Scripture or a sermon would use for that
act. No verdict and no modern label. There is no topic list. Those phrases are
embedded and searched in scripture, beliefs, and mosaic. Direct hits are kept;
a duplicate keeps the earlier copy. Up to `ExpansionMaxScripture` new scripture
windows are reserved so the budget trim cannot drop them in favor of a
higher-scoring sermon. Other new chunks are merged up to `ExpansionMaxPassages`.
If that step returns nothing, the question stays a single embedding.

After the budget trim, `cross_link_passages` may append up to
`CrossLinkMaxPassages` scripture windows:

- Group mosaic hits by `source_id`.
- A reference is "sustained" when it appears in at least `CrossLinkMinChunks`
  chunks and is not the sermon's primary passage. One overlapping form counts
  once per chunk.
- If the question names a reference, keep sustained refs that overlap the
  question. If the question already hit the primary passage, do not expand
  that sermon's illustrations of the primary text.
- If the question names nothing, keep sustained refs that also appear in the
  retrieved windows.
- Fetch the tightest indexed scripture window that covers the reference
  (`select_scripture_window`). Label it `Connected in {sermon}`.

Those sustained refs are also a one-hop bridge to other sermons.
`related_sermon_passages` keeps up to `CrossLinkMaxRelatedSermons` sermons
whose `primary_refs` overlap a bridge ref, one chunk each (the window that
quotes the ref), labeled `Also teaches {ref}`. Sermons already retrieved are
skipped. A passing mention does not qualify. The related sermon's own asides
are not followed. The scan of `primary_refs` is not cut off at 500 rows.
Scripture links and related sermons still have to fit in `MaxContextChars`.

`lib/prompt.py` renders connected Scripture and related sermons under church
teaching, not under the Scripture block. `SYSTEM_PROMPT` is the authority
order. The reply is one integrated answer: Scripture leads, church teaching
is woven in, and general knowledge is marked in the sentence rather than under
its own heading. A modern word in the question is answered by naming the act
the supplied passages describe. General knowledge does not fill that gap or
contradict a supplied passage. The model does not print a source list;
citations are appended after the answer. `mosaic_is_silent` is true when no
beliefs or mosaic passage has raw similarity above 0.35; the user message then
tells the model to say the church has not addressed it.

`build_messages` keeps the last six history turns. Citations come from
`format_citations`.

Chat is `llama.cpp` via `lib/llm.py`, model file `Chat.ModelFile`, server
`Chat.ServerUrl` on loopback. The web UI (`query/web.py`) refuses a
non-loopback bind (`Web.Host` / `Web.Port`).

## Bundle

`Mosaic-NightJob.ps1 -Action Export`:

- Default index-only: index, transcripts, pages, Bible metadata, settings,
  `lib/`, `query/`, `mosaic.sh`, `requirements-mac.txt`, Mac README.
- `-Full`: also the embedding snapshot and the chat GGUF. Download the GGUF
  first if it is missing or shorter than `Chat.MinBytes`.
- Audio is excluded (`Export.ExcludeAudio`).
- `manifest.json` at the bundle root has sha256 and size for every file,
  plus embedding identity and collection counts.

`./mosaic.sh verify` (`query/verify_bundle.py`) replays checksums, checks
index counts, and loads the embedding model. Exit `0` verified, `1` bundle
problem, `2` embedding mismatch. Do not bypass a mismatch.

## Tests

| Test | What it locks |
|---|---|
| `tests/test_pipeline.py` | Discovery, rerun, batching, dedup, single URL, resume, orphan audio, excluded campus. Uses `tests/fixture_server.py` and `MOSAIC_ROOT` |
| `tests/test_cross_links.py` | Ref pack/unpack, primary stamp, sustained filter, boosts, cross-link hops, related sermons, expansion matching, prompt labels, side-index replace |
| `tests/test_chat_model.py` | GGUF download size checks |
| `tests/test_embedding_snapshot.py` | 6.1 Normalize path loads on 5.7 without editing the bundled file |

`tests/fixtures/` is a local capture cache (`tests/capture_fixtures.py`), not
source.

## Rebuild checklist

- [ ] Settings load from the tree root and local overrides deep-merge
- [ ] Job log is append-only; steps never regress; missing artifacts re-queue
- [ ] Artifacts rename into place; startup deletes `*.part`
- [ ] Sermons dedupe on normalized audio URL
- [ ] Bible ingest refuses fewer than 66 books
- [ ] Three collections; beliefs stay whole; sermon page prose is not double-indexed
- [ ] Manifest records embedding identity; query and verify refuse a mismatch
- [ ] Reference boosts and cross-links match `lib/retrieval.py`
- [ ] Prompt writes one answer, attributes `Connected in` and `Also teaches`, and admits silence
- [ ] Night loop drains on-disk audio first and does not let index starve crawl
- [ ] Reindex replaces the live index only when the side index is finished
- [ ] Export checksums every shipped file; Mac verify replays them
- [ ] Chat and web bind to loopback; query env forces the hub offline
- [ ] Verse text is not in git
