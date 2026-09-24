# Windows build machine

This machine builds the library: it crawls the public site, downloads sermon
audio, transcribes it on the GPU, and maintains the search index. It also
produces the bundle the Mac queries from.

Target hardware: Windows 11, RTX 4070 Ti Super, 64 GB RAM, 100-250 GB free.

## First run

```powershell
cd C:\source\repos\mosaic-theologian
.\Mosaic-NightJob.ps1 -Action EnsureDeps
```

That installs Python, ffmpeg, and git via winget if they are missing, creates
`.venv`, installs the CUDA build of torch, downloads the Bible JSON, caches the
embedding model, and checks that faster-whisper can actually reach the GPU.
Everything it did or found is recorded in `state\deps.json`. Nothing is ever
uninstalled.

Two lines of the output are worth reading:

- `faster-whisper GPU mode confirmed` - if instead it warns that the GPU is not
  being used, transcription will run about ten times slower. The usual cause is
  missing cuDNN/cuBLAS DLLs on `PATH`.
- `Bible JSON validated: 66 books` - a smaller number means the download was
  partial and the file is rejected rather than indexed.

The chat model is a Llama 3.1 8B Instruct Q5_K_M GGUF at `models\chat.gguf`.
The crawl, transcribe, and index stages run without it. A full export
downloads that file when it is missing, then copies it into the bundle.
If you already have a GGUF there, the export keeps it.

## A useful first slice

Do not wait for twenty years of sermons before asking anything:

```powershell
.\Mosaic-NightJob.ps1 -Action Run -MaxPages 15 -Until 06:00
```

That gives you the belief statements, Scripture, and a handful of recent
messages. You can query immediately while the backfill continues on later
nights.

## Nightly

```powershell
.\Mosaic-NightJob.ps1 -Action Run -Until 06:00
```

A run does existing work first, and only crawls when those queues are empty.

**1. Repair.** Reconcile state against disk, including MP3s already in
`data\audio` that never got attached to a job.

**2. Drain what is already here.** If audio is waiting, transcribe all of it
before any crawl or any new download. You will see `NOW: transcribe N
file(s)` and a `START` / `DONE` line for each sermon. Transcripts stay in
`data\transcripts`. Audio is deleted after each success. Indexing runs
best-effort after the drain; a failed index does **not** stop crawl or
download.

**3. Then download, then crawl, then discover.** New MP3s are fetched only
when the transcribe queue is empty. New message pages are fetched only when
there is nothing left to download or transcribe. Listing-page discovery runs
after that, so a restart does not walk the archive again while 30 files sit
untouched. Indexing is last and never blocks the frontier.

`-Until` is optional. Without it (and without `-MaxMinutes`), the job keeps
cycling until work is idle or two consecutive cycles make no progress. With
`-Until 06:00`, it stops at that wall-clock time and leaves remaining work
for the next run.

CUDA is pinned in `.env` (`CUDA_VERSION` / `CUDA_PATH`). Other CUDA bins are
stripped from PATH for this process so cublas/cudnn are not loaded from two
places. If the GPU cannot load, the run stops instead of crawling for an hour.

The loop stops when there is nothing left to do, when `-Until` / `-MaxMinutes`
is reached, or when two consecutive cycles make no progress at all. That last
one is a guard against permanently failing items keeping the job spinning all
night. A sticky index failure alone does not trip that guard while download
or crawl work remains.

Useful flags:

| Flag | Effect |
|---|---|
| `-BatchSize 20` | Items per stage per cycle. Not a cap on the whole run. |
| `-MaxCycles 5` | Stop after N cycles. Handy for a short test run. |
| `-Until 06:00` | Optional wall-clock stop. Omit to run until idle/stall. |
| `-DiscoverFirst` | After on-disk audio is drained, refresh listings before download/crawl. |
| `-SkipTranscribe` | Crawl and index page text only; leave the GPU alone. On-disk audio stays put and does not block crawl/download. |
| `-MaxPages 15` | Cap how many message pages discovery picks up. |

Existing audio is always transcribed before discovery, including with
`-DiscoverFirst`. That flag only changes what happens after the on-disk
backlog is clear: listings are refreshed before new downloads.

Scheduled task:

```powershell
$action  = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\source\repos\mosaic-theologian\Mosaic-NightJob.ps1" -Action Run -Until 06:00'
$trigger = New-ScheduledTaskTrigger -Daily -At 11pm
Register-ScheduledTask -TaskName 'Mosaic night job' -Action $action -Trigger $trigger
```

Set "Stop the task if it runs longer than" as a backstop. That is a hard kill,
which is safe here for the reason below.

## Stopping is always safe

Press Ctrl+C whenever you like. Close the window. Let Task Scheduler kill it.
Pull the power.

Resume safety does not depend on catching the interrupt, because in Windows
PowerShell it often cannot be caught: the process dies without running cleanup
and takes the child `python.exe` with it. Instead:

- State advances only after the artifact it describes exists on disk.
- Artifacts are written to a `.part` file and renamed into place, and rename is
  atomic, so you get either the old file or the complete new one.
- On startup, stray `.part` files are deleted and any job whose recorded step
  has no artifact behind it is re-queued.
- A lock whose PID is gone is treated as stale, so a hard kill never blocks the
  next run.

The cost of a kill is the single item in flight. Start the script again and it
continues.

## How long the backfill takes

Roughly 1,000 to 2,000 messages at about 45 minutes each is 750 to 1,500 hours
of audio. On this GPU that is about 25 to 75 hours of transcription, so at
seven hours a night expect somewhere between four and eleven nights.

Audio is deleted after a successful transcript by default, which keeps the
working set well under the disk budget. Set `Audio.KeepAfterTranscribe` to
`true` in `config\settings.json` if you want to keep the MP3s.

## Checking on it

```powershell
.\Mosaic-NightJob.ps1 -Action Status
```

Shows how many items sit at each pipeline stage, how many audio files and
transcripts are on disk, which sermons are ready to transcribe, the scripture
and index row counts, and anything that failed. Failures are retried on the
next run. Status can be run while a job is in progress.

## Exporting for the Mac

First handoff, including the models:

```powershell
.\Mosaic-NightJob.ps1 -Action Export -Destination E:\mosaic-portable -Full
```

`-Full` downloads `models\chat.gguf` first when that file is not already on
disk (about 6 GB), then copies the index and both models.

Later re-syncs, after new sermons are indexed:

```powershell
.\Mosaic-NightJob.ps1 -Action Export -Destination E:\mosaic-portable
```

The default is index-only, which is the right choice almost always: adding a
sermon changes the index but not the model files, and the models are nearly all
of the bulk. A full export runs 5-10 GB; an index-only export is usually tens
to a few hundred MB.

Re-run `-Full` only when you deliberately change a model. Note the asymmetry:
swapping the chat model is cheap, but swapping the **embedding** model
invalidates every vector in the index and means a full rebuild here
(`-Action Run -Force`) before exporting again.

## Adding one URL

If a page was missed:

```powershell
.\Mosaic-NightJob.ps1 -Action Run -Url https://thisismosaic.org/messages/...
```

## Being a good guest on the site

The crawler reads `robots.txt`, identifies itself, waits 2-5 seconds between
requests, and stops the run entirely on a 429 or 403 rather than retrying
harder. Those values live under `Site` in `config\settings.json`. Please do not
lower the delays.

## Layout

```
Mosaic-NightJob.ps1    orchestrator
config/settings.json   all tunables
lib/                   shared with the Mac: chunking, retrieval, prompt
workers/               crawl, download, transcribe, ingest, index, export
query/                 CLI and web chat
state/                 jobs.jsonl, catalog.jsonl, deps.json, run.log
data/                  bible, pages, audio, transcripts
index/                 LanceDB: scripture, beliefs, mosaic
models/                chat GGUF and the cached embedding model
```

`state/jobs.jsonl` is append-only and the last record for an id wins, so it
doubles as a history of what happened to every item.
