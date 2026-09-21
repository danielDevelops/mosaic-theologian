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

Then drop a chat model GGUF at `models\chat.gguf`. Llama 3.1 8B Instruct
Q5_K_M is a good default because the same file also runs on the Mac. The
crawl, transcribe, and index stages all work without it; only chat needs it.

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

Stages run in order: reconcile, crawl, Bible ingest, audio download,
transcribe, index. `-Until` is checked between items, so the job stops on an
item boundary and leaves the GPU free during the day.

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

Shows how many items sit at each pipeline stage, the scripture and index row
counts, the embedding model identity, and anything that failed. Failures are
retried on the next run rather than abandoned. Status can be run while a job
is in progress.

## Exporting for the Mac

First handoff, including the models:

```powershell
.\Mosaic-NightJob.ps1 -Action Export -Destination E:\mosaic-portable -Full
```

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
