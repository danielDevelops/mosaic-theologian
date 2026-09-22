"""End-to-end pipeline tests against captured pages.

Runs the real workers as subprocesses against a local fixture server, with
crawl delays set to zero, so the whole pipeline is exercised in seconds.

    python tests/test_pipeline.py

Each scenario gets a fresh throwaway project directory via MOSAIC_ROOT, so
nothing here touches real state, real data, or the live site.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixture_server import FixtureSite, available_fixtures  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
VENV_PY = PROJECT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY if VENV_PY.exists() else Path(sys.executable))

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    return ok


class Sandbox:
    """A throwaway project root wired to the fixture server."""

    def __init__(self, base_url: str) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="mosaic-test-"))
        self.base_url = base_url

        (self.dir / "config").mkdir(parents=True)
        settings = json.loads(
            (PROJECT / "config" / "settings.json").read_text(encoding="utf-8")
        )

        settings["Site"]["BaseUrl"] = base_url
        settings["Site"]["DelaySecondsMin"] = 0
        settings["Site"]["DelaySecondsMax"] = 0
        settings["Site"]["TimeoutSeconds"] = 10
        # Only the paths we captured.
        settings["Site"]["SeedPaths"] = ["/messages/archive/", "/messages/",
                                         "/about/core-beliefs/"]
        settings["Bible"]["SourceUrl"] = ""
        settings["Bible"]["FallbackUrl"] = ""

        (self.dir / "config" / "settings.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )

    def run(self, module: str, *args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["MOSAIC_ROOT"] = str(self.dir)
        env["PYTHONPATH"] = str(PROJECT)
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [PYTHON, "-X", "utf8", "-m", module, *args],
            cwd=str(PROJECT), env=env, capture_output=True, text=True, timeout=300,
        )

    def jobs(self) -> dict[str, dict]:
        path = self.dir / "state" / "jobs.jsonl"
        if not path.is_file():
            return {}
        latest: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[row["id"]] = row
        return latest

    def worklist(self) -> dict:
        proc = self.run("workers.worklist")
        try:
            return json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {"_error": proc.stdout + proc.stderr}

    def audio_files(self) -> list[Path]:
        folder = self.dir / "data" / "audio"
        return sorted(folder.glob("*.mp3")) if folder.is_dir() else []

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def scenario_discovery(site: FixtureSite) -> None:
    print("\n1. Discovery phase (listings only)")
    box = Sandbox(site.base_url)
    try:
        proc = box.run("workers.crawl", "--refresh-listings",
                       "--listings-only", "--batch-size", "0")
        check("crawl exits cleanly", proc.returncode == 0,
              proc.stderr[-400:])

        work = box.worklist()
        check("messages were enumerated", work.get("queued_messages", 0) > 0,
              f"queued_messages={work.get('queued_messages')}")
        check("all listings consumed", work.get("queued_listings", 0) == 0,
              f"queued_listings={work.get('queued_listings')}")

        # Listing pages are navigation; they must not become indexable jobs.
        jobs = box.jobs()
        listing_jobs = [j for j in jobs.values()
                        if "archive" in j["url"] and j["kind"] != "message"]
        check("listing pages did not become jobs", not listing_jobs,
              f"{len(listing_jobs)} listing job(s)")

        # No sermon detail page should have been fetched in this phase.
        detail_hits = [p for p in site.hits
                       if p.count("/") >= 3 and p.startswith("/messages/")
                       and not p.rstrip("/").endswith("archive")]
        check("no message detail pages fetched during discovery",
              not detail_hits, f"fetched {detail_hits[:3]}")
    finally:
        box.cleanup()


def scenario_repeat_run(site: FixtureSite) -> None:
    print("\n2. Repeat run (regression: KeyError on second discovery)")
    box = Sandbox(site.base_url)
    try:
        first = box.run("workers.crawl", "--refresh-listings",
                        "--listings-only", "--batch-size", "0")
        check("first discovery ok", first.returncode == 0, first.stderr[-300:])

        second = box.run("workers.crawl", "--refresh-listings",
                         "--listings-only", "--batch-size", "0")
        check("second discovery does not crash", second.returncode == 0,
              second.stderr[-400:])
        check("no KeyError", "KeyError" not in second.stderr,
              second.stderr[-300:])
        check("listings were re-queued for new messages",
              "Re-queued" in second.stdout, second.stdout[-200:])
    finally:
        box.cleanup()


def scenario_batch_and_audio(site: FixtureSite) -> None:
    print("\n3. Batch: fetch messages, download audio")
    box = Sandbox(site.base_url)
    try:
        box.run("workers.crawl", "--refresh-listings",
                "--listings-only", "--batch-size", "0")

        proc = box.run("workers.crawl", "--messages-only", "--batch-size", "5")
        check("message batch exits cleanly", proc.returncode == 0,
              proc.stderr[-400:])

        jobs = box.jobs()
        messages = [j for j in jobs.values() if j["kind"] == "message"]
        check("message jobs created", len(messages) > 0,
              f"{len(messages)} message job(s)")

        with_audio = [j for j in messages if j["audio_url"]]
        check("messages carry an audio URL", len(with_audio) > 0,
              f"{len(with_audio)} of {len(messages)}")

        work = box.worklist()
        check("audio queued for download", work.get("need_audio", 0) > 0,
              f"need_audio={work.get('need_audio')}")

        dl = box.run("workers.download_audio", "--batch-size", "5")
        check("download exits cleanly", dl.returncode == 0, dl.stderr[-400:])

        files = box.audio_files()
        check("MP3s written to disk", len(files) > 0,
              f"{len(files)} file(s)")

        after = box.worklist()
        check("state advanced to transcribe",
              after.get("need_audio", 1) == 0 and after.get("need_transcribe", 0) > 0,
              f"need_audio={after.get('need_audio')} "
              f"need_transcribe={after.get('need_transcribe')}")
    finally:
        box.cleanup()


def scenario_dedup(site: FixtureSite) -> None:
    print("\n4. Deduplication (same sermon, two series paths)")
    box = Sandbox(site.base_url)
    try:
        a = f"{site.base_url}/messages/1st-john/1-john-1.3-4/"
        b = f"{site.base_url}/messages/the-letters-of-john/1-john-1.3-4-2/"
        box.run("workers.crawl", "--single", a)
        box.run("workers.crawl", "--single", b)

        jobs = box.jobs()
        messages = [j for j in jobs.values() if j["kind"] == "message"]
        check("same sermon stored once", len(messages) == 1,
              f"{len(messages)} job(s): {[j['url'] for j in messages]}")

        if messages:
            check("duplicate URL recorded as an alternate",
                  len(messages[0]["alt_urls"]) == 1,
                  f"alt_urls={messages[0]['alt_urls']}")
    finally:
        box.cleanup()


def scenario_single_targeted(site: FixtureSite) -> None:
    print("\n5. --single fetches only the URL given")
    box = Sandbox(site.base_url)
    try:
        box.run("workers.crawl", "--refresh-listings",
                "--listings-only", "--batch-size", "0")
        before = box.worklist().get("queued_messages", 0)

        target = f"{site.base_url}/messages/hebrews/hebrews-4.14-16/"
        proc = box.run("workers.crawl", "--single", target)
        check("single crawl exits cleanly", proc.returncode == 0,
              proc.stderr[-300:])

        jobs = box.jobs()
        messages = [j for j in jobs.values() if j["kind"] == "message"]
        check("exactly one message fetched", len(messages) == 1,
              f"{len(messages)} message job(s)")

        after = box.worklist().get("queued_messages", 0)
        check("queue only dropped by one", before - after == 1,
              f"{before} -> {after}")
    finally:
        box.cleanup()


def scenario_resume(site: FixtureSite) -> None:
    print("\n6. Resume after a hard kill")
    box = Sandbox(site.base_url)
    try:
        box.run("workers.crawl", "--refresh-listings",
                "--listings-only", "--batch-size", "0")
        box.run("workers.crawl", "--messages-only", "--batch-size", "3")
        before = len(box.jobs())

        # A kill mid-append leaves a torn final line.
        jobs_path = box.dir / "state" / "jobs.jsonl"
        with open(jobs_path, "a", encoding="utf-8") as handle:
            handle.write('{"id": "torn", "kind": "mess')

        # And a stray .part from an interrupted artifact write.
        (box.dir / "data" / "pages" / "ghost.json.part").write_text("x", encoding="utf-8")

        rec = box.run("workers.reconcile")
        check("reconcile exits cleanly", rec.returncode == 0, rec.stderr[-300:])
        check("torn line ignored, jobs intact", len(box.jobs()) == before,
              f"{before} -> {len(box.jobs())}")
        check("stray .part cleaned up",
              not (box.dir / "data" / "pages" / "ghost.json.part").exists())

        nxt = box.run("workers.crawl", "--messages-only", "--batch-size", "2")
        check("crawl resumes after damage", nxt.returncode == 0,
              nxt.stderr[-300:])
    finally:
        box.cleanup()


def scenario_wdw_excluded(site: FixtureSite) -> None:
    print("\n7. WDW campus excluded")
    box = Sandbox(site.base_url)
    try:
        box.run("workers.crawl", "--refresh-listings",
                "--listings-only", "--batch-size", "0")
        box.run("workers.crawl", "--messages-only", "--batch-size", "10")

        jobs = box.jobs()
        wdw = [j for j in jobs.values() if j.get("campus") == "wdw"]
        check("no WDW messages indexed", not wdw,
              f"{len(wdw)} wdw job(s)")

        wdw_audio = [j for j in jobs.values() if "wdw.mp3" in (j.get("audio_url") or "")]
        check("no WDW audio queued", not wdw_audio, f"{len(wdw_audio)}")
    finally:
        box.cleanup()


def main() -> int:
    fixtures = available_fixtures()
    if not fixtures:
        print("No fixtures. Run: python tests/capture_fixtures.py")
        return 2

    print(f"Fixtures: {len(fixtures)} page(s)")

    with FixtureSite() as site:
        print(f"Fixture server: {site.base_url}")
        scenario_discovery(site)
        scenario_repeat_run(site)
        scenario_batch_and_audio(site)
        scenario_dedup(site)
        scenario_single_targeted(site)
        scenario_resume(site)
        scenario_wdw_excluded(site)

    passed = sum(1 for _, ok, _ in results if ok)
    failed = [(n, d) for n, ok, d in results if not ok]

    print("\n" + "=" * 66)
    print(f"{passed}/{len(results)} checks passed")
    if failed:
        print("\nFailures:")
        for name, detail in failed:
            print(f"  - {name}")
            if detail:
                print(f"      {detail[:300]}")
    print("=" * 66)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
