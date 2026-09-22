"""Crawl the public Mosaic site and queue messages for download.

Polite by construction: robots.txt is honoured, requests are spaced by a
randomised delay, and a 429 or 403 ends the run rather than retrying harder.
The frontier lives in state/catalog.jsonl so a killed crawl resumes instead of
starting over.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from lib.pagetext import dedup_key, extract, is_listing_url, is_message_url
from lib.state import Job, atomic_write_json, utcnow
from workers.common import WorkerContext, base_parser, log

STOP_STATUSES = {401, 403, 429, 451}


class Catalog:
    """Discovered URLs and whether they have been fetched."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.entries[row["url"]] = row

    def add(self, url: str, source: str = "") -> bool:
        if url in self.entries:
            return False
        self._append({"url": url, "state": "queued",
                      "source": source, "updated": utcnow()})
        return True

    def mark(self, url: str, state: str, note: str = "") -> None:
        self._append({"url": url, "state": state, "note": note,
                      "updated": utcnow()})

    def _append(self, row: dict) -> None:
        self.entries[row["url"]] = row
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    def queued(self) -> list[str]:
        return [u for u, r in self.entries.items() if r.get("state") == "queued"]


def build_robots(base_url: str, user_agent: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(urljoin(base_url, "/robots.txt"))
    try:
        parser.read()
    except Exception as exc:
        log(f"robots.txt unreadable ({exc}); defaulting to allow with delays.")
    return parser


def allowed(url: str, robots: RobotFileParser, user_agent: str,
            deny_patterns: list[str]) -> bool:
    lowered = url.lower()
    if any(pattern.lower() in lowered for pattern in deny_patterns):
        return False
    try:
        return robots.can_fetch(user_agent, url)
    except Exception:
        return True


def same_site(url: str, base_netloc: str) -> bool:
    return urlparse(url).netloc.lower() in ("", base_netloc.lower())


def slug_for(url: str) -> str:
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    return "-".join(parts)[:120] or "index"


def main() -> int:
    parser = base_parser("Crawl thisismosaic.org")
    parser.add_argument("--single", default=None, help="Queue one URL and stop")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Stop after discovering this many message pages")
    parser.add_argument("--refresh-listings", action="store_true",
                        help="Re-queue archive and series pages to find new "
                             "messages. Pass once per run, not once per batch.")
    args = parser.parse_args()

    with WorkerContext(args) as ctx:
        site = ctx.settings["Site"]
        base = site["BaseUrl"].rstrip("/")
        base_netloc = urlparse(base).netloc
        user_agent = site["UserAgent"]
        deny = list(site.get("DenyPatterns", []))
        belief_paths = {p.rstrip("/") for p in site.get("BeliefPaths", [])}
        audio_hosts = list(site.get("AudioHosts", []))
        delay_min = float(site.get("DelaySecondsMin", 2.0))
        delay_max = float(site.get("DelaySecondsMax", 5.0))
        timeout = int(site.get("TimeoutSeconds", 45))

        catalog = Catalog(ctx.paths.state / "catalog.jsonl")
        robots = build_robots(base, user_agent)

        session = requests.Session()
        session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })

        if args.single:
            catalog.add(args.single, source="manual")
            log(f"Queued {args.single}")

        elif not catalog.entries:
            for path in site["SeedPaths"]:
                catalog.add(urljoin(base + "/", path.lstrip("/")), source="seed")
            log(f"Seeded {len(catalog.entries)} start URLs")

        elif args.refresh_listings:
            # Every URL is marked fetched after the first full crawl, so
            # without this the frontier stays empty and a new sermon is never
            # discovered. Re-queue only the pages that gain links over time;
            # message pages do not change once published.
            requeued = 0

            for path in site["SeedPaths"]:
                url = urljoin(base + "/", path.lstrip("/"))
                if catalog.add(url, source="seed"):
                    requeued += 1
                elif catalog.entries[url].get("state") != "queued":
                    catalog.mark(url, "queued", "revisit")
                    requeued += 1

            for url, row in list(catalog.entries.items()):
                if row.get("state") == "queued":
                    continue
                if is_listing_url(url):
                    catalog.mark(url, "queued", "revisit")
                    requeued += 1

            log(f"Re-queued {requeued} listing page(s) to look for new messages.")

        processed = 0
        discovered_messages = 0
        frontier = catalog.queued()

        while frontier:
            if ctx.should_stop(processed):
                break

            url = frontier.pop(0)

            if not same_site(url, base_netloc):
                catalog.mark(url, "skipped", "offsite")
                continue
            if not allowed(url, robots, user_agent, deny):
                catalog.mark(url, "skipped", "robots or deny pattern")
                continue

            try:
                response = session.get(url, timeout=timeout)
            except requests.RequestException as exc:
                catalog.mark(url, "error", str(exc)[:200])
                log(f"  error {url}: {exc}")
                continue

            if response.status_code in STOP_STATUSES:
                # Back off for the night rather than hammering the site.
                catalog.mark(url, "queued", f"http {response.status_code}")
                log(f"Received HTTP {response.status_code}. Stopping this run; "
                    f"state is saved and the next run resumes here.")
                break

            if response.status_code != 200:
                catalog.mark(url, "error", f"http {response.status_code}")
                continue

            if "text/html" not in response.headers.get("Content-Type", ""):
                catalog.mark(url, "skipped", "not html")
                continue

            page = extract(response.text, url, audio_hosts=audio_hosts)
            processed += 1

            # Queue newly found in-domain links.
            for link in page.links:
                if same_site(link, base_netloc) and allowed(link, robots, user_agent, deny):
                    catalog.add(link, source=url)

            path_key = urlparse(url).path.rstrip("/")
            is_belief = path_key in belief_paths
            is_message = is_message_url(url)

            if not (is_message or is_belief or page.body):
                catalog.mark(url, "fetched", "no usable body")
                time.sleep(random.uniform(delay_min, delay_max))
                continue

            # Identity is date + campus + audio, never the page URL: the same
            # sermon is published under more than one series path.
            key = dedup_key(page.date, page.campus, page.audio_url) if is_message else f"page:{path_key}"
            existing = ctx.jobs.by_key(key)

            if existing and not ctx.force:
                if url != existing.url and url not in existing.alt_urls:
                    existing.alt_urls.append(url)
                    ctx.jobs.put(existing)
                    log(f"  duplicate of {existing.id}: {url}")
                catalog.mark(url, "fetched", "duplicate")
                time.sleep(random.uniform(delay_min, delay_max))
                continue

            job_id = existing.id if existing else slug_for(url)
            page_path = ctx.paths.pages / f"{job_id}.json"

            # Artifact first, then state. If this process dies between the two,
            # reconcile re-queues the item on the next run.
            atomic_write_json(page_path, page.to_dict())

            job = existing or Job(id=job_id, kind="message" if is_message else "page", url=url)
            job.key = key
            job.title = page.title
            job.speaker = page.speaker
            job.date = page.date
            job.campus = page.campus
            job.series = page.series
            job.audio_url = page.audio_url
            job.scripture_refs = page.scripture_refs
            job.page_path = ctx.relative(page_path)
            job.kind = "belief" if is_belief else job.kind
            job.advance("page_saved")
            if not page.audio_url:
                # No audio is a normal outcome for older years and for pages.
                # Index the text and move on rather than failing the item.
                job.advance("transcribed")
            job.error = ""
            ctx.jobs.put(job)

            catalog.mark(url, "fetched")

            if is_message:
                discovered_messages += 1
                log(f"  [{discovered_messages}] {page.date or '????-??-??'} {page.title}")
                if args.max_pages and discovered_messages >= args.max_pages:
                    log(f"Reached --max-pages {args.max_pages}")
                    break

            time.sleep(random.uniform(delay_min, delay_max))

            if not frontier:
                frontier = catalog.queued()

        remaining = len(catalog.queued())
        log(f"Crawl finished: {processed} fetched this run, "
            f"{discovered_messages} messages, {remaining} URLs still queued.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
