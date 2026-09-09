#!/usr/bin/env python3
"""Expand papers.jsonl by walking the citation graph out from your includes.

    python snowball.py --seeds screened.csv --papers papers.jsonl

For each seed it pulls (a) what the seed cites — backwards, finds the
pre-2021 canon OpenReview doesn't have — and (b) what cites the seed —
forwards, finds recent work that used different vocabulary than your
keywords. New papers are appended to papers.jsonl in the same schema, so
`python screen.py` afterwards picks them up and skips everything already done.

Ranking signal: `seed_links` counts how many of your seeds a paper connects
to. A paper cited by six of your includes is almost certainly relevant even
if its abstract never says "hyperparameter". Sort by that before reading.

Semantic Scholar's API is free. Unauthenticated it's a shared pool and you
will hit 429s; a free key (https://www.semanticscholar.org/product/api)
raises the limit substantially. Set S2_API_KEY if you have one.
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import requests

S2 = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,venue,externalIds,citationCount,openAccessPdf"


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


# --------------------------------------------------------------------------
# rate-limited session
# --------------------------------------------------------------------------

class S2Client:
    def __init__(self, rps=1.0, max_retries=5):
        self.session = requests.Session()
        key = os.environ.get("S2_API_KEY")
        if key:
            self.session.headers["x-api-key"] = key
        self.session.headers["User-Agent"] = "lit-screen/1.0"
        self.min_interval = 1.0 / rps
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._last = 0.0

    def _wait(self):
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()

    def get(self, path, **params):
        for attempt in range(self.max_retries):
            self._wait()
            try:
                r = self.session.get(f"{S2}{path}", params=params, timeout=30)
            except requests.RequestException as e:
                print(f"  ! {type(e).__name__} on {path}", file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                print(f"  {r.status_code}, waiting {wait:.0f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  ! HTTP {r.status_code} on {path}", file=sys.stderr)
            return None
        return None


# --------------------------------------------------------------------------
# seed resolution
# --------------------------------------------------------------------------

def arxiv_from_doi(doi: str):
    """arXiv's DataCite DOIs (10.48550/arXiv.NNNN) are not indexed by S2 as
    DOIs, but the same paper is indexed by its arXiv id. Convert rather than
    lose the lookup."""
    m = re.match(r"10\.48550/arxiv\.(.+)$", doi or "", re.I)
    return m.group(1) if m else None


def doi_via_openalex(session, title, key=None):
    """Title -> DOI, so we can use S2's fast id lookup instead of its search.

    S2's /paper/search/match runs at roughly one request a minute even with a
    key. OpenAlex answers the same question in ~0.2s, and a DOI turns the S2
    call into a direct lookup.
    """
    q = re.sub(r"[,|]", " ", title).strip()
    try:
        r = session.get("https://api.openalex.org/works",
                        params={"filter": f"title.search:{q}", "per-page": 3,
                                "select": "doi,display_name"},
                        headers={"Authorization": f"Bearer {key}"} if key else {},
                        timeout=45)
    except requests.RequestException:
        return None
    if not r.ok:
        return None
    for w in r.json().get("results", []):
        if norm_title(w.get("display_name")) == norm_title(title) and w.get("doi"):
            return w["doi"].replace("https://doi.org/", "")
    return None


def resolve_seed(client, seed, session=None, oa_key=None):
    """Get an S2 paperId for one of our papers.

    Prefer an exact id we already hold (arXiv, DOI). Only OpenReview and bare
    OpenAlex records need resolving, and those go via OpenAlex rather than S2's
    search endpoint, which is too slow to use in bulk.
    """
    pid = seed.get("id", "")
    kind, _, rest = pid.partition(":")
    exact = None
    if kind == "arxiv":
        exact = f"ARXIV:{rest.split('v')[0]}"
    elif kind == "doi":
        arx = arxiv_from_doi(rest)
        exact = f"ARXIV:{arx}" if arx else f"DOI:{rest}"
    elif kind == "s2":
        return rest
    if exact:
        data = client.get(f"/paper/{exact}", fields="paperId,title")
        if data and data.get("paperId"):
            return data["paperId"]

    if session is not None:
        doi = doi_via_openalex(session, seed["title"], oa_key)
        if doi:
            arx = arxiv_from_doi(doi)
            ref = f"ARXIV:{arx}" if arx else f"DOI:{doi}"
            data = client.get(f"/paper/{ref}", fields="paperId,title")
            if data and data.get("paperId"):
                return data["paperId"]
        return None

    data = client.get("/paper/search/match", query=seed["title"], fields="paperId,title")
    if not data:
        return None
    match = (data.get("data") or [{}])[0]
    # search/match is fuzzy; reject it if the title isn't actually the same paper.
    if match.get("paperId") and norm_title(match.get("title")) == norm_title(seed["title"]):
        return match["paperId"]
    if match.get("title"):
        print(f"  ? title mismatch, skipping: {seed['title'][:55]!r} "
              f"-> {match['title'][:55]!r}", file=sys.stderr)
    return None


# --------------------------------------------------------------------------
# graph walk
# --------------------------------------------------------------------------

def fetch_edge(client, paper_id, direction, limit):
    """direction: 'references' (what it cites) or 'citations' (what cites it)."""
    key = "citedPaper" if direction == "references" else "citingPaper"
    out, offset = [], 0
    while offset < limit:
        page = client.get(f"/paper/{paper_id}/{direction}",
                          fields=FIELDS, limit=min(100, limit - offset), offset=offset)
        if not page or not page.get("data"):
            break
        for row in page["data"]:
            p = row.get(key)
            if p and p.get("title"):
                out.append(p)
        if len(page["data"]) < 100:
            break
        offset += 100
    return out


def to_record(p, direction):
    ext = p.get("externalIds") or {}
    arx = ext.get("ArXiv")
    if arx:
        pid, url = f"arxiv:{arx}", f"https://arxiv.org/abs/{arx}"
    elif ext.get("DOI"):
        pid, url = f"doi:{ext['DOI']}", f"https://doi.org/{ext['DOI']}"
    else:
        pid, url = f"s2:{p['paperId']}", f"https://www.semanticscholar.org/paper/{p['paperId']}"
    return {
        "id": pid,
        "source": "s2-snowball",
        "venue_id": None,
        "venue": p.get("venue") or "",
        "accepted": None,
        "title": (p.get("title") or "").replace("\n", " ").strip(),
        "abstract": (p.get("abstract") or "").replace("\n", " ").strip(),
        "keywords": [],
        "url": url,
        "pdf": (p.get("openAccessPdf") or {}).get("url"),
        "year": p.get("year"),
        "direction": direction,
        "citation_count": p.get("citationCount"),
        "seed_links": 1,
    }


# --------------------------------------------------------------------------

def matches(rec, patterns) -> bool:
    """Same idea as fetch.py --require. With many seeds, seed_links surfaces
    whatever the whole field cites (PPO, Adam, Gym) rather than your topic, so
    the graph needs a topical filter as much as a keyword search does."""
    if not patterns:
        return True
    hay = f"{rec.get('title') or ''} {rec.get('abstract') or ''}".lower()
    return all(rx.search(hay) for rx in patterns)


def load_seeds(path, decisions):
    p = Path(path)
    rows = []
    if p.suffix == ".csv":
        with p.open() as f:
            rows = list(csv.DictReader(f))
    else:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rows if (r.get("decision") or "").strip().lower() in decisions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="screened.csv",
                    help="screened.csv (respects hand-edits) or screened.jsonl")
    ap.add_argument("--papers", default="papers.jsonl", help="appended to in place")
    ap.add_argument("--decisions", default="include",
                    help="comma-separated: include / include,maybe")
    ap.add_argument("--direction", default="both",
                    choices=["both", "references", "citations"])
    ap.add_argument("--max-per-seed", type=int, default=100)
    ap.add_argument("--min-year", type=int, help="drop papers older than this")
    ap.add_argument("--min-links", type=int, default=1,
                    help="only keep papers connected to at least N seeds")
    ap.add_argument("--rps", type=float, default=1.0, help="requests/sec to S2")
    ap.add_argument("--require", action="append", default=[], metavar="REGEX",
                    help="keep only papers whose title/abstract match this "
                         "(repeatable; all must match)")
    ap.add_argument("--dump", metavar="FILE",
                    help="write every reached paper here regardless of filters "
                         "or --dry-run. The crawl is the expensive part; never "
                         "throw it away")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be added, write nothing")
    args = ap.parse_args()

    decisions = {d.strip().lower() for d in args.decisions.split(",")}
    seeds = load_seeds(args.seeds, decisions)
    if not seeds:
        print(f"no seeds with decision in {decisions} — screen something first",
              file=sys.stderr)
        return
    print(f"{len(seeds)} seeds", file=sys.stderr)

    papers_path = Path(args.papers)
    known_ids, known_titles = set(), set()
    if papers_path.exists():
        for line in papers_path.read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                known_ids.add(p["id"])
                known_titles.add(norm_title(p["title"]))
    print(f"{len(known_ids)} papers already in {papers_path}", file=sys.stderr)

    client = S2Client(rps=args.rps)
    session = requests.Session()
    session.headers["User-Agent"] = "lit-screen/1.0"
    oa_key = os.environ.get("OPENALEX_API_KEY")
    dirs = ["references", "citations"] if args.direction == "both" else [args.direction]

    found, unresolved = {}, 0
    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] {seed['title'][:60]}", file=sys.stderr)
        sid = resolve_seed(client, seed, session, oa_key)
        if not sid:
            unresolved += 1
            continue
        for d in dirs:
            for p in fetch_edge(client, sid, d, args.max_per_seed):
                rec = to_record(p, d)
                key = norm_title(rec["title"])
                if not key:
                    continue
                if key in found:
                    # seen via another seed — that's the useful signal
                    found[key]["seed_links"] += 1
                    if found[key]["direction"] != d:
                        found[key]["direction"] = "both"
                else:
                    found[key] = rec

    if args.dump:
        with open(args.dump, "w") as f:
            for rec in found.values():
                f.write(json.dumps(rec) + "\n")
        print(f"dumped {len(found)} reached papers -> {args.dump}", file=sys.stderr)

    # ---- filter ----
    patterns = [re.compile(r, re.I) for r in args.require]
    new = []
    skipped = defaultdict(int)
    for key, rec in found.items():
        if rec["id"] in known_ids or key in known_titles:
            skipped["already have"] += 1
        elif rec["seed_links"] < args.min_links:
            skipped["too few seed links"] += 1
        elif args.min_year and (rec["year"] or 0) < args.min_year:
            skipped["too old"] += 1
        elif not matches(rec, patterns):
            skipped["off-topic"] += 1
        elif not rec["abstract"]:
            # screen.py judges on abstracts; without one it can't. Keep it but
            # flag it, so you eyeball these by hand rather than losing them.
            rec["no_abstract"] = True
            new.append(rec)
        else:
            new.append(rec)

    new.sort(key=lambda r: (-r["seed_links"], -(r.get("citation_count") or 0)))

    print(f"\n{len(found)} unique papers reached, {len(new)} new", file=sys.stderr)
    for k, v in skipped.items():
        print(f"  skipped {v} ({k})", file=sys.stderr)
    if unresolved:
        print(f"  {unresolved} seeds could not be resolved on S2", file=sys.stderr)
    no_abs = sum(1 for r in new if r.get("no_abstract"))
    if no_abs:
        print(f"  {no_abs} have no abstract on S2 — grep '\"no_abstract\": true' "
              f"and check those by hand", file=sys.stderr)

    print("\ntop by seed_links:", file=sys.stderr)
    for r in new[:15]:
        print(f"  {r['seed_links']:2d} links {str(r['year'] or '????'):>4}  "
              f"{r['title'][:70]}", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: nothing written", file=sys.stderr)
        return

    with papers_path.open("a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    print(f"\nappended {len(new)} -> {papers_path}\nnow run: python screen.py",
          file=sys.stderr)


if __name__ == "__main__":
    main()
