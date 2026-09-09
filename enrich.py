#!/usr/bin/env python3
"""Backfill citation counts (and missing year/venue) from Semantic Scholar.

    python enrich.py                      # papers.jsonl + screened.jsonl
    python enrich.py --title-match        # also resolve OpenReview papers

Most records arrive without citation counts: OpenReview and arXiv do not
carry them, and the ones OpenAlex supplies go stale. Anything ranking or
sorting by influence needs this run first.

Papers with an arXiv id or DOI resolve exactly, 500 per request. OpenReview
records have neither, so they need a fuzzy title match at one request each —
opt in with --title-match.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import requests

from snowball import S2Client, norm_title

BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "title,year,venue,citationCount,influentialCitationCount"


def arxiv_from_doi(doi: str):
    """arXiv's DataCite DOIs (10.48550/arXiv.NNNN) are not indexed by S2 as
    DOIs, but the same paper is indexed by its arXiv id. Convert rather than
    lose the lookup."""
    m = re.match(r"10\.48550/arxiv\.(.+)$", doi or "", re.I)
    return m.group(1) if m else None


def s2_id(pid: str):
    """Our id -> an id Semantic Scholar's batch endpoint understands."""
    kind, _, rest = pid.partition(":")
    if kind == "arxiv":
        return "ARXIV:" + rest.split("v")[0]
    if kind == "doi":
        arx = arxiv_from_doi(rest)
        return f"ARXIV:{arx}" if arx else "DOI:" + rest
    if kind == "s2":
        return rest
    return None                      # openreview / openalex -> title match


def openalex_match(session, title, key=None):
    """Resolve one title against OpenAlex.

    S2's /paper/search/match is throttled to roughly one request per minute
    even with a key, which makes it useless for more than a handful. OpenAlex
    serves the same citation counts on a budget you can actually see.
    """
    # commas and pipes are filter syntax in OpenAlex; strip rather than escape
    q = re.sub(r"[,|]", " ", title).strip()
    params = {"filter": f"title.search:{q}", "per-page": 3,
              "select": "display_name,publication_year,cited_by_count,primary_location"}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = session.get("https://api.openalex.org/works", params=params,
                        headers=headers, timeout=45)
    except requests.RequestException:
        return None
    if not r.ok:
        return None
    for w in r.json().get("results", []):
        if norm_title(w.get("display_name")) == norm_title(title):
            src = (w.get("primary_location") or {}).get("source") or {}
            return {"title": w.get("display_name"),
                    "year": w.get("publication_year"),
                    "venue": src.get("display_name") or "",
                    "citationCount": w.get("cited_by_count"),
                    "influentialCitationCount": None}
    return None


def fetch_batch(ids, key, max_retries=4):
    headers = {"User-Agent": "lit-screen/1.0"}
    if key:
        headers["x-api-key"] = key
    for attempt in range(max_retries):
        r = requests.post(BATCH, params={"fields": FIELDS},
                          json={"ids": ids}, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"  {r.status_code}, waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        print(f"  ! HTTP {r.status_code}: {r.text[:120]}", file=sys.stderr)
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*",
                    default=["papers.jsonl", "screened.jsonl"])
    ap.add_argument("--title-match", action="store_true",
                    help="also resolve records with no arXiv id or DOI "
                         "(OpenReview) by fuzzy title, 1 request each")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip records that already have a count")
    ap.add_argument("--decisions",
                    help="only enrich these decisions, e.g. 'include' or "
                         "'include,maybe'. Skips work you will never look at")
    ap.add_argument("--title-source", default="openalex",
                    choices=["openalex", "s2"],
                    help="s2's match endpoint is ~1 request/minute; openalex "
                         "is the practical choice")
    ap.add_argument("--cache", default=".enrich-cache.json",
                    help="resolved lookups, so an interrupted run resumes")
    ap.add_argument("--rps", type=float, default=5.0)
    args = ap.parse_args()

    files = [Path(f) for f in args.files if Path(f).exists()]
    if not files:
        sys.exit("no input files found")

    # one lookup per unique id, then apply to every file that mentions it
    records = {}
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                prev = records.get(p["id"])
                # papers.jsonl is read first and has no decisions; prefer the
                # screened copy so --decisions has something to filter on
                if prev is None or ("decision" in p and "decision" not in prev):
                    records[p["id"]] = p
    todo = [p for p in records.values()
            if not (args.only_missing and p.get("citation_count") is not None)]
    if args.decisions:
        want = {d.strip() for d in args.decisions.split(",")}
        before = len(todo)
        todo = [p for p in todo if p.get("decision") in want]
        print(f"  --decisions {args.decisions}: {before} -> {len(todo)}",
              file=sys.stderr)
    print(f"{len(records)} unique papers across {len(files)} files; "
          f"{len(todo)} to look up", file=sys.stderr)

    key = os.environ.get("S2_API_KEY")
    if not key:
        print("  (no S2_API_KEY — expect 429s)", file=sys.stderr)

    cache_path = Path(args.cache)
    resolved = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if resolved:
        print(f"  {len(resolved)} already resolved (cache)", file=sys.stderr)
    todo = [p for p in todo if p["id"] not in resolved]
    by_id = {}
    for p in todo:
        sid = s2_id(p["id"])
        if sid:
            by_id.setdefault(sid, []).append(p["id"])
    ids = list(by_id)
    print(f"  {len(ids)} resolvable by arXiv id / DOI", file=sys.stderr)

    for i in range(0, len(ids), 500):  # batch endpoint: fast, barely throttled
        chunk = ids[i:i + 500]
        data = fetch_batch(chunk, key)
        if data:
            for sid, row in zip(chunk, data):
                if row:
                    for ours in by_id[sid]:
                        resolved[ours] = row
        print(f"  {min(i+500, len(ids))}/{len(ids)}", file=sys.stderr)

    if args.title_match:
        rest = [p for p in todo if not s2_id(p["id"])]
        print(f"  {len(rest)} by title match ({args.title_source})",
              file=sys.stderr)
        session = requests.Session()
        session.headers["User-Agent"] = "lit-screen/1.0"
        oa_key = os.environ.get("OPENALEX_API_KEY")
        client = S2Client(rps=args.rps) if args.title_source == "s2" else None
        for n, p in enumerate(rest, 1):
            if args.title_source == "openalex":
                m = openalex_match(session, p["title"], oa_key) or {}
                time.sleep(0.15)
            else:
                d = client.get("/paper/search/match", query=p["title"],
                               fields=FIELDS)
                m = (d or {}).get("data", [{}])
                m = m[0] if m else {}
                if m.get("title") and norm_title(m["title"]) != norm_title(p["title"]):
                    m = {}                # fuzzy match, not the same paper
            if m:
                resolved[p["id"]] = m
            if n % 10 == 0:                       # checkpoint: this endpoint is
                cache_path.write_text(json.dumps(resolved))   # slow, don't lose it
                print(f"  {n}/{len(rest)}", file=sys.stderr, flush=True)
        cache_path.write_text(json.dumps(resolved))

    # write back
    filled = 0
    for f in files:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for r in rows:
            m = resolved.get(r["id"])
            if not m:
                continue
            if m.get("citationCount") is not None:
                if r.get("citation_count") != m["citationCount"]:
                    filled += 1
                r["citation_count"] = m["citationCount"]
            r["influential_citations"] = m.get("influentialCitationCount")
            if not r.get("year") and m.get("year"):
                r["year"] = m["year"]
            if not r.get("venue") and m.get("venue"):
                r["venue"] = m["venue"]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        have = sum(1 for r in rows if r.get("citation_count") is not None)
        print(f"  {f}: {have}/{len(rows)} now have counts", file=sys.stderr)
    print(f"resolved {len(resolved)} papers, {filled} counts written",
          file=sys.stderr)


if __name__ == "__main__":
    main()
