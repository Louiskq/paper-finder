#!/usr/bin/env python3
"""Pull candidate papers from OpenReview and arXiv into papers.jsonl.

Usage:
    python fetch.py --venues venues.txt --arxiv-query "..." --out papers.jsonl

Re-running is safe: existing papers.jsonl is loaded first and only new
records are appended, so you can add a venue later without refetching.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# dedup key
# --------------------------------------------------------------------------

def norm_title(t: str) -> str:
    """Loose title key: lowercase, alphanumeric only. Catches most duplicates
    between the arXiv preprint and the camera-ready version."""
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def matches(p, patterns) -> bool:
    """True if title+abstract+keywords match every --require pattern.

    A local filter applied after fetching: OpenReview has no topical query, so
    a venue like ICLR is all of ML and needs narrowing here. Use alternation
    inside one pattern for OR, repeat --require for AND.
    """
    if not patterns:
        return True
    hay = " ".join([
        p.get("title") or "",
        p.get("abstract") or "",
        " ".join(p.get("keywords") or []),
    ]).lower()
    return all(rx.search(hay) for rx in patterns)


# --------------------------------------------------------------------------
# OpenReview (API v2 — ICLR/NeurIPS/ICML 2023+, RLC all years)
# --------------------------------------------------------------------------

_or_client = None


def openreview_client():
    """Cached, logged-in client. OpenReview challenge-gates anonymous reads of
    the notes endpoint (403 ChallengeRequiredError), so credentials are
    required even though the data is public. A free account is enough.

    Login is one round-trip, so cache it rather than repeating it per venue.
    """
    global _or_client
    if _or_client is None:
        from openreview.api import OpenReviewClient

        user = os.environ.get("OPENREVIEW_USERNAME")
        pw = os.environ.get("OPENREVIEW_PASSWORD")
        if not (user and pw):
            raise RuntimeError(
                "OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD not set — anonymous "
                "note reads are challenge-gated. Put them in .env (gitignored)."
            )
        _or_client = OpenReviewClient(
            baseurl="https://api2.openreview.net", username=user, password=pw
        )
        print(f"logged in to OpenReview as {user}", file=sys.stderr)
    return _or_client


def fetch_openreview(venue_id: str):
    """Yield paper dicts for one venue, e.g. 'rl-conference.cc/RLC/2026/Conference'.

    Venue IDs are the last part of the OpenReview group URL. Find them at
    https://openreview.net/venues — e.g.
        ICLR.cc/2026/Conference
        NeurIPS.cc/2025/Conference
        rl-conference.cc/RLC/2026/Conference
    """
    client = openreview_client()

    # The submission invitation name varies by venue ("Submission", "Blind_Submission"),
    # so read it off the venue group rather than hardcoding.
    group = client.get_group(venue_id)
    sub_name = group.content.get("submission_name", {}).get("value", "Submission")

    # No year field on the note; it's in the venue string ("RLC 2025") and in
    # the venue id. Take the id — it's the one we control.
    m = re.search(r"/(20\d{2})/", venue_id) or re.search(r"\b(20\d{2})\b", venue_id)
    venue_year = int(m.group(1)) if m else None

    notes = client.get_all_notes(invitation=f"{venue_id}/-/{sub_name}")
    print(f"  {venue_id}: {len(notes)} submissions", file=sys.stderr)

    for n in notes:
        c = n.content

        def val(field, default=""):
            v = c.get(field)
            return v.get("value", default) if isinstance(v, dict) else default

        # The 'venue' field holds the outcome: "ICLR 2026 poster", "Submitted to ...".
        venue_str = val("venue")
        yield {
            "id": f"openreview:{n.id}",
            "source": "openreview",
            "venue_id": venue_id,
            "venue": venue_str,
            "accepted": "submitted to" not in venue_str.lower() and bool(venue_str),
            "title": val("title"),
            "abstract": val("abstract"),
            "keywords": val("keywords", []),
            "url": f"https://openreview.net/forum?id={n.id}",
            "pdf": f"https://openreview.net/pdf?id={n.id}",
            "year": venue_year,
        }


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------

def fetch_arxiv(query: str, max_results: int = 2000):
    """Yield paper dicts for an arXiv API query.

    Query syntax: http://info.arxiv.org/help/api/user-manual.html#query_details
    Example:
        cat:cs.LG AND (abs:"hyperparameter" AND abs:"reinforcement learning")
    """
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    n = 0
    for r in client.results(search):
        n += 1
        yield {
            "id": f"arxiv:{r.get_short_id()}",
            "source": "arxiv",
            "venue_id": None,
            "venue": r.journal_ref or "",
            "accepted": None,
            "title": r.title.replace("\n", " ").strip(),
            "abstract": r.summary.replace("\n", " ").strip(),
            "keywords": r.categories,
            "url": r.entry_id,
            "pdf": r.pdf_url,
            "year": r.published.year if r.published else None,
        }
    print(f"  arxiv '{query[:50]}...': {n} results", file=sys.stderr)


# --------------------------------------------------------------------------
# OpenAlex — the whole published record, not just what a venue puts online
# --------------------------------------------------------------------------

OPENALEX = "https://api.openalex.org/works"


def _abstract_from_index(inv):
    """OpenAlex ships abstracts as {word: [positions]}. Put it back in order."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def fetch_openalex(query: str, from_year: int = 2015, max_results: int = 3000,
                   mailto: str = None):
    """Yield paper dicts for one OpenAlex title+abstract search.

    `query` is passed to title_and_abstract.search, which accepts AND / OR /
    NOT and quoted phrases:
        '"replay ratio" OR "update-to-data"'
        'hyperparameter AND "reinforcement learning"'

    Unlike OpenReview this reaches every venue and every year, so it is the
    only source here that sees ICML 2018 or NeurIPS 2020.

    `mailto` opts into OpenAlex's faster "polite pool" — your choice whether
    to hand them an address, so it is off unless you pass --mailto.
    """
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "lit-screen/1.0"
    # Keyless gets ~1/10th the daily budget of a free account. The key is free
    # and takes 30s to make: https://openalex.org/settings/api
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        session.headers["Authorization"] = f"Bearer {key}"
    else:
        print("  (no OPENALEX_API_KEY — running on the small keyless budget)",
              file=sys.stderr)
    select = ("id,doi,display_name,publication_year,abstract_inverted_index,"
              "primary_location,best_oa_location,cited_by_count")
    cursor, n = "*", 0

    while cursor and n < max_results:
        params = {
            "filter": f"title_and_abstract.search:{query},"
                      f"from_publication_date:{from_year}-01-01,"
                      f"type:article|preprint",
            "select": select,
            "per-page": min(200, max_results - n),
            "cursor": cursor,
        }
        if mailto:
            params["mailto"] = mailto
        r = session.get(OPENALEX, params=params, timeout=60)
        if not r.ok:
            # Loud: a truncated query otherwise looks exactly like a complete
            # one, and you calibrate against a corpus with a hole in it.
            print(f"  !! openalex HTTP {r.status_code} after {n} results — "
                  f"QUERY INCOMPLETE: {query[:50]}\n     {r.text[:120]}",
                  file=sys.stderr)
            return
        data = r.json()
        for w in data.get("results", []):
            n += 1
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            oa_id = (w.get("id") or "").rsplit("/", 1)[-1]
            src = (w.get("primary_location") or {}).get("source") or {}
            yield {
                "id": f"doi:{doi}" if doi else f"openalex:{oa_id}",
                "source": "openalex",
                "venue_id": None,
                "venue": src.get("display_name") or "",
                "accepted": None,
                "title": (w.get("display_name") or "").replace("\n", " ").strip(),
                "abstract": _abstract_from_index(w.get("abstract_inverted_index")),
                "keywords": [],
                "url": w.get("doi") or w.get("id"),
                "pdf": (w.get("best_oa_location") or {}).get("pdf_url"),
                "year": w.get("publication_year"),
                "citation_count": w.get("cited_by_count"),
            }
        cursor = (data.get("meta") or {}).get("next_cursor")
        time.sleep(0.2)
    if n >= max_results and cursor:
        print(f"  !! openalex hit --openalex-max ({max_results}) with more to "
              f"fetch — QUERY INCOMPLETE: {query[:50]}", file=sys.stderr)
    print(f"  openalex '{query[:45]}...': {n} results", file=sys.stderr)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venues", help="file with one OpenReview venue id per line")
    ap.add_argument("--arxiv-query", action="append", default=[],
                    help="arXiv API query (repeatable)")
    ap.add_argument("--arxiv-max", type=int, default=2000)
    ap.add_argument("--openalex-query", action="append", default=[],
                    help="OpenAlex title+abstract search, supports AND/OR/NOT "
                         "and quoted phrases (repeatable)")
    ap.add_argument("--openalex-from-year", type=int, default=2015)
    ap.add_argument("--openalex-max", type=int, default=3000)
    ap.add_argument("--mailto", help="opt in to OpenAlex's faster polite pool")
    ap.add_argument("--out", default="papers.jsonl")
    ap.add_argument("--require", action="append", default=[], metavar="REGEX",
                    help="keep only papers whose title/abstract/keywords match "
                         "this (repeatable; all must match)")
    ap.add_argument("--include-rejected", action="store_true",
                    help="also keep submissions that were not accepted. Off by "
                         "default; rerun later with it to append them (dedup "
                         "means nothing is refetched)")
    args = ap.parse_args()

    out = Path(args.out)
    seen_ids, seen_titles = set(), set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                seen_ids.add(p["id"])
                seen_titles.add(norm_title(p["title"]))
        print(f"loaded {len(seen_ids)} existing papers from {out}", file=sys.stderr)

    patterns = [re.compile(r, re.I) for r in args.require]
    added = 0
    dropped = {"rejected": 0, "off-topic": 0}
    with out.open("a") as f:
        streams = []
        if args.venues:
            for line in Path(args.venues).read_text().splitlines():
                v = line.split("#", 1)[0].strip()   # split on LINES, not words:
                if v:                               # a comment is not five venues
                    streams.append(fetch_openreview(v))
        for q in args.arxiv_query:
            streams.append(fetch_arxiv(q, args.arxiv_max))
        for q in args.openalex_query:
            streams.append(fetch_openalex(q, args.openalex_from_year,
                                          args.openalex_max, args.mailto))

        for stream in streams:
            try:
                for p in stream:
                    key = norm_title(p["title"])
                    if p["id"] in seen_ids or (key and key in seen_titles):
                        continue
                    # accepted is None for arXiv — only drop an explicit False
                    if not args.include_rejected and p["accepted"] is False:
                        dropped["rejected"] += 1
                        continue
                    if not matches(p, patterns):
                        dropped["off-topic"] += 1
                        continue
                    # screen.py judges on abstracts; OpenAlex is missing ~25% of
                    # them. Keep but flag, so they get eyeballed not lost.
                    if not p.get("abstract"):
                        p["no_abstract"] = True
                    seen_ids.add(p["id"])
                    seen_titles.add(key)
                    f.write(json.dumps(p) + "\n")
                    f.flush()
                    added += 1
            except Exception as e:  # one bad venue shouldn't kill the run
                print(f"  ! stream failed: {type(e).__name__}: {e}", file=sys.stderr)

    for reason, n in dropped.items():
        if n:
            print(f"  dropped {n} ({reason})", file=sys.stderr)
    print(f"added {added} new papers -> {out} ({len(seen_ids)} total)", file=sys.stderr)


if __name__ == "__main__":
    main()
