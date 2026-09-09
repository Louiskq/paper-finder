#!/usr/bin/env python3
"""Rank the includes by how directly they bear on the research question.

    python rank.py                    # reads screened.jsonl -> ranked.csv

Screening asks "is this in scope?". This asks "what should I read first?",
which is a different question: a paper can be squarely in scope and still be
background rather than evidence. Scores are 1-10 against the research
question in criteria.md, tie-broken by citations.
"""
import argparse, csv, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from screen import PROVIDERS, pick_provider

SYSTEM = """You are triaging a reading list for a specific paper. You get the \
author's research question, then a batch of papers they have already decided \
are in scope.

Score each 1-10 for READING PRIORITY — how directly it bears on the question:

10-9  Direct evidence for or against the question as stated. The paper
      measures the thing the question asks about. Read first.
8-6   Strong method or analysis the author must engage with: proposes or
      tests the mechanism, or a study they would have to cite as prior art.
5-3   Context. Useful background, plausible baseline, or related work, but
      not evidence about the question itself.
2-1   In scope but peripheral. Cite if needed, do not spend a morning on it.

Judge only on the title and abstract. Do not reward a paper for being famous \
or recent. `why` is ONE clause saying what the author would get from reading \
it — not a summary of the paper.

Return ONLY a JSON array, in the order given:
[{"n": 1, "score": 8, "why": "..."}, ...]"""

SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "n": {"type": "INTEGER"},
    "score": {"type": "INTEGER"},
    "why": {"type": "STRING"}},
    "required": ["n", "score", "why"]}}


def research_question(criteria: str) -> str:
    m = re.search(r"## Research question\n(.*?)\n## ", criteria, re.S)
    return m.group(1).strip() if m else criteria


def rank_batch(client, model, question, batch):
    from google.genai import types
    prompt = "\n\n".join(
        f"[{i}] TITLE: {p['title']}\nABSTRACT: {(p.get('abstract') or '')[:2000]}"
        for i, p in enumerate(batch, 1))
    for attempt in range(4):
        try:
            r = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=f"{SYSTEM}\n\n<research_question>\n"
                                       f"{question}\n</research_question>",
                    response_mime_type="application/json", response_schema=SCHEMA,
                    temperature=0.0, max_output_tokens=4000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)))
            rows = json.loads(r.text)
            out = []
            for i, p in enumerate(batch, 1):
                row = next((x for x in rows if x.get("n") == i), None)
                if row is None and len(rows) == len(batch):
                    row = rows[i - 1]
                out.append({**p,
                            "score": int(row.get("score", 0)) if row else 0,
                            "why": (row or {}).get("why", "RANK FAILED")})
            return out
        except Exception as e:
            if attempt == 3:
                return [{**p, "score": 0, "why": f"RANK FAILED: {e}"} for p in batch]
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screened", default="screened.jsonl")
    ap.add_argument("--criteria", default="criteria.md")
    ap.add_argument("--decisions", default="include")
    ap.add_argument("--out", default="ranked.csv")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    want = {d.strip() for d in args.decisions.split(",")}
    papers = [json.loads(l) for l in Path(args.screened).read_text().splitlines()
              if l.strip()]
    papers = [p for p in papers if p["decision"] in want]
    question = research_question(Path(args.criteria).read_text())
    have = sum(1 for p in papers if p.get("citation_count") is not None)
    print(f"ranking {len(papers)} papers", file=sys.stderr)
    if have < 0.8 * len(papers):
        print(f"  ! only {have}/{len(papers)} have citation counts — ties will "
              f"break on year instead. Run enrich.py first.", file=sys.stderr)

    provider = pick_provider("auto")
    if provider != "gemini":
        sys.exit("rank.py currently implements the gemini path only")
    from google import genai
    client = genai.Client(api_key=os.environ[PROVIDERS[provider]["key"]])
    model = PROVIDERS[provider]["model"]

    batches = [papers[i:i + args.batch_size]
               for i in range(0, len(papers), args.batch_size)]
    results, lock, done = [], threading.Lock(), {"n": 0}

    def run(b):
        rows = rank_batch(client, model, question, b)
        with lock:
            results.extend(rows)
            done["n"] += 1
            print(f"  {done['n']}/{len(batches)}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, batches))

    # citations break ties; a paper nobody cites is not the one to read first
    results.sort(key=lambda r: (-r["score"], -(r.get("citation_count") or 0),
                                -(r.get("year") or 0)))
    for i, r in enumerate(results, 1):
        r["rank"] = i
        r["tier"] = ("1 read first" if r["score"] >= 9 else
                     "2 core" if r["score"] >= 6 else
                     "3 context" if r["score"] >= 3 else "4 peripheral")
    cols = ["rank", "tier", "score", "why", "topic", "title", "year", "venue",
            "citation_count", "url", "id"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    import collections
    print(f"\n{dict(collections.Counter(r['tier'] for r in results))}  -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
