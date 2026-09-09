#!/usr/bin/env python3
"""Screen papers.jsonl against criteria.md. Writes screened.jsonl + screened.csv.

    # whichever key is in .env decides the provider; --provider forces one
    python screen.py --papers papers.jsonl --criteria criteria.md

Safe to interrupt and rerun: already-screened ids are skipped.
Check --dry-run before spending anything.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DECISIONS = {"include", "maybe", "exclude"}

# model + list rates ($/MTok in, out) per provider. Rates are only used for the
# --dry-run estimate; check current pricing if the number matters to you.
PROVIDERS = {
    "anthropic": {"model": "claude-haiku-4-5", "key": "ANTHROPIC_API_KEY",
                  "rates": (1.0, 5.0)},
    # 3.8-flash promo rates, good through 2026-12-31; doubles on 2027-01-01
    "gemini":    {"model": "gemini-3.8-flash", "key": "GEMINI_API_KEY",
                  "rates": (0.75, 3.75)},
}

SYSTEM = """You are screening academic papers for a literature review. \
You will be given screening criteria, then a batch of paper abstracts.

For EACH paper, decide: include, maybe, or exclude.

Rules:
- Judge only on the title and abstract given. Do not use outside knowledge \
about the paper, and do not guess at content the abstract does not state.
- When genuinely torn between two decisions, choose the more inclusive one. \
A wrongly included paper costs the reviewer 30 seconds; a wrongly excluded \
paper is never seen again.
- `reason` must be one clause citing something the abstract actually says. \
If you cannot ground it in the abstract, that is a signal to say maybe.
- `topic` must be one of the tags listed in the criteria.

Return ONLY a JSON array, one object per paper, in the order given:
[{"n": 1, "decision": "include", "topic": "tag", "reason": "..."}, ...]
No prose, no markdown fences."""

# Gemini can be held to this; Anthropic gets it as prose in SYSTEM above.
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "n": {"type": "INTEGER"},
            "decision": {"type": "STRING", "enum": ["include", "maybe", "exclude"]},
            "topic": {"type": "STRING"},
            "reason": {"type": "STRING"},
        },
        "required": ["n", "decision", "topic", "reason"],
    },
}


def build_batch_prompt(batch):
    parts = []
    for i, p in enumerate(batch, 1):
        abstract = (p.get("abstract") or "").strip().replace("\n", " ")
        parts.append(f"[{i}] TITLE: {p['title']}\nABSTRACT: {abstract[:2500]}")
    return "\n\n".join(parts)


def parse_response(text, batch):
    """Tolerant parse: strip fences, find the array, match by position.
    Still used for Gemini — a schema guarantees shape, not that every `n`
    came back."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array in response: {text[:200]}")
    rows = json.loads(text[start:end + 1])

    out = []
    for i, p in enumerate(batch, 1):
        row = next((r for r in rows if r.get("n") == i), None)
        if row is None and len(rows) == len(batch):
            row = rows[i - 1]          # model dropped "n" but kept order
        if row is None:
            row = {"decision": "maybe", "topic": "other",
                   "reason": "PARSE FAILURE - review manually"}
        d = str(row.get("decision", "maybe")).lower().strip()
        # judged on the title alone, so don't let it exclude on a guess
        if p.get("no_abstract") and d == "exclude":
            d, row = "maybe", {**row, "reason": "NO ABSTRACT - title only, "
                               "check by hand: " + str(row.get("reason", ""))[:80]}
        out.append({
            **p,
            "decision": d if d in DECISIONS else "maybe",
            "topic": row.get("topic", "other"),
            "reason": row.get("reason", ""),
        })
    return out


# --------------------------------------------------------------------------
# providers: each exposes .call(criteria, batch) -> raw text, and .retryable
# --------------------------------------------------------------------------

class AnthropicBackend:
    def __init__(self, model):
        import anthropic
        self.anthropic = anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.retryable = (anthropic.RateLimitError, anthropic.APIStatusError)

    def call(self, criteria, batch):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=[
                {"type": "text", "text": SYSTEM},
                # Cached: the criteria block is identical on every call, so
                # after the first it bills at 10% of the input rate.
                {"type": "text", "text": f"<criteria>\n{criteria}\n</criteria>",
                 "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": build_batch_prompt(batch)}],
        )
        return resp.content[0].text


class GeminiBackend:
    def __init__(self, model):
        from google import genai
        from google.genai import errors, types
        self.types = types
        self.model = model
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.retryable = (errors.ClientError, errors.ServerError)

    def call(self, criteria, batch):
        resp = self.client.models.generate_content(
            model=self.model,
            contents=build_batch_prompt(batch),
            config=self.types.GenerateContentConfig(
                system_instruction=f"{SYSTEM}\n\n<criteria>\n{criteria}\n</criteria>",
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.0,
                max_output_tokens=8000,
                # fixed rubric, nothing to reason about; thinking just costs tokens
                thinking_config=self.types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text


def make_backend(name):
    return {"anthropic": AnthropicBackend,
            "gemini": GeminiBackend}[name](PROVIDERS[name]["model"])


def pick_provider(requested):
    """Explicit --provider wins; otherwise use whichever key is present."""
    if requested != "auto":
        if not os.environ.get(PROVIDERS[requested]["key"]):
            sys.exit(f"--provider {requested} but {PROVIDERS[requested]['key']} is not set")
        return requested
    available = [n for n, p in PROVIDERS.items() if os.environ.get(p["key"])]
    if not available:
        sys.exit("no API key found — set " +
                 " or ".join(p["key"] for p in PROVIDERS.values()) + " in .env")
    if len(available) > 1:
        print(f"multiple keys set, using {available[0]} (--provider to force)",
              file=sys.stderr)
    return available[0]


def screen_batch(backend, criteria, batch, max_retries=5):
    for attempt in range(max_retries):
        try:
            return parse_response(backend.call(criteria, batch), batch)
        except backend.retryable as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.random()
            print(f"  retry in {wait:.1f}s ({type(e).__name__})", file=sys.stderr)
            time.sleep(wait)
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                # Don't lose the batch: flag it for manual review instead.
                return [{**p, "decision": "maybe", "topic": "other",
                         "reason": f"SCREENING FAILED: {e}"} for p in batch]
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="papers.jsonl")
    ap.add_argument("--criteria", default="criteria.md")
    ap.add_argument("--out", default="screened.jsonl")
    ap.add_argument("--csv", default="screened.csv")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "anthropic", "gemini"])
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only screen the first N (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="estimate cost, call nothing")
    args = ap.parse_args()

    papers = [json.loads(l) for l in Path(args.papers).read_text().splitlines() if l.strip()]
    criteria = Path(args.criteria).read_text()

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [p for p in papers if p["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(papers)} papers, {len(done)} already screened, {len(todo)} to do",
          file=sys.stderr)

    provider = pick_provider(args.provider)
    model = PROVIDERS[provider]["model"]
    rate_in, rate_out = PROVIDERS[provider]["rates"]

    if args.dry_run:
        chars = sum(len(p.get("abstract") or "") + len(p["title"]) for p in todo)
        in_tok = chars / 4 + len(todo) / args.batch_size * len(criteria) / 4
        out_tok = len(todo) * 40
        print(f"~{in_tok/1e6:.2f}M input + {out_tok/1e6:.3f}M output tokens\n"
              f"~${in_tok/1e6*rate_in + out_tok/1e6*rate_out:.2f} at {model} "
              f"list rates (less with caching)", file=sys.stderr)
        return

    if not todo:
        print("nothing to do", file=sys.stderr)
    else:
        print(f"screening with {model}", file=sys.stderr)
        backend = make_backend(provider)
        batches = [todo[i:i + args.batch_size]
                   for i in range(0, len(todo), args.batch_size)]
        lock = threading.Lock()
        counter = {"n": 0}

        def run(batch):
            rows = screen_batch(backend, criteria, batch)
            with lock:                       # checkpoint after every batch
                with out_path.open("a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                counter["n"] += 1
                print(f"  {counter['n']}/{len(batches)} batches", file=sys.stderr)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(run, batches))

    # ---- CSV, sorted so the includes are at the top ----
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    order = {"include": 0, "maybe": 1, "exclude": 2}
    rows.sort(key=lambda r: (order.get(r["decision"], 3), r.get("topic", ""), r["title"]))
    cols = ["decision", "topic", "reason", "title", "venue", "year", "url", "id"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    counts = {d: sum(1 for r in rows if r["decision"] == d) for d in DECISIONS}
    print(f"\n{counts}  ->  {args.csv}", file=sys.stderr)
    print(f"read the {counts['include']} includes; skim the {counts['maybe']} maybes; "
          f"spot-check 20 random excludes", file=sys.stderr)


if __name__ == "__main__":
    main()
