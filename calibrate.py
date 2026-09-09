#!/usr/bin/env python3
"""Check the screener against papers you labelled yourself.

    # 1. make a sample to label by hand (50 is enough)
    python calibrate.py sample --papers papers.jsonl --n 50 --out gold.csv

    # 2. open gold.csv, fill the 'my_label' column with include/exclude
    #    (do this BEFORE looking at screened.csv, or you'll anchor on it)

    # 3. score
    python calibrate.py score --gold gold.csv --screened screened.jsonl

Recall is the number that matters. Precision just costs you reading time;
recall costs you papers you will never know existed. If recall on includes
is below ~0.95, fix criteria.md and rerun rather than trusting the output.
"""
import argparse
import collections
import csv
import json
import random
from pathlib import Path


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def cmd_sample(args):
    random.seed(args.seed)
    if args.stratified:
        # Uniform sampling breaks down once the corpus is mostly irrelevant:
        # at ~2% prevalence, 50 random papers contain ~1 relevant one and
        # recall computed on it means nothing. Draw across the screener's own
        # decisions instead — the exclude stratum is where misses hide.
        screened = load_jsonl(args.screened)
        by_decision = {}
        for r in screened:
            by_decision.setdefault(r["decision"], []).append(r)
        wanted = {}
        for part in args.strata.split(","):
            k, _, v = part.partition("=")
            wanted[k.strip()] = int(v)
        sample = []
        for decision, k in wanted.items():
            pool = by_decision.get(decision, [])
            if len(pool) < k:
                print(f"  only {len(pool)} {decision}s available (wanted {k})")
            sample += random.sample(pool, min(k, len(pool)))
        random.shuffle(sample)   # don't hand them over grouped by verdict
        print(f"stratified over {len(screened)} screened papers: "
              + ", ".join(f"{d}={len(by_decision.get(d, []))}" for d in wanted))
    else:
        papers = load_jsonl(args.papers)
        sample = random.sample(papers, min(args.n, len(papers)))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "my_label", "title", "abstract"])
        for p in sample:
            # match screen.py's cutoff: labelling on less text than the model
            # saw turns a fair comparison into a handicap
            w.writerow([p["id"], "", p["title"], (p.get("abstract") or "")[:2500]])
    print(f"wrote {len(sample)} papers to {args.out} — fill in my_label "
          f"(include / maybe / exclude) by hand, then run: calibrate.py score")


def cmd_score(args):
    gold = {}
    with open(args.gold) as f:
        for row in csv.DictReader(f):
            raw = (row.get("my_label") or "").strip().lower()
            # people annotate their labels; take the verdict, keep the note out
            lab = next((d for d in ("include", "maybe", "exclude") if d in raw), "")
            if lab:
                gold[row["id"]] = lab
            elif raw:
                print(f"  ? unreadable label {raw[:40]!r} — skipping {row['id']}")
    if not gold:
        print("no labels found in my_label column")
        return

    pred = {}
    for line in Path(args.screened).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            pred[r["id"]] = r

    # Treat include+maybe as "kept" — that's what you'll actually read.
    def kept(label):
        return label in ("include", "maybe")

    # Stratum populations, so a stratified gold set can be reweighted back to
    # what the whole corpus looks like. For a uniform sample the weights all
    # come out equal and this is a no-op.
    pop = collections.Counter(r["decision"] for r in pred.values())
    drawn = collections.Counter(pred[i]["decision"] for i in gold if i in pred)
    weight = {d: (pop[d] / drawn[d]) if drawn.get(d) else 0.0 for d in pop}

    tp = fp = fn = tn = 0
    wtp = wfn = 0.0
    misses = []
    scored = 0
    for pid, g in gold.items():
        if pid not in pred:
            continue
        scored += 1
        p = pred[pid]["decision"]
        w = weight.get(p, 1.0)
        if kept(g) and kept(p):
            tp += 1; wtp += w
        elif kept(g) and not kept(p):
            fn += 1; wfn += w
            misses.append((pred[pid], g))
        elif not kept(g) and kept(p):
            fp += 1
        else:
            tn += 1

    if not scored:
        print("none of the labelled papers appear in screened.jsonl — "
              "run screen.py over the full set first")
        return

    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    print(f"scored {scored} labelled papers")
    print(f"  recall    {recall:.2f}   ({fn} of yours were dropped)")
    print(f"  precision {precision:.2f}   ({fp} extra papers to skim)")
    if len(set(round(w, 6) for w in weight.values() if w)) > 1:
        wr = wtp / (wtp + wfn) if wtp + wfn else float("nan")
        print(f"  corpus-weighted recall {wr:.2f}   <- the one to trust on a "
              f"stratified sample")
        print("  (strata: " + ", ".join(f"{d} {drawn[d]}/{pop[d]}" for d in pop
                                        if drawn.get(d)) + ")")
    if misses:
        print("\nMISSED — these are what to fix in criteria.md:")
        for p, g in misses:
            print(f"  [you: {g} | it: {p['decision']}] {p['title'][:80]}")
            print(f"      its reason: {p.get('reason', '')[:110]}")
    else:
        print("\nno misses in this sample. label another 50 from a different "
              "seed before trusting it on the full set.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--papers", default="papers.jsonl")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="gold.csv")
    s.add_argument("--stratified", action="store_true",
                   help="draw across the screener's decisions instead of "
                        "uniformly (use once screen.py has run)")
    s.add_argument("--screened", default="screened.jsonl")
    s.add_argument("--strata", default="exclude=20,maybe=15,include=15")
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("score")
    c.add_argument("--gold", default="gold.csv")
    c.add_argument("--screened", default="screened.jsonl")
    c.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
