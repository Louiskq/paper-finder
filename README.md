# paper-finder

Turn a keyword search that returns 20,000 papers into a ranked reading list of
200, without reading 20,000 abstracts.

```
fetch → screen → calibrate → enrich → rank → snowball ⤴
```

## Quick start

```bash
pip install -r requirements.txt
cp criteria.example.md criteria.md      # then rewrite it for your question
echo 'GEMINI_API_KEY=...' > .env        # or ANTHROPIC_API_KEY
set -a; . .env; set +a

python fetch.py --openalex-query '"low-rank adaptation" OR "parameter-efficient"'
python screen.py --dry-run              # cost estimate, calls nothing
python screen.py                        # ~$1-2 for 4,000 abstracts
python rank.py                          # -> ranked.csv, read from the top
```

## The scripts

| | |
|---|---|
| `fetch.py` | Pull candidates from OpenAlex, arXiv and OpenReview into `papers.jsonl` |
| `screen.py` | Judge each abstract against `criteria.md` → include / maybe / exclude |
| `calibrate.py` | Score the screener against labels you write by hand |
| `enrich.py` | Backfill citation counts from Semantic Scholar |
| `rank.py` | Order the includes by reading priority |
| `snowball.py` | Walk the citation graph outward from your includes |

Every stage appends and resumes. Interrupt anything, rerun it, nothing is
refetched or rescreened.

## criteria.md is the whole product

Output quality is entirely a function of how precisely this file describes
what you want. Two rules:

**Write include rules as observable properties.** "Reports a head-to-head
comparison" can be tested against an abstract. "Is relevant to my work"
cannot.

**Define your load-bearing terms.** The screener applies them literally, and a
vague word is one it will read generously.

## Sources

| | reaches | lacks |
|---|---|---|
| OpenAlex | every venue, every year | accept/reject; ~25% have no abstract |
| arXiv | preprints, months early | anything never posted as one |
| OpenReview | ICLR/NeurIPS/ICML 2023+, incl. rejected | anything older |

ICLR 2026 alone is ~20,000 submissions, so `--require` filters locally.
Alternation is OR, repeating the flag is AND:

```bash
python fetch.py --venues venues.txt \
  --require 'reinforcement learning|\bRL\b' \
  --require 'hyper.?parameter|learning rate|schedul'
```

It applies to every stream, so run differently-filtered queries as separate
passes. Rejected submissions are dropped unless you pass `--include-rejected`.

## Calibrate

```bash
python calibrate.py sample --stratified --n 50
# label gold.csv by hand, WITHOUT looking at screened.csv
python calibrate.py score
```

Use `--stratified` whenever most of the corpus is irrelevant. A uniform sample
of 50 from a 5%-relevant corpus contains two relevant papers, and recall
computed on two papers isn't a measurement.

Recall is the number that matters. Below ~0.95, the misses print with the
reason the screener gave, which usually names the criterion that was too
narrow. Then run it again with `--seed 1` — a rubric tuned on one sample of 50
has been tuned to that sample.

## Snowball

```bash
python snowball.py --dry-run --dump reached.jsonl
python snowball.py --min-links 2 --require '...'
python screen.py                     # screens only the new arrivals
```

Follows citations instead of keywords: backwards for the canon your phrases
miss, forwards for recent work using different vocabulary. `seed_links` counts
how many of your includes each paper connects to; sort by it.

With many seeds the top of that list fills with whatever the whole field cites,
so `--require` matters here too. Always pass `--dump` — the crawl is the
expensive part and shouldn't be repeated to retune a threshold.
