# Literature screening pipeline

Fetch → screen → calibrate → enrich → rank → snowball. No framework, six
scripts, one rubric you write yourself.

Built for the case where a keyword search returns twenty thousand papers,
two hundred of them matter, and you need to find those two hundred without
reading twenty thousand abstracts.

```bash
pip install -r requirements.txt
cp criteria.example.md criteria.md      # then rewrite it for your question
```

Credentials go in `.env` (gitignored), loaded with `set -a; . .env; set +a`:

```
GEMINI_API_KEY=...              # or ANTHROPIC_API_KEY — screening picks either
OPENALEX_API_KEY=...            # free, 10x the keyless budget
OPENREVIEW_USERNAME=you@example.com
OPENREVIEW_PASSWORD=...
S2_API_KEY=...                  # free, for snowballing
```

None are strictly required except a model key, but each one you skip costs
you a source. OpenReview credentials are not optional if you want OpenReview:
anonymous note reads return `403 ChallengeRequiredError`.

## 1. Write the criteria first

`criteria.md` is the whole product. The scripts are plumbing you will never
touch again; the quality of the output is entirely a function of how
precisely this file describes what you are looking for.

Two rules that matter more than they look:

**Write include rules as observable properties.** "Reports a head-to-head
comparison" can be tested against an abstract. "Is relevant to my work"
cannot.

**Define your load-bearing terms.** The screener applies them literally. A
word you leave vague is a word it will interpret generously, and you will
not find out until calibration.

## 2. Fetch

Three sources, each reaching what the others cannot.

```bash
# OpenAlex — every venue, every year. Your primary source.
python fetch.py --openalex-query '"low-rank adaptation" OR "parameter-efficient fine-tuning"' \
                --openalex-query 'adapter AND "fine-tuning" AND "language model"'

# arXiv — preprints, often months ahead of publication
python fetch.py --arxiv-query 'cat:cs.LG AND abs:"parameter-efficient"'

# OpenReview — recent venue submissions, plus accept/reject status
python fetch.py --venues venues.txt --require 'fine.?tun|adapter|low.rank'
```

| source | reaches | lacks |
|---|---|---|
| OpenAlex | all venues, all years | accept/reject; ~25% have no abstract |
| arXiv | preprints | anything never posted as one |
| OpenReview | ICLR/NeurIPS/ICML/RLC **2023+**, incl. rejected | anything older |

Re-running only appends, so add a source next week without refetching.
Dedup is on normalised title, so the arXiv preprint and the camera-ready
collapse to one record.

### Narrowing

A big venue is all of ML, not your topic — ICLR 2026 alone is ~20,000
submissions. `--require` filters locally on title, abstract and keywords.
Alternation inside one pattern is OR; repeating the flag is AND.

It applies to **every** stream, so run differently-filtered queries as
separate passes rather than compromising on one pattern. Small on-topic
venues are worth taking whole — filtering a specialist venue for its own
subject only loses you papers.

Rejected submissions are dropped by default; `--include-rejected` keeps
them, and because dedup only skips what was written, rerunning later with
the flag appends them without refetching.

## 3. Screen

```bash
python screen.py --dry-run          # cost estimate, calls nothing
python screen.py --limit 60         # try 60 first, read them, then commit
python screen.py                    # the rest
```

Batches 15 abstracts per call against `criteria.md`. Works with Gemini or
Anthropic — whichever key is present, or force it with `--provider`.
Gemini's path constrains the output with a response schema, so the model
cannot return a fourth category.

For ~4,000 abstracts expect **$1–2** and a few minutes. Interrupt it
whenever; results append after every batch and finished ids are skipped on
restart.

Papers with no abstract are never excluded on the title alone — they become
`maybe`, tagged `NO ABSTRACT`. Check those by hand.

## 3b. Enrich — citation counts

```bash
python enrich.py                                  # arXiv ids + DOIs, 500/request
python enrich.py --title-match --decisions include  # the rest, by title
```

Nothing upstream supplies citation counts reliably: OpenReview and arXiv
have none, and OpenAlex's go stale. Anything sorting by influence needs
this first. Resolved lookups are cached, so an interrupted run resumes.

`--decisions include` is worth using — there is no reason to enrich papers
you already excluded.

## 4. Calibrate — do not skip this

```bash
python calibrate.py sample --stratified --n 50   # writes gold.csv
# label gold.csv by hand, WITHOUT looking at screened.csv first
python calibrate.py score
```

**Use `--stratified` on any corpus where most papers are irrelevant.**
Uniform sampling breaks down at low prevalence: 50 random papers from a
4,000-paper corpus that is 5% relevant contains two relevant papers, and
recall computed on two papers is not a measurement. Stratified draws across
the screener's own verdicts — the exclude stratum is where misses hide —
and `score` reweights each stratum back to its true size.

Recall is the number that matters. Precision costs you skimming time;
recall costs you papers you will never know existed. Below ~0.95, the
misses print with the reason the screener gave, which usually names the
criterion that was too narrow.

Then do it once more with `--seed 1`. **A rubric tuned on one sample of 50
has been tuned to that sample** — scoring it against the papers you used to
diagnose it will show near-perfect recall and mean nothing.

## 5. Rank — what to read first

```bash
python rank.py                      # includes -> ranked.csv
```

Screening asks "is this in scope?". Ranking asks "is this evidence, or
background?" — a different question, and the one that decides your morning.
Papers are scored 1-10 against the research question, tie-broken by
citations, and bucketed into read-first / core / context / peripheral.

```
rank tier          score  cites  title
   1 1 read first     10     35  Revisiting X: when the optimum moves
   2 1 read first      9    436  An Empirical Model of Large-Batch Training
   9 2 core            8    930  Population Based Training of Neural Networks
```

Run `enrich.py` first or ties break on publication year, which quietly
favours recent work.

## 6. Snowball — where the older work comes from

```bash
python snowball.py --dry-run --dump reached.jsonl   # see what it would add
python snowball.py --min-links 2 --require '...'    # append to papers.jsonl
python screen.py                                    # screens only the new ones
```

Every search above finds papers by matching words. Snowball finds them by
citation: backwards to what your includes cite (the pre-2015 canon your
phrases miss), forwards to what cites them (recent work in different
vocabulary).

`seed_links` counts how many of your includes a paper connects to. Sort by
it. But note that with many seeds the top of that list fills with whatever
the whole field cites — the framework papers, the optimiser, the benchmark
suite — so `--require` matters here as much as in fetch.

**Always pass `--dump`.** The crawl is the expensive part, and a dump lets
you retune `--min-links` and the filters without re-crawling.

Two hops is usually where it stops paying and starts returning all of
deep learning.

## What this does not do

It reads titles and abstracts only. A paper that undersells its own
contribution will be excluded. Two mitigations: keep `maybe` generous, and
snowball from your confirmed includes rather than relying on keywords to
find everything.

Nothing here writes prose for you, and nothing here should. The `reason`
and `why` fields are routing signals, generated from the abstract alone.
Anything you cite should trace to a PDF you opened.

## Rate limits, briefly

The numbers that will bite you:

- **OpenAlex** — $1/day free with an account, ~10c without. A request is
  $0.001 and returns 200 records, so cost scales with queries, not papers.
- **Semantic Scholar** — `/paper/search/match` runs at roughly one request
  per minute even with a key. Never use it in bulk; resolve by arXiv id or
  DOI instead. The batch endpoint takes 500 ids at once and is fine.
- **arXiv DOIs** (`10.48550/arXiv.NNNN`) are not indexed by S2 as DOIs.
  Convert them to arXiv ids or the lookup silently fails.
