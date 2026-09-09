# Screening criteria

<!-- THIS FILE IS THE ONLY PART THAT MATTERS. Everything else is plumbing.
     Copy it to criteria.md, rewrite it for your question, run calibrate.py,
     look at what it missed, rewrite it again. Two or three rounds of that is
     the difference between a screener you can trust and one that quietly
     drops the paper your reviewer asks about.

     What follows is a worked example on an unrelated topic, to show the
     shape. Replace all of it. -->

## Research question

When does parameter-efficient fine-tuning (LoRA, adapters, prefix tuning)
match full fine-tuning, and what property of the task or model predicts the
gap?

<!-- One or two sentences, and be specific about the *mechanism*, not the
     topic area. "PEFT methods" is a topic. "What predicts the gap between
     PEFT and full fine-tuning" is a mechanism. The screener can test the
     second against an abstract; it cannot test the first. -->

Define your load-bearing terms. The screener applies them literally, so a
word you leave vague is a word it will interpret generously:

- "parameter-efficient" means fewer than 5% of weights updated
- "matches" means within one standard deviation on the paper's own metric
- "task property" means anything measurable before fine-tuning: domain
  distance, dataset size, label noise, sequence length

Being *about* PEFT is not on its own enough to include — see below.

## INCLUDE if the paper does any of these

<!-- Write these as observable properties: "reports X", "compares Y".
     Not "is interesting" or "is relevant" — those are not testable
     against an abstract. -->

- reports a head-to-head comparison of a PEFT method against full
  fine-tuning on the same task and model
- identifies a task or model property that predicts when the gap widens
- measures how the gap changes with model scale, data size, or rank
- proposes a PEFT method AND reports where it fails, not only where it wins
- analyses *why* low-rank updates suffice, or when they do not

## MAYBE if

<!-- Deliberately generous. A maybe costs 30 seconds of skimming; a wrong
     exclude costs you a section you will never know was missing. -->

- proposes a PEFT method with no full fine-tuning baseline
- studies the same question in a different modality (vision, speech)
- surveys or benchmarks PEFT methods
- adjacent mechanism under other vocabulary: sparse fine-tuning, model
  merging, distillation with frozen backbones
- theory of low-rank structure in trained networks, with no fine-tuning
  experiments

## EXCLUDE if

- uses a PEFT method as a tool for an unrelated result, with no comparison
  or analysis of the method itself
- pure application paper with no methodological content
- efficiency claims about inference or serving rather than training
- clearly a different field that matched on keyword collision ("adapter"
  in hardware, "LoRA" the radio protocol)

Do NOT exclude a paper merely because it is not framed as a PEFT paper. If
it reports the comparison, it counts.

## Topic tags

<!-- Used to cluster the shortlist so your related-work section writes
     itself. Keep it to 5-8. -->

- `head-to-head`        <!-- direct PEFT vs full comparisons -->
- `predicts-the-gap`    <!-- task or model properties that explain it -->
- `scaling`             <!-- how the gap moves with size or rank -->
- `method`              <!-- new PEFT methods -->
- `theory`              <!-- why low-rank works -->
- `other-modality`
- `benchmark-or-survey`
- `other`
