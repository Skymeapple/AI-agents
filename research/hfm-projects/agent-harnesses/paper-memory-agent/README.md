# Building an Agent for Preserving Knowledge from Research Papers

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_paper_memory` (paused on HFM)
- **Project status:** Blocked

## The challenge

This work describes the development of a reusable system that converts NLP research papers, ranging from 4,000 to 86,000 characters in length, into concise notes of at most 3,000 characters. The system uses a frozen model with an 8,000-character input limit per call, and the notes are intended for a later reader who uses the same frozen model with an undisclosed fixed prompt.

- **Metric:** `answer_f1` (maximize)
- **HFM baseline:** 0.2405
- **HFM best so far:** 0.4794
- **HFM effort:** 71 proposals, 65 scored, 8 accepted improvements, about $1,303 of compute
- **HFM's current best idea:** Allowing qualitative extra facts in p836's final consolidation raises practice answer_f1 over its numerically focused consolidation prompt.

## Existing theory and methods

Query-focused summarisation, retrieval with BM25, structured fact extraction, question anticipation.

## Public data

The task uses NLP papers; public equivalents are QASPER and the ACL Anthology.

## What limits progress

A 3,000-character note must answer unknown questions; what to keep is a guess.

## Route to a step change

Write notes as dense fact tables in the reader's likely answer format, built by anticipating questions.

## Experiment and results

No experiment has been run for this project.

- **Why:** Needs paid access to a language-model API to run the frozen model.

## How far this can go

Bounded by the frozen reader's accuracy with the full paper in view. That number is not published.

## Next steps

Measure the full-paper ceiling on QASPER with any available model.
