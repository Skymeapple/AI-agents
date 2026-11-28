# Building an Agent for Preserving Knowledge from Research Papers

- **Field:** agent-harnesses
- **HFM track:** `discovery_live_live_harness_paper_memory` (paused on HFM)
- **Project status:** Tested on public data (model-free floor; HFM's own task still needs a language-model key)

## Readiness

- **State:** A model-free floor has run. The agent itself is ready to run once a language-model key is connected.
- **Data access:** QASPER on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** CPU for the harness; the model runs behind an API or on a GPU.
- **Still missing:** A language-model key, or a free GPU account to host an open-weight model. GitHub Models did not accept the workflow's built-in token.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

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

**What was run.** [`note_floor.py`](note_floor.py) builds 3,000-character notes without a language model, by selecting sentences, and measures how much of the material that questions are answered from is still in the note. This is a floor on the note's information content, not an entry in the track: HFM's score is the answer F1 of a frozen model reading the note, which needs a paid key.

- **Data:** QASPER (Dasigi et al. 2021, CC-BY 4.0): NLP papers with questions, answers and the evidence paragraphs annotators marked. Papers outside HFM's 4,000 to 86,000 character range are dropped, leaving 818 training, 274 validation and 406 test papers (median length about 22,000 characters). Only answerable questions with text evidence are scored (1,308 on test).
- **Note builders:** the paper's opening; the abstract plus each paragraph's first sentence; sentences nearest the paper's TF-IDF centroid; sentences with numbers first; and a logistic regression trained on the training papers to predict which sentences are answer evidence, with and without word features; and gradient-boosted trees that score whole paragraphs from position, section type, numbers, a short fixed list of cue words and similarity to the title and abstract (which is all the QASPER question writers saw), then pack whole paragraphs. An oracle that knows the questions packs gold evidence first and gives the ceiling for the budget.
- **Scores:** *evidence share* is the share of a question's gold evidence sentences (by characters) that are in the note. *Span kept* is the share of extractive questions whose whole answer span is in the note. *Evidence kept* is lenient: one sentence of an evidence paragraph is enough.
- **Selection:** the builder was chosen on validation by evidence share and reported on test.
- **Compute:** a free GitHub Actions runner, about two minutes. Log: [`../../../compute/results/paper-note-floor/run.log`](../../../compute/results/paper-note-floor/run.log).

**Results (test papers, 3,000 characters).**

| Note builder | Evidence share | Span kept | Evidence kept (lenient) |
|---|---|---|---|
| Opening of the paper | 0.093 | 0.130 | 0.141 |
| Abstract + paragraph first sentences | 0.140 | 0.224 | 0.528 |
| Central sentences (TF-IDF) | 0.203 | 0.234 | 0.615 |
| Numbers first | 0.131 | 0.193 | 0.338 |
| Learned, with word features | 0.347 | 0.398 | 0.781 |
| Learned, position and section type only | 0.379 | 0.375 | 0.569 |
| **Whole paragraphs, boosted trees** (chosen on validation) | **0.455** | 0.460 | 0.586 |
| Oracle (knows the questions) | 0.808 | 0.765 | 0.887 |

- **The budget is not the binding limit; not knowing the questions is.** With the questions known, 3,000 characters hold 81% of the evidence. Without them, the best builder holds 46%.
- **The opening of a paper is close to useless.** The abstract and introduction hold 9% of the evidence. No evidence paragraph is in the abstract, and only 5% of evidence sentences are in the first tenth of the paper.
- **Evidence sits in the data and experimental-setup sections.** The share of sentences that are evidence is 19% in data sections, 13% in setup, 12% in results, and 4% in related work.
- **For sentence selection, structure carries the signal; free word features do not add to it.** The sentence model with only position, section type and number features has a higher evidence share than the model that also sees words (0.379 against 0.347). The word model spreads its picks over more paragraphs, which raises the lenient score and lowers the strict one.
- **Selecting whole paragraphs beats selecting sentences.** Scoring and packing paragraphs raises evidence share from 0.379 to 0.455 and span kept from 0.375 to 0.460. Evidence is marked by paragraph, and an answer usually needs the sentences around it.
- **Doubling the budget helps less than knowing the questions.** At 6,000 characters the chosen builder holds 67% of the evidence; the oracle holds 81% at 3,000.

**Limits of this measurement.**

- Evidence being in the note is necessary for an answer, not sufficient; a reader model can also fail to use it. Conversely a model-written note can paraphrase and compress, which sentence selection cannot, so a language model can beat these numbers at the same budget.
- QASPER questions were written from the title and abstract only, which is part of why the abstract holds no evidence. HFM's questions are undisclosed and may be distributed differently.
- The scores are not comparable with HFM's answer F1 of 0.2405 → 0.4794.

**What it means for the track.** A note-writing agent should spend its budget on datasets, experimental settings and results, and little on the abstract, introduction and related work. The large gap to the oracle says the gain to look for is in predicting the questions, for example by generating likely questions per paper and keeping what answers them.

## How far this can go

Bounded by the frozen reader's accuracy with the full paper in view. That number is not published.

## Next steps

Measure the full-paper ceiling on QASPER with any available model.
