# Pretraining architecture development on frozen public training pool

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_pretrain_proposals_20261110` (active on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. Anything near HFM's scale waits for a GPU.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops and trains pretraining architectures on a frozen public training pool, using prior research to improve generalization and four-domain public-validation BPB. Each proposal delivers its agent-selected final checkpoint for independent scoring.

- **Metric:** `negative_public_validation_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.7048
- **HFM effort:** 26 proposals, 20 scored, 0 accepted improvements, about $785 of compute
- **HFM's current best idea:** Training a function-preserving depth-26, FFN-3040 continuation of p1645 with 0.60/0.20/0.20 FineWeb-Edu/GitHub-code/FineMath document sampling will improve web and overall four-domain generalization relative to the same capacity expansion trained with balanced 0.50/0.25/0.25 sampling.

## Existing theory and methods

Function-preserving growth (Net2Net), data-mix scheduling.

## Public data

FineWeb-Edu, GitHub code and FineMath are public.

## What limits progress

Engineering reliability and throughput.

## Route to a step change

Scaling-law extrapolation from short runs to rank ideas.

## Experiment and results

A toy-scale study on CPU of the question behind HFM's best proposal: does sampling web, code and mathematics at 0.60/0.20/0.20 instead of 0.50/0.25/0.25 improve web and overall four-domain generalization? Scripts: [`corpora.py`](corpora.py) fetches 30 MB of training text and the next 4 MB of held-out documents from each corpus; [`mix_tiny.py`](mix_tiny.py) trains on a stated mix. Logs in `research/compute/results/mix-tiny-pool-*`. It cannot say what happens at HFM's scale and is not comparable to HFM's score.

- **Setup:** 4-layer byte-level model (1.9M parameters, width 192), 6,000 steps, 25M bytes seen, three seeds per mix. Corpora: FineWeb-Edu (web), codeparrot Python files (code), FineMath (mathematics); DCLM is held out as a fourth domain that is never trained on. Each training sequence comes from one corpus.

| Web / code / maths | Web | DCLM | Code | Maths | Mean of four |
|---|---|---|---|---|---|
| 0.50 / 0.25 / 0.25 | 2.069 | 2.287 | 1.880 | 2.378 | **2.153** |
| 1/3 each | 2.124 | 2.324 | 1.821 | 2.356 | 2.156 |
| 0.20 / 0.40 / 0.40 | 2.176 | 2.358 | 1.780 | 2.332 | 2.162 |
| 0.60 / 0.20 / 0.20 | 2.054 | 2.281 | 1.966 | 2.434 | 2.184 |
| 0.80 / 0.10 / 0.10 | 2.029 | 2.271 | 2.170 | 2.546 | 2.254 |
| web only | 2.011 | 2.314 | 5.436 | 3.036 | 3.199 |

(bits per byte, mean of three seeds; the seed spread of the mean is 0.003 to 0.017, larger for code and maths)

- **HFM's proposed shift helps web and hurts the four-domain mean.** Moving from 0.50/0.25/0.25 to 0.60/0.20/0.20 gains 0.015 on web and 0.006 on DCLM, and loses 0.086 on code and 0.056 on maths: the mean gets 0.031 worse, about twice the seed spread. The proposal's claim about web holds here; its claim about overall generalization does not.
- **The mean is flat across balanced mixes.** 0.50/0.25/0.25, equal thirds and 0.20/0.40/0.40 lie within 0.009 of one another, inside seed noise. The score is sensitive only when a domain is starved: code at 10% costs 0.29 on code, and no code at all costs 3.6.
- **Unseen-domain transfer follows web share.** DCLM, never trained on, tracks the web share closely (2.271 at 80% web, 2.358 at 20%), while the web-only model is worse on DCLM than the 80% mix. A little code and mathematics does not hurt general web text.

## How far this can go

Not estimable without a baseline. At toy scale the four-domain mean is flat over balanced mixes, so mix changes of the size HFM proposes are unlikely to move it much; starving any one domain moves it a lot.

## Next steps

Publish a reference score. With a GPU, repeat the 0.50/0.25/0.25 against 0.60/0.20/0.20 comparison at 100M+ parameters, reporting each domain and seed spread.
