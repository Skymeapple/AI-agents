# Looped Transformer Pretraining with Recurrent Depth Architectures

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_looped_pretrain_research_v1` (end on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. Anything near HFM's scale waits for a GPU.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work explores language models that reuse computation and refine internal state through looped or recurrent-depth architectures. The research campaign uses eight H20 GPUs over 168 hours to design and train a model from random initialization, selecting one final checkpoint. The investigation covers architectures including partial sharing, adapters, mixture of experts, hybrids, nested loops, learned exits, and dynamic computation.

- **Metric:** `negative_hidden_dclm_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -1.553
- **HFM effort:** 20 proposals, 2 scored, 0 accepted improvements, about $130 of compute
- **HFM's current best idea:** Continuing the p481 hybrid checkpoint with four-loop execution exposed immediately and a subsequent 1/2/4-loop balanced schedule will improve held-out causal language-model likelihood, because p481 stopped after 233 steps while its 6000-step curriculum was still training only the one-loop path.

## Existing theory and methods

Recurrent-depth models, learned exits.

## Public data

Public pretraining corpora.

## What limits progress

2 of 20 proposals produced a score.

## Route to a step change

Reliability before novelty.

## Experiment and results

A toy-scale study on CPU ([`loops_tiny.py`](loops_tiny.py); logs in `research/compute/results/loops-tiny-*`). It cannot say what happens at HFM's scale and is not comparable to HFM's score.

- **Setup:** byte-level model of 1.9M parameters (1 layer, 2 shared layers applied N times, 1 layer; width 192), 6,000 steps on TinyStories, two seeds per arm, same bytes in the same order. Score is bits per byte on held-out text; lower is better. The two seeds differ by 0.001 to 0.006.
- **Arms:** always 3 loops; 1, 2 or 4 loops drawn per step (HFM's proposed schedule; 22% less training compute); 2, 3 or 4 loops drawn per step (same training compute as fixed 3).

| Trained with | 1 loop | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| always 3 loops | 1.874 | 1.185 | **1.097** | 1.137 | 1.563 | 2.557 |
| 1, 2 or 4 loops | 1.176 | 1.148 | 1.145 | 1.150 | 1.176 | 1.219 |
| 2, 3 or 4 loops | 1.354 | 1.128 | 1.110 | 1.114 | 1.153 | 1.239 |

(mean of two seeds)

- **Mixing loop counts costs accuracy at the usual loop count.** At 3 loops the 2/3/4 mix is 0.013 bits per byte worse than fixed 3 at equal training compute, and the 1/2/4 mix is 0.048 worse. Both gaps are several times the seed spread.
- **No arm improves when given more loops than it trained with.** The fixed-3 model falls apart beyond 4 loops. The mixed models degrade slowly instead, and the 1/2/4 model is nearly flat from 1 to 6 loops: it has learned to do almost all its work in the first loop (1.176 at one loop, which is worse than the separate 4-layer model's 1.141 in the neighbouring study).
- **What mixing buys is tolerance, not quality.** A 1/2/4-trained model can run at 1 loop for 0.031 bits per byte more than at 3 loops; the fixed-3 model loses 0.78.
- **Leaving early on confidence does not pay here.** Exiting each token at the first loop count where the top probability passes a threshold never beats running a fixed number of loops at the same mean cost: for the fixed-3 model, a 0.99 threshold uses 3.4 loops for 1.139, worse than plain 3 loops.
- **The hindsight bound is large.** Picking each token's best loop count after seeing the answer gives 0.890 at 2.7 mean loops for the fixed-3 model. No rule can reach this, and part of it is just choosing the luckiest of four predictions, but it shows that loop counts disagree token by token and that top probability is a poor signal of which is right.
- **Related evidence:** the study in [`../looped-pretraining-fixed-compute/`](../looped-pretraining-fixed-compute/README.md) found that looping closes 63% of the gap between a 4-layer and an 8-layer model at equal parameters and loses at equal compute.

## How far this can go

Not estimable from a toy model. The result argues against expecting a balanced 1/2/4 schedule to improve the score at the evaluation loop count; its value would be in flexible inference cost.

## Next steps

The track is closed on HFM. If a GPU becomes available: repeat at 100M+ parameters, and train an exit head rather than thresholding the top probability.
