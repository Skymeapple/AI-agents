# Pretraining architecture development from scratch

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_agentic_pretrain_research` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. Anything near HFM's scale waits for a GPU.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work develops and trains new architectures from scratch using a frozen public training pool and tokenizer. The research is conducted on eight H20 GPUs in a single container over 168 hours. The session involves proposing, selecting, and iterating on multiple research ideas, with flexibility to revise, pause, or discard experiments.

- **Metric:** `negative_public_validation_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.9624
- **HFM effort:** 3 proposals, 2 scored, 0 accepted improvements, about $18 of compute
- **HFM's current best idea:** Because the p468 depth896 branch was the only complete stable candidate, extending that recipe to substantially more tokens with eight-GPU throughput and screening only low-risk variants will yield a lower four-domain macro BPB than its short 45M-token checkpoint.

## Existing theory and methods

Staged architecture search.

## Public data

Public pretraining corpora.

## What limits progress

Only 3 proposals ran.

## Route to a step change

Short-run scaling-law ranking.

## Experiment and results

A toy-scale study on CPU of the decision an agent in this track has to make repeatedly: choosing which recipes to keep from runs much shorter than the final one. The eight recipes of the factorial study in [`../agentic-pretraining/`](../agentic-pretraining/README.md) were trained for 300, 750 and 1,500 steps, each with its own complete learning-rate schedule, three seeds each, and ranked against the full 6,000-step results. Script: [`short_runs.py`](short_runs.py); logs in `research/compute/results/short-runs-*` and `factorial-tiny-*`. It cannot say what happens at HFM's scale and is not comparable to HFM's score.

| Recipe | 6,000 steps | Seeds |
|---|---|---|
| base | 1.147 | 1.151, 1.149, 1.142 |
| rope | 1.115 | 1.115, 1.111, 1.119 |
| wsd | 1.116 | 1.120, 1.117, 1.112 |
| long | 1.131 | 1.108, 1.169, 1.117 |
| rope + wsd | 1.095 | 1.094, 1.092, 1.100 |
| wsd + long | 1.087 | 1.051, 1.123 |
| rope + long | 1.072 | 1.067, 1.069, 1.080 |
| rope + wsd + long | **1.044** | 1.044, 1.045 |

(bits per byte on held-out TinyStories; the last two recipes have two seeds because the third hit the job's time limit)

| Short run | Rank correlation with 6,000 steps | Best recipe in the short run | Rank of long-sequence recipes |
|---|---|---|---|
| 300 steps (5%) | 0.52 | rope + wsd | long alone last; wsd + long 7th of 8 |
| 750 steps (12.5%) | 0.52 | rope + wsd | long alone last; wsd + long 7th of 8 |
| 1,500 steps (25%) | 0.79 | rope + wsd + long | long alone last; wsd + long 6th of 8 |

- **Short runs rank recipes poorly when a change pays off late.** At 5% and 12.5% of the final length the rank correlation is only 0.52. Longer sequences look harmful: at 300 steps long alone is the worst recipe by 0.32 bits per byte, but at 6,000 steps it beats the base by 0.016, and the best recipe overall contains it.
- **A quarter-length run gets the winner right.** At 1,500 steps the best recipe matches the 6,000-step winner and the correlation rises to 0.79.
- **Changes that act early are judged reliably.** Rotary positions lead from the first short run onwards and keep their lead.
- **The practical rule this suggests:** use very short runs to drop recipes that are worse everywhere, but judge changes whose cost is paid early and whose benefit comes late (longer context, slower schedules) at a quarter of the final length or more.

## How far this can go

Not estimable. At toy scale, short-run screening is reliable for changes that act early and misleading for changes that pay off late.

## Next steps

Publish a reference score. With a GPU, repeat the screening comparison at 100M+ parameters, adding a scaling-law fit across the short runs to see whether extrapolation recovers the long-run ranking.
