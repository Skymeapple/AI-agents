# Agentic Pretraining Under Fixed Inference-Cost Constraint

- **Field:** pretraining
- **HFM track:** `discovery_live_exp_agentic_pretrain_v2` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A toy-scale test runs today on CPU. Anything near HFM's scale waits for a GPU.
- **Data access:** FineWeb-Edu and DCLM on Hugging Face: Verified reachable from the free compute on 2026-11-25.
- **Compute:** HFM's budgets are 4 to 8 GPUs for 24 to 168 hours. A free GPU supports only scaled-down studies (small models, hours not days).
- **Still missing:** A free GPU account for scaled-down studies. Matching HFM's budget needs a research allocation or paid compute.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

This work examines the design of a model architecture that is pretrained from scratch using a frozen corpus pool and tokenizer, with a budget of 4 GPUs for 24 hours. The architecture is evaluated against a reference model on a frozen held-out suite, with the requirement that it achieves superior performance at a lower or equal inference cost.

- **Metric:** `neg_cost_adjusted_bpb` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** -0.667
- **HFM effort:** 10 proposals, 9 scored, 0 accepted improvements, about $640 of compute
- **HFM's current best idea:** p392 训练峰值显存仅 40.4GB/141GB、吞吐 120k tok/s;把 micro_batch 4→8~16(grad_accum 等比下调保持全局 262k tok/step)、加大 ce_chunk 或启用 fused CE、扩大 torch.compile 覆盖范围,可把训练吞吐提升 ≥20%,同墙钟多训 ≥1.4B token,在模型与推理代价路径零改动下把 raw macro bpb 再压低 ≥3σ(≈0.0085)。

## Existing theory and methods

Throughput engineering, small dense models, MoE.

## Public data

Public pretraining corpora.

## What limits progress

Throughput: the best idea is using idle GPU memory.

## Route to a step change

Maximise tokens seen per hour first.

## Experiment and results

Code: [`shape_tiny.py`](shape_tiny.py) in this folder (it reuses the model from the looped-pretraining project). It ran on free GitHub Actions compute (four CPU cores, 1.3 to 3 hours per model); logs are in [`research/compute/results/`](../../../compute/results/) under `shape-tiny-*`.

- **This is a toy-scale test.** HFM trains on 4 GPUs for 24 hours. These are byte-level models of about 3.7 million parameters trained on the public TinyStories corpus. It shows which lever matters when everything else is equal. It cannot say what happens at HFM's scale.
- **Question 1: at a fixed inference cost, which shape is best?** Four standard transformers with the same per-token compute (12 × layers × width², about 3.55M) and near-equal parameters see the same bytes for 6,000 steps, two seeds each. Score is bits per byte on held-out text, lower is better.

  | Layers × width | Parameters | Bits per byte (two seeds) | Mean |
  | --- | --- | --- | --- |
  | 2 × 384 | 3.79M | 1.157, 1.135 | 1.146 |
  | 4 × 272 | 3.73M | 1.084, 1.093 | 1.088 |
  | 8 × 192 | 3.67M | 1.069, 1.072 | **1.071** |
  | 16 × 136 | 3.64M | 1.080, 1.077 | 1.079 |

  The middle shape wins. Very shallow is clearly worse (+0.075). Deeper and narrower than 8 layers is slightly worse (+0.008), which is larger than the spread between the two seeds of either shape (0.003) but rests on two seeds each. Among the three shapes with 4 or more layers the whole range is 0.017.
- **Question 2: what does more training buy at the same inference cost?** The 8 × 192 model trained for 12,000 steps instead of 6,000 scores 0.984 and 0.990 (mean 0.987), a gain of 0.084 bits per byte with no change to the model.
- **Reading:** doubling the training the model receives is worth about five times the whole spread between reasonable shapes, and it costs nothing at inference. This supports the HFM agents' current best idea (raise training throughput) over architecture changes, at this scale. A model this small is far from saturated, so the size of the training gain will shrink at HFM's scale; the ordering of the two levers is the finding, not the numbers.
- **Not tested:** the HFM metric also divides by a cost term whose formula is not published, so these runs hold cost equal instead of trading it. Looping, mixture-of-experts and early exit were not tried here; looping is covered in the [looped-pretraining project](../looped-pretraining-fixed-compute/README.md).

## How far this can go

Not estimable without a baseline. The toy result says the first thing to try with a GPU is throughput, then a modest depth sweep around the reference shape.

## Next steps

Publish a reference score.
