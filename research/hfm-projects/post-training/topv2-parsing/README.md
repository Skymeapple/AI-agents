# TOPv2 compositional semantic parsing

- **Field:** post-training
- **HFM track:** `discovery_live_live_post_topv2_fixed_auto` (paused on HFM)
- **Project status:** Tested on public data

## Readiness

- **State:** A CPU study has run. The language-model experiment is ready once a free GPU account is connected.
- **Data access:** Qwen2.5-7B and TOPv2 (`WillHeld/top_v2`) are reachable from the free compute, verified 2026-11-25.
- **Compute:** One GPU. A 16 GB free GPU fits LoRA training of the 2B model directly and the 7B model with 4-bit loading.
- **Still missing:** A free GPU account.
- **Literature and materials:** [`REFERENCES.md`](REFERENCES.md) in this folder.

## The challenge

Post-train Qwen2.5-7B-Instruct on TOPv2 compositional semantic parsing. Choose recipe, curriculum, hyperparameters, and efficiency tradeoffs within original task rules. Produce a LoRA adapter from this exact base checkpoint. Read frozen tests, task context, and templates first.

- **Metric:** `exact_match` (maximize)
- **HFM baseline:** not published
- **HFM best so far:** 83
- **HFM effort:** 17 proposals, 14 scored, 0 accepted improvements, about $252 of compute
- **HFM's current best idea:** 在保持1353的12288条训练行、清洁dev、单遍数据顺序和按样本进度计算的学习率曲线的条件下，将有效batch从16减至8，会比1353获得更高的耦合全树解析准确率。

## Existing theory and methods

Seq2seq parsing, grammar-constrained decoding.

## Public data

TOPv2 is public and was downloaded on the free compute (`WillHeld/top_v2` on Hugging Face).

## What limits progress

Recipe tuning on a fixed 12,288 rows. 10% of test parses need a tree shape absent from those rows.

## Route to a step change

Grammar-constrained decoding and grammar-sampled synthetic trees.

## Experiment and results

A CPU study on the public dataset (124,597 training and 38,785 test parses; 8,000 test parses scored). Script: [`topv2_classical.py`](topv2_classical.py); log: [`run.log`](../../../compute/results/topv2-classical/run.log). It asks how much of the task is recall of tree structures seen in training.

| Training examples | Test parses whose tree shape was seen in training | Same utterance and parse in training | Classical tagger, exact match | On seen shapes | On unseen shapes | Flat parses | Nested parses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12,288 (HFM's budget) | 90.0% | 4.3% | 38.5% | 42.5% | 2.4% | 42.0% | 18.6% |
| 30,000 | 93.1% | 6.4% | 43.9% | 47.0% | 1.8% | 47.9% | 21.4% |

- **Tree shapes are mostly seen; utterances are not.** 90% of test parses use a shape present in the 12,288 training rows, but only 4% repeat a training utterance.
- **A classical tagger reaches 38.5%, far below HFM's 83.** Unlike FabNER, a simple method does not match the fine-tuned language model here. The model's contribution is real.
- **Unseen shapes are the hard tenth.** The tagger gets about 2% of them. If HFM's model were perfect on seen shapes it would score 90, so its 83 leaves at most 7 points there and the remaining 10 points need new shapes.
- **More data helps slowly.** 2.4 times the training rows adds 5.4 points for the tagger and 3 points of shape coverage.
- **A first tagger did not finish.** A conditional random field over 1,271 distinct tags was still fitting after two hours and was replaced by a per-word linear classifier.

**Follow-up: grammar-constrained decoding, tested without fine-tuning.** A small open model (Qwen2.5-3B-Instruct, 4-bit, on a free CPU runner) is shown the 8 most similar training utterances with their parses and writes the parse. Test set: 300 random test parses plus 100 more with unseen tree shapes. Script: [`topv2_retrieval_fewshot.py`](topv2_retrieval_fewshot.py); outputs: [`topv2-fewshot-plain`](../../../compute/results/topv2-fewshot-plain/), [`topv2-fewshot-grammar`](../../../compute/results/topv2-fewshot-grammar/).

| Decoding, same model and prompt | Exact match (300 random) | Seen shapes (270) | Unseen shapes (130) | Well-formed output |
| --- | --- | --- | --- | --- |
| Free generation | 26.7% | 29.6% | 4.6% | 64.8% |
| Constrained to well-formed trees with known labels | 45.7% | 50.0% | 6.9% | 100% |

- **The grammar adds 19 points with no training.** On the 400 examples it fixed 60 and broke 2. It also beats the trained classical tagger (38.5%).
- **The gain is on seen shapes.** On unseen shapes it moves from 6 to 9 correct of 130, which is within noise. A grammar removes malformed output; it does not supply new compositions.
- **So the proposed route is half right.** Constrained decoding is worth testing on the fine-tuned model, where malformed output is rarer and the gain will be smaller. The unseen tenth needs something else: training trees sampled from the grammar.
- **Retrieval alone is limited:** one of the 8 retrieved examples has the right shape for 77% of seen-shape questions.

These numbers come from public data and are not comparable to HFM's score, which uses private data and scoring.

## How far this can go

HFM's best is 83.0 exact match. With 12,288 training rows, 10% of test parses have a tree shape never seen in training, so recipe tuning that only sharpens seen shapes tops out near 90. Published full-data results are in the high 80s.

## Next steps

- Split the fine-tuned model's errors by seen and unseen shape. This needs a GPU.
- Measure how often the fine-tuned model's errors are malformed trees; that share is what constrained decoding can recover.
- For the unseen shapes, add training trees sampled from the grammar. Constrained decoding did not help there.
