"""Exp 12 (revision 1): back-of-envelope evaluation-noise budget for held-out log-loss (reviewer W3).
This is an ASSUMPTION-DRIVEN calculation, not a measurement. Assumptions (stated in the paper):
  * per-token loss standard deviation s_tok in {2, 3} nats (heavy-tailed token losses around a mean of ~2 nats);
  * design effect deff in {1, 10} for within-document correlation (n_eff = n / deff);
  * mean loss L ~ 2 nats, so the SE in log-loss is SE_nats / L.
It ignores seed-to-seed and recipe variation, which add to (and at scale probably dominate) evaluation noise."""
import json, itertools
from common import RES
rows = []
for n, s, deff in itertools.product([1e6, 1e7, 1e8, 1e9], [2.0, 3.0], [1, 10]):
    rows.append(dict(eval_tokens=n, s_tok=s, design_effect=deff, se_log_loss=s / (n / deff) ** 0.5 / 2.0))
for r in rows:
    print(r)
json.dump(rows, open(f"{RES}/exp12_noise_budget.json", "w"), indent=2)
