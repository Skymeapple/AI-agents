"""Closed-form 'reliability tax' (Prop. 3): delay of a q-reliability definition relative to q=0.5 equals
logit(q)/|b| doublings, i.e. logit(q)/|b| * DT_50 months, if the logistic slope b is constant.
Compares the closed form with direct per-q trend fits on METR v1.1 point estimates.
"""
import json
import numpy as np
import pandas as pd
from core import load_runs, fit_model, log2_horizon
from analysis_metr import trend, TARGETS, frontier_mask

R = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "results")
df = load_runs("1-1"); df = df[df.release >= pd.Timestamp("2023-01-01")]
rows = []
for m, g in df.groupby("alias"):
    f = fit_model(g)
    rows.append(dict(alias=m, t=(g.release.iloc[0] - pd.Timestamp("2023-01-01")).days / 365.25, a=f["a"], b=f["b"]))
P = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
L = TARGETS["1 work-month (167h)"]
P["l50"] = log2_horizon(P.a, P.b, 0.5)
fm = frontier_mask(P.t.values, P.l50.values)
b_front = float(P.b[fm].mean())
a50, s50 = trend(P.t.values, P.l50.values, True, 0.0)
dt50 = 12 / s50
arr50 = 2023 + (L - a50) / s50
out = {"mean_b_frontier": b_front, "dt50_months": dt50, "arrival50": arr50, "rows": []}
for q in [0.8, 0.9, 0.95, 0.99, 0.999]:
    lq = log2_horizon(P.a, P.b, q)
    a_, s_ = trend(P.t.values, lq.values, True, 0.0)
    closed = np.log(q / (1 - q)) / abs(b_front)  # doublings
    out["rows"].append(dict(q=q, closed_form_doublings=closed, closed_form_delay_years=closed * dt50 / 12,
                            direct_fit_arrival=2023 + (L - a_) / s_, direct_delay_years=2023 + (L - a_) / s_ - arr50,
                            direct_dt_months=12 / s_))
json.dump(out, open(f"{R}/reliability_tax.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
