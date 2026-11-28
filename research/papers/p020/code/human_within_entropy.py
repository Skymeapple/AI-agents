"""Mean within-list category entropy (bits) of human lists (first 10 responses), cited in Discussion."""
import json, os
import numpy as np, pandas as pd
from infodecomp import H_counts
HERE = os.path.dirname(os.path.abspath(__file__))
H = pd.read_csv(os.path.join(HERE, "..", "data", "data_humans.csv"))
out = {}
for t in (1, 2, 3):
    h = H[H.task == t].sort_values(["pid", "order"])
    h = h[h.groupby("pid").pid.transform("size") >= 10].groupby("pid").head(10)
    x = pd.factorize(h.category)[0]
    out[t] = float(h.assign(x=x).groupby("pid").x.apply(lambda v: H_counts(np.bincount(v))).mean())
json.dump(out, open(os.path.join(HERE, "..", "results", "human_within_list_entropy.json"), "w"), indent=1)
print(out)
