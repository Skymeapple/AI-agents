"""Specification multiverse: exposure measure x task weighting x aggregation rule.

For every specification computes
  H_net   : employment-weighted mean displacement share (positive = net job loss), May-2024 employment
  H_gross : employment-weighted mean of max(d, 0)
  H_half  : share of employment in occupations with d >= 0.5
  sector-level displacement shares for 20 NAICS sectors and 22 SOC major groups.
Writes results/multiverse_headline.csv, results/multiverse_sector.csv, results/multiverse_major.csv,
results/occ_stats.pkl
"""
import os, pickle, itertools
import numpy as np, pandas as pd
from lib import R, MEASURES, WEIGHTS, occ_stats, rule_table

L = pd.read_csv(os.path.join(R, "tasks_merged.csv"), dtype={"onet": str, "oes": str, "soc": str})
E = pd.read_csv(os.path.join(R, "emp_national.csv"), dtype={"soc": str})
S = pd.read_csv(os.path.join(R, "emp_sector.csv"), dtype={"soc": str, "NAICS": str})
E = E[E["has_tasks"] & E["emp2024"].notna()].set_index("soc")
emp = E["emp2024"]

rules = rule_table()
stats = {}
head, sect, major = [], [], []
S = S[S["soc"].isin(emp.index) & S["emp"].notna()]
for m, w in itertools.product(MEASURES, WEIGHTS):
    st = occ_stats(L, m, w).reindex(emp.index)
    stats[(m, w)] = st
    for name, fam, f in rules:
        d = pd.Series(f(st), index=st.index)
        H_net = np.average(d, weights=emp)
        H_gross = np.average(np.clip(d, 0, None), weights=emp)
        H_half = emp[d >= 0.5].sum() / emp.sum()
        head.append(dict(measure=m, weight=w, rule=name, family=fam,
                         H_net=H_net, H_gross=H_gross, H_half=H_half))
        Sd = S.assign(d=S["soc"].map(d))
        ks = Sd.groupby("NAICS").apply(lambda x: np.average(x["d"], weights=x["emp"]), include_groups=False)
        for k, v in ks.items():
            sect.append(dict(measure=m, weight=w, rule=name, family=fam, sector=k, D=v))
        mj = pd.DataFrame({"d": d, "emp": emp, "maj": emp.index.str[:2]})
        for k, x in mj.groupby("maj"):
            major.append(dict(measure=m, weight=w, rule=name, family=fam, major=k,
                              D=np.average(x["d"], weights=x["emp"])))
pd.DataFrame(head).to_csv(os.path.join(R, "multiverse_headline.csv"), index=False)
pd.DataFrame(sect).to_csv(os.path.join(R, "multiverse_sector.csv"), index=False)
pd.DataFrame(major).to_csv(os.path.join(R, "multiverse_major.csv"), index=False)
with open(os.path.join(R, "occ_stats.pkl"), "wb") as fh:
    pickle.dump(stats, fh)
print("specs:", len(head))
