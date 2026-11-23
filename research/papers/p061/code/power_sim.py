"""Monte-Carlo identification experiment: when can occupational employment data tell
aggregation rules apart?

World: true rule r* generates long-run displacement d*_j. By the observation date a share phi
of it has materialised: y_j = -phi * d*_j + u_j, where u_j are bootstrap draws of the actual
within-major-group residuals of 2021->2024 OEWS log employment changes (so the noise has the
real occupational distribution and heteroskedasticity). For each draw we compute, after
partialling out major-group fixed effects, the employment-weighted correlation between -y and
each candidate rule's d_j, and record whether the true rule wins.
"""
import os, pickle, json
import numpy as np, pandas as pd
from lib import R, rule_table

rng = np.random.default_rng(20261122)
E = pd.read_csv(os.path.join(R, "emp_national.csv"), dtype={"soc": str}).set_index("soc")
E = E[E["has_tasks"]][["emp2021", "emp2024"]].dropna()
E = E[(E.emp2021 > 0) & (E.emp2024 > 0)]
st = pickle.load(open(os.path.join(R, "occ_stats.pkl"), "rb"))[("hum_beta", "im")].reindex(E.index)
w = E.emp2021.values / E.emp2021.mean()
maj = E.index.str[:2]
G = pd.get_dummies(maj).values.astype(float)


def demean(v):
    # weighted within-group demeaning
    num = G.T @ (w * v); den = G.T @ w
    return v - G @ (num / den)


y_obs = np.log(E.emp2024 / E.emp2021).values
u = demean(y_obs)
CAND = ["linear", "threshold_0.5", "oring_indep", "ces_labelled_s0.5_e1.0_t0.9",
        "ces_automate_s2.0_e0.5_t0.9", "ces_automate_s0.5_e1.5_t0.9"]
rules = {n: f for n, fam, f in rule_table()}
D = {n: demean(np.asarray(rules[n](st), float)) for n in CAND}
# pairwise correlation of candidate predictions (after FE), for context
C = pd.DataFrame({n: D[n] for n in CAND})
corr = pd.DataFrame([[np.sum(w * C[a] * C[b]) / np.sqrt(np.sum(w * C[a] ** 2) * np.sum(w * C[b] ** 2))
                      for b in CAND] for a in CAND], index=CAND, columns=CAND)
corr.to_csv(os.path.join(R, "candidate_rule_corr.csv"))

# note: ces_augment_s0.5_e0.5_t0.9 is dropped from the race because after FE its prediction is
# collinear with the linear rule (corr > 0.99); see candidate_rule_corr_full.csv
full = ["linear", "ces_augment_s0.5_e0.5_t0.9", "ces_augment_s2.0_e0.5_t0.9", "ces_automate_s0.5_e1.0_t0.9"]
Cf = pd.DataFrame({n: demean(np.asarray(rules[n](st), float)) for n in full})
pd.DataFrame([[np.sum(w*Cf[a]*Cf[b])/np.sqrt(np.sum(w*Cf[a]**2)*np.sum(w*Cf[b]**2)) for b in full] for a in full],
             index=full, columns=full).to_csv(os.path.join(R, "candidate_rule_corr_full.csv"))
phis = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
REPS = 400
rows = []
crit, tnull = {}, {}
for true in CAND:
    for phi in phis:
        wins = 0; sig = 0
        for _ in range(REPS):
            idx = rng.integers(0, len(u), len(u))
            y = -phi * D[true] + u[idx]
            y = demean(y)
            sc = {n: np.sum(w * D[n] * (-y)) / np.sqrt(np.sum(w * D[n] ** 2) * np.sum(w * y ** 2)) for n in CAND}
            wins += max(sc, key=sc.get) == true
            # crude detection: |t| of true-rule slope > 1.96 (homoskedastic weighted)
            x = D[true]; b = np.sum(w * x * y) / np.sum(w * x * x)
            e = y - b * x; se = np.sqrt(np.sum(w * e ** 2) / (len(y) - 2) / np.sum(w * x * x))
            sig += abs(b / se) > crit.get(true, 1.96)
            if phi == 0.0: tnull.setdefault(true, []).append(abs(b / se))
        if phi == 0.0: crit[true] = float(np.quantile(tnull[true], 0.95))
        rows.append(dict(true_rule=true, phi=phi, p_correct=wins / REPS, power_detect_size_adjusted=sig / REPS,
                         crit_t=crit[true]))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(R, "power_sim.csv"), index=False)
print(corr.round(2).to_string())
print(P.pivot(index="true_rule", columns="phi", values="p_correct").round(2).to_string())
print(P.pivot(index="true_rule", columns="phi", values="power_detect_size_adjusted").round(2).to_string())
# minimal detectable effect for the linear exposure slope in 2021->2024 (from regressions.csv)
reg = pd.read_csv(os.path.join(R, "regressions.csv"))
se = reg[(reg.measure == "hum_beta") & (reg.period == "2021_2024") & (reg.spec == "M2_s_FE")]["se"].iloc[0]
json.dump({"MDE_80pct_power_per_SD_s_2021_2024": 2.8 * se, "se": se, "chance_level": 1 / len(CAND)},
          open(os.path.join(R, "mde.json"), "w"), indent=1)
print("MDE (80% power, 5% two-sided) per 1 SD of exposure:", 2.8 * se)
