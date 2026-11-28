"""E3: Is a mean component score a sufficient statistic for replication risk?
E4: Checkpointed vs atomic pipelines, bottleneck bound, elasticities on an
    ILLUSTRATIVE 4-stage profile (values are not estimates of any real model)."""
import json, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import *

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(1)
K = 4

# ---- E3: random component profiles (p_k ~ Beta(0.6,1.2), common h)
n = 20000
p = rng.beta(0.6, 1.2, size=(n, K)).clip(1e-4, 0.999)
h = 0.05
R_ck = R_checkpoint(p, np.full_like(p, h))
R_at = R_atomic_multistage(p, h)
mean_score = p.mean(1)
pass10 = 1 - (1 - p) ** 10
mean_pass10 = pass10.mean(1)
frac_gt50 = (pass10 > 0.5).mean(1)
minp = p.min(1)
odds_min = stage_odds(p, h).min(1)
stats = {}
for name, s in [("mean_p1", mean_score), ("mean_pass@10", mean_pass10), ("frac_families_pass@10>50%", frac_gt50),
                ("min_p1", minp), ("geo_mean_p1", np.exp(np.log(p).mean(1))), ("min_stage_odds", odds_min)]:
    rho = spearmanr(s, np.log(R_ck)).correlation
    # classification: does threshold on statistic separate R>1?
    sup = R_ck > 1
    # best achievable balanced accuracy with a single threshold
    order = np.sort(np.unique(np.quantile(s, np.linspace(0, 1, 401))))
    ba = max(0.5 * (((s >= t) & sup).sum() / max(sup.sum(), 1) + ((s < t) & ~sup).sum() / max((~sup).sum(), 1))
             for t in order)
    stats[name] = dict(spearman_vs_logR=float(rho), best_threshold_balanced_acc=float(ba))
    print(f"{name:28s} rho={rho:.3f}  bal.acc={ba:.3f}")
stats["frac_supercritical_checkpoint"] = float((R_ck > 1).mean())
stats["frac_supercritical_atomic"] = float((R_at > 1).mean())
# bound check: R <= sigma * min odds
stats["bound_violations"] = int((R_ck > odds_min * (1 + 1e-9)).sum())

# Explicit counterexample pair: equal mean pass@10, opposite criticality
pairs = []
A = np.array([0.9, 0.9, 0.9, 0.004]); B = np.array([0.2, 0.2, 0.2, 0.2])
C = np.array([0.99, 0.99, 0.99, 0.01]); D = np.array([0.12, 0.12, 0.12, 0.12])
# pairwise discordance: fraction of random profile pairs whose ordering by statistic disagrees with ordering by R
ii = rng.integers(0, n, 200000); jj = rng.integers(0, n, 200000)
for nm_, s_ in [("mean_p1", mean_score), ("mean_pass@10", mean_pass10), ("min_stage_odds", odds_min)]:
    disc = np.mean(np.sign(s_[ii] - s_[jj]) != np.sign(R_ck[ii] - R_ck[jj]))
    stats[nm_]["pairwise_discordance"] = float(disc); print(nm_, "discordance", disc)
for nm, v in [("A", A), ("B", B), ("C", C), ("D", D)]:
    pairs.append(dict(name=nm, p1=v.tolist(), mean_p1=float(v.mean()),
                      mean_pass10=float((1 - (1 - v) ** 10).mean()),
                      R_checkpoint=float(R_checkpoint(v, np.full(4, h))),
                      R_atomic=float(R_atomic_multistage(v, h))))
stats["counterexample"] = pairs
print(pairs)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
idx = rng.choice(n, 3000, replace=False)
axs[0].scatter(mean_pass10[idx], R_ck[idx], s=3, alpha=0.4, color="#3b6ea5")
axs[0].set_yscale("log"); axs[0].axhline(1, color="k", lw=0.7)
axs[0].set_xlabel("mean pass@10 across 4 stages"); axs[0].set_ylabel("R (checkpointed, h=0.05)")
axs[0].set_title(f"mean score (Spearman {stats['mean_pass@10']['spearman_vs_logR']:.2f})")
axs[1].scatter(odds_min[idx], R_ck[idx], s=3, alpha=0.4, color="#b5523b")
axs[1].set_xscale("log"); axs[1].set_yscale("log")
xx = np.logspace(-3, 4, 10); axs[1].plot(xx, xx, "k--", lw=0.8, label="R = min odds (bound)")
axs[1].axhline(1, color="k", lw=0.7); axs[1].legend(fontsize=7)
axs[1].set_xlabel("min stage odds  p/((1-p)h)"); axs[1].set_title(
    f"bottleneck odds (Spearman {stats['min_stage_odds']['spearman_vs_logR']:.2f})")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e3_aggregation.png", dpi=150)

# ---- E4: illustrative profile; atomic vs checkpoint; elasticities
stages = ["obtain resources", "exfiltrate weights", "deploy to compute", "persist"]
p_ill = np.array([0.01, 0.2, 0.5, 0.1])   # ILLUSTRATIVE per-attempt success
e4 = dict(stages=stages, p_illustrative=p_ill.tolist(), rows=[])
for hh in [0.3, 0.1, 0.03, 0.01, 0.003]:
    hv = np.full(4, hh)
    eh, ep = elasticities_checkpoint(p_ill, hv)
    e4["rows"].append(dict(h=hh, R_atomic=float(R_atomic_multistage(p_ill, hh)),
                           R_checkpoint=float(R_checkpoint(p_ill, hv)),
                           min_odds=float(stage_odds(p_ill, hv).min()),
                           elasticity_h=eh.tolist(), elasticity_p=ep.tolist()))
    print(hh, e4["rows"][-1]["R_atomic"], e4["rows"][-1]["R_checkpoint"], np.round(eh, 3))
# Defense allocation: double detection at one stage, baseline h=0.01
base = np.full(4, 0.01); R0 = R_checkpoint(p_ill, base)
alloc = {}
for k in range(4):
    hv = base.copy(); hv[k] *= 10
    alloc[stages[k]] = float(R_checkpoint(p_ill, hv))
hv = base * 10 ** 0.25; alloc["all stages x10^0.25 (same log-budget)"] = float(R_checkpoint(p_ill, hv))
e4["defense_allocation_base_h0.01"] = dict(R_base=float(R0), R_after_10x_at_stage=alloc)
print("allocation", R0, alloc)
stats["E4"] = e4
json.dump(stats, open(f"{OUT}/e3_e4_aggregation.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(5, 3.4))
hs = np.logspace(-3.5, 0, 100)
ax.plot(hs, [R_atomic_multistage(p_ill, x) for x in hs], label="atomic retries")
ax.plot(hs, [R_checkpoint(p_ill, np.full(4, x)) for x in hs], label="checkpointed retries")
ax.plot(hs, [stage_odds(p_ill, np.full(4, x)).min() for x in hs], "k--", lw=0.8, label="bound: min stage odds")
ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("detection prob. per failed try h (all stages)"); ax.set_ylabel("R")
ax.set_title("E4: illustrative 4-stage profile"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e4_pipeline.png", dpi=150)
