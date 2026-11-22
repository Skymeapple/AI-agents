"""Exp 1: error-correlation structure of 46 real LLM 'agents' on MMLU-Pro.
Outputs: results/exp1_*.json/.csv and figures."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, family

d = load()
A, y, nopt, models = d["A"], d["y"], d["nopt"], d["models"]
n, m = A.shape
C = (A == y[:, None]).astype(float)
W = (A != y[:, None]) & (A >= 0)          # committed wrong answer
acc = C.mean(0)
fam = [family(x) for x in models]

# pairwise correctness correlation (phi)
rho = np.corrcoef(C.T)
# conditional wrong-answer agreement: P(same letter | both committed wrong); chance = E[1/(nopt-1)]
agree = np.full((m, m), np.nan)
chance = np.full((m, m), np.nan)
inv = 1.0 / (nopt - 1)
for i in range(m):
    for j in range(i + 1, m):
        both = W[:, i] & W[:, j]
        if both.sum() >= 50:
            agree[i, j] = agree[j, i] = (A[both, i] == A[both, j]).mean()
            chance[i, j] = chance[j, i] = inv[both].mean()

iu = np.triu_indices(m, 1)
same = np.array([fam[i] == fam[j] and fam[i] != "other" for i, j in zip(*iu)])
r_u, a_u, c_u = rho[iu], agree[iu], chance[iu]
# control for accuracy similarity: regress rho on |acc_i-acc_j| and min accuracy, test family indicator
X = np.column_stack([np.ones(len(r_u)), np.abs(acc[iu[0]] - acc[iu[1]]), np.minimum(acc[iu[0]], acc[iu[1]]), same])
beta_r, *_ = np.linalg.lstsq(X, r_u, rcond=None)
ok = ~np.isnan(a_u)
beta_a, *_ = np.linalg.lstsq(X[ok], a_u[ok], rcond=None)

# permutation test for family effect on residual agreement (shuffle family labels)
rng = np.random.default_rng(0)
def fam_coef(f, target, mask):
    s = np.array([f[i] == f[j] and f[i] != "other" for i, j in zip(*iu)])
    Xp = np.column_stack([X[:, :3], s])[mask]
    return np.linalg.lstsq(Xp, target[mask], rcond=None)[0][3]
perm_r, perm_a = [], []
for _ in range(2000):
    f = list(rng.permutation(fam))
    perm_r.append(fam_coef(f, r_u, np.ones(len(r_u), bool)))
    perm_a.append(fam_coef(f, a_u, ok))
p_r = (np.sum(np.abs(perm_r) >= abs(beta_r[3])) + 1) / 2001
p_a = (np.sum(np.abs(perm_a) >= abs(beta_a[3])) + 1) / 2001

# Item-level: pool answer distribution, plurality ceiling (Prop. 1) and coverage
counts = np.zeros((n, 10))
for j in range(m):
    ok_ = A[:, j] >= 0
    counts[np.nonzero(ok_)[0], A[ok_, j]] += 1
q = counts / m
qc = q[np.arange(n), y]
qw = q.copy(); qw[np.arange(n), y] = -1
qwmax = qw.max(1)
ceiling = np.mean(qc > qwmax) + 0.5 * np.mean(qc == qwmax) * 0  # ties counted as loss (conservative)
ceiling_ties_half = np.mean(qc > qwmax) + 0.5 * np.mean(qc == qwmax)
coverage = np.mean(qc > 0)
frac_wrong_attractor = np.mean(qwmax > qc)
# fraction of items where a single wrong option is chosen by >= 50% of the pool
strong_attr = np.mean(qwmax >= 0.5)

out = dict(
    n_items=int(n), n_agents=int(m), acc_min=float(acc.min()), acc_max=float(acc.max()), acc_mean=float(acc.mean()),
    rho_mean=float(r_u.mean()), rho_median=float(np.median(r_u)), rho_min=float(r_u.min()), rho_max=float(r_u.max()),
    wrong_agree_mean=float(np.nanmean(a_u)), wrong_agree_median=float(np.nanmedian(a_u)),
    wrong_agree_chance_mean=float(np.nanmean(c_u)), wrong_agree_ratio_mean=float(np.nanmean(a_u / c_u)),
    rho_same_family_mean=float(r_u[same].mean()), rho_cross_family_mean=float(r_u[~same].mean()),
    agree_same_family_mean=float(np.nanmean(a_u[same])), agree_cross_family_mean=float(np.nanmean(a_u[~same])),
    n_pairs_same_family=int(same.sum()), n_pairs=int(len(same)),
    family_coef_rho=float(beta_r[3]), family_coef_rho_perm_p=float(p_r),
    family_coef_agree=float(beta_a[3]), family_coef_agree_perm_p=float(p_a),
    regression_rho=beta_r.tolist(), regression_agree=beta_a.tolist(),
    pool_plurality_ceiling=float(ceiling), pool_plurality_ceiling_ties_half=float(ceiling_ties_half),
    pool_coverage_any_correct=float(coverage), frac_items_wrong_option_beats_correct=float(frac_wrong_attractor),
    frac_items_single_wrong_option_ge_half=float(strong_attr),
)
json.dump(out, open("../results/exp1_summary.json", "w"), indent=2)
with open("../results/exp1_agents.csv", "w") as f:
    f.write("model,family,accuracy,abstain_rate\n")
    for i in np.argsort(-acc):
        f.write(f"{models[i]},{fam[i]},{acc[i]:.4f},{(A[:, i] < 0).mean():.4f}\n")
print(json.dumps(out, indent=2))

# Figures
o = np.argsort(-acc)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
im = ax[0].imshow(rho[np.ix_(o, o)], cmap="viridis", vmin=0, vmax=1)
ax[0].set_title("Correctness correlation (phi), agents sorted by accuracy")
plt.colorbar(im, ax=ax[0], fraction=0.046)
ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].scatter(c_u[~same], a_u[~same], s=6, alpha=.4, label="cross-family pairs")
ax[1].scatter(c_u[same], a_u[same], s=10, alpha=.8, c="C3", label="same-family pairs")
ax[1].plot([0, .4], [0, .4], "k--", lw=1, label="chance (uniform wrong)")
ax[1].set_xlabel("chance agreement E[1/(K-1)]"); ax[1].set_ylabel("P(same wrong answer | both wrong)")
ax[1].set_xlim(0.1, 0.2); ax[1].legend(fontsize=8); ax[1].set_title("Wrong-answer agreement")
ax[2].hist(qwmax - qc, bins=60, color="C1")
ax[2].axvline(0, c="k", lw=1)
ax[2].set_xlabel("pool share of top wrong option minus share of correct option")
ax[2].set_ylabel("items"); ax[2].set_title(f"Items right of 0 are lost by any large vote ({frac_wrong_attractor:.1%})")
plt.tight_layout(); plt.savefig("../results/fig1_correlation.png", dpi=140)
