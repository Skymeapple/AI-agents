"""Experiment R: test the beta-binomial redundancy/division model on real LLM answers.

Data: results/answers.npz = item x model answer matrix built (code/build_matrix.py) from the
public MMLU-Pro released model outputs (TIGER-AI-Lab/MMLU-Pro, eval_results/*.zip).
11,429 items with >=3 options, 46 models. Each model = one 'agent' (one sample per item).

R1  Vote scaling: random n-subsets of an exchangeable-ish 'peer' pool. Fit (p, rho) on a
    calibration half of the items; predict held-out plurality accuracy under two bounding
    wrong-answer models: m=2 (all wrong agents agree) and m=10 uniform wrong answers.
R2  Division of labor by routing: per category, 'specialist' teams are random n-subsets of the
    top-T models *for that category*; 'generalist' teams are random n-subsets of the top-T models
    overall (both selected on calibration items). Composite task = one held-out item from each of
    K=4 random categories; composite accuracy = product of per-category accuracies (items drawn
    independently). Observed break-even hand-off error eps*(n) vs. pilot-based prediction.

Run: python3 code/real_data_test.py   (~1-2 min)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, plurality
from theory import vote_acc, ceiling, mc_team

os.chdir(HERE)
OUT = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(11)
d = load()
A, y, cat, models = d["A"], d["y"], d["cat"], d["models"]
C = A == y[:, None]
n_items, M = A.shape
perm = rng.permutation(n_items)
cal = np.zeros(n_items, bool); cal[perm[: n_items // 2]] = True
hold = ~cal
acc_cal = C[cal].mean(0)


def fit_p_rho(Cm):
    """Method-of-moments (p, ICC) on an items x agents 0/1 matrix."""
    na = Cm.shape[1]; m = Cm.mean(1); p = m.mean(); v = m.var(ddof=1)
    rho = (v / (p * (1 - p)) - 1 / na) / (1 - 1 / na)
    return float(p), float(np.clip(rho, 1e-4, 0.99))


def obs_curve(Asub, ysub, ns, reps=150):
    out = []
    for n in ns:
        a = 0.0
        for _ in range(reps):
            idx = rng.choice(Asub.shape[1], n, replace=False)
            a += (plurality(Asub[:, idx], rng) == ysub).mean()
        out.append(a / reps)
    return np.array(out)


def pred_m10(n, p, rho, items=6000):
    return mc_team(n, 1, p, rho, 10, items, np.random.default_rng(n), "coord")


R = {"n_items": int(n_items), "n_models": int(M)}
# ---------------- R1 ----------------
pools = {"peer_0.60-0.80": np.nonzero((acc_cal >= 0.60) & (acc_cal <= 0.80))[0],
         "mid_0.35-0.60": np.nonzero((acc_cal >= 0.35) & (acc_cal < 0.60))[0]}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
R["R1"] = {}
for ax, (name, pool) in zip(axes, pools.items()):
    ns = [n for n in [1, 3, 5, 7, 9, 11, 13, 15, 19] if n <= len(pool)]
    p, rho = fit_p_rho(C[cal][:, pool])
    obs = obs_curve(A[hold][:, pool], y[hold], ns)
    p2 = np.array([vote_acc(n, p, rho) for n in ns])
    p10 = np.array([pred_m10(n, p, rho) for n in ns])
    piid = np.array([pred_m10(n, p, 1e-9) if False else mc_team(n, 1, p, 0.0, 10, 6000, np.random.default_rng(n), "coord") for n in ns])
    full = (plurality(A[hold][:, pool], rng) == y[hold]).mean()
    R["R1"][name] = dict(models=[models[i] for i in pool], n_pool=int(len(pool)), p=p, rho=rho, ns=ns,
                         obs=obs.tolist(), pred_m2=p2.tolist(), pred_m10=p10.tolist(), pred_iid_m10=piid.tolist(),
                         ceiling_m2=ceiling(p, rho, 2), ceiling_m10=ceiling(p, rho, 10), full_pool_plurality=float(full),
                         MAE_m2=float(np.abs(obs - p2).mean()), MAE_m10=float(np.abs(obs - p10).mean()),
                         MAE_iid=float(np.abs(obs - piid).mean()),
                         obs_within_bounds=bool(np.all((obs >= p2 - 0.01) & (obs <= p10 + 0.01))))
    ax.plot(ns, obs, "ko", label="observed (held-out)")
    ax.plot(ns, p2, "-", label="beta-binomial, m=2 (worst case)")
    ax.plot(ns, p10, "--", label="beta-binomial, m=10 uniform wrong")
    ax.plot(ns, piid, ":", label="independent errors (rho=0), m=10")
    ax.set_title(f"{name}: {len(pool)} models, p={p:.2f}, rho={rho:.2f}", fontsize=9)
    ax.set_xlabel("models voting (n)"); ax.set_ylabel("plurality accuracy")
    print(name, len(pool), round(p, 3), round(rho, 3), "obs", obs.round(3), "m2", p2.round(3), "m10", p10.round(3), flush=True)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig8_real_vote.png"), dpi=150); plt.close()

# ---------------- R2 ----------------
cats = np.unique(cat)
T, K = 8, 4
ns = [1, 3, 5, 7]
top1 = int(np.argmax(acc_cal))
base = np.array([j for j in range(M) if j != top1])   # exclude the single dominant model
R["R2"] = {"T": T, "K": K, "ns": ns, "excluded_top_model": models[top1]}
gen_pool = base[np.argsort(-acc_cal[base])[:T]]
per = {}
for c in cats:
    ic, ih = cal & (cat == c), hold & (cat == c)
    acc_c = C[ic].mean(0)
    spec_pool = base[np.argsort(-acc_c[base])[:T]]
    row = {"spec_pool": [models[i] for i in spec_pool], "n_hold": int(ih.sum())}
    for typ, pool in [("G", gen_pool), ("S", spec_pool)]:
        p, rho = fit_p_rho(C[ic][:, pool])
        row[typ] = dict(p=p, rho=rho, obs=obs_curve(A[ih][:, pool], y[ih], ns, reps=100).tolist(),
                        pred_m2=[vote_acc(n, p, rho) for n in ns],
                        pred_m10=[pred_m10(n, p, rho, 3000) for n in ns])
    per[c] = row
R2sets = [rng.choice(len(cats), K, replace=False) for _ in range(300)]


def eps_from(key_G, key_S, typ_key):
    out = []
    for s in R2sets:
        g = np.prod([per[cats[i]]["G"][typ_key] for i in s], axis=0)
        sp = np.prod([per[cats[i]]["S"][typ_key] for i in s], axis=0)
        out.append(1 - (np.array(g) / np.array(sp)) ** (1 / (K - 1)))
    return np.array(out)


E = {k: eps_from("G", "S", k) for k in ["obs", "pred_m2", "pred_m10"]}
R2s = {k: {"mean": E[k].mean(0).round(4).tolist(), "sd_over_category_sets": E[k].std(0).round(4).tolist()} for k in E}
R["R2"]["eps_star"] = R2s
R["R2"]["per_category"] = per
R["R2"]["mean_p_G"] = float(np.mean([per[c]["G"]["p"] for c in cats])); R["R2"]["mean_p_S"] = float(np.mean([per[c]["S"]["p"] for c in cats]))
R["R2"]["mean_rho_G"] = float(np.mean([per[c]["G"]["rho"] for c in cats])); R["R2"]["mean_rho_S"] = float(np.mean([per[c]["S"]["rho"] for c in cats]))
R["R2"]["frac_sets_eps_obs_decreasing_n1_to_n7"] = float((E["obs"][:, -1] < E["obs"][:, 0]).mean())
print("R2 eps*", json.dumps(R2s), "pG", R["R2"]["mean_p_G"], "pS", R["R2"]["mean_p_S"], "rhoG", R["R2"]["mean_rho_G"], "rhoS", R["R2"]["mean_rho_S"])
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for k, st in [("obs", "ko-"), ("pred_m2", "C0--"), ("pred_m10", "C1--")]:
    ax.errorbar(ns, R2s[k]["mean"], yerr=R2s[k]["sd_over_category_sets"], fmt=st, capsize=2, label=k)
ax.set_xlabel("models per subtask (n)"); ax.set_ylabel("break-even hand-off error eps*")
ax.set_title("MMLU-Pro routing (K=4 categories, T=8)", fontsize=9); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig9_real_eps_star.png"), dpi=150); plt.close()
json.dump(R, open(os.path.join(OUT, "real_data_results.json"), "w"), indent=1)
