"""Second-stage analysis on results/per_order_maxima.npy (produced by analysis.py):
 (1) refutation hazard as a function of record age, vs an exchangeable-record null;
 (2) false-acceptance rates of 'stable' conjectures (age >= a) as universe grows;
 (3) monotone-drift census (Proposition 2);
 (4) late counterexamples among stable conjectures, with witnessing families;
 (5) figures.
Run: python3 analysis2.py
"""
import os, json, csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis import templates, describe, evaluate, NAMES, BIG, load, DATA, RES

M = np.load(os.path.join(RES, "per_order_maxima.npy"))
T = templates(); nT = len(T)
kinds = np.array([t[0] for t in T])
out = {}


def fco_for(k, E):
    f = np.zeros(nT, int)
    for j in list(range(k + 1, 11)) + BIG:
        hit = (M[:, j] > E) & (f == 0); f[hit] = j
    return f


def feats(k):
    E = M[:, 2:k + 1].max(1)
    jfirst = np.argmax(M[:, 2:k + 1] == E[:, None], axis=1) + 2
    return E, k - jfirst


# (1) one-step hazard h(k, a) = P(refuted at order k+1 | accepted at k with record age a)
haz = {}
for k in range(4, 10):
    E, age = feats(k)
    ref_next = M[:, k + 1] > E
    haz[k] = {}
    for a in range(0, k - 1):
        sel = age == a
        if sel.sum() >= 1:
            haz[k][a] = {"n": int(sel.sum()), "hazard": float(ref_next[sel].mean())}
out["one_step_hazard"] = haz

# exchangeable-record null: if M(2..k+1) were exchangeable with continuous law, P(new strict record at k+1)=1/k.
out["exchangeable_null_hazard"] = {k: 1.0 / k for k in range(4, 10)}

# (2) stable conjectures: age >= a
stab = {}
for k in range(4, 11):
    E, age = feats(k)
    f = fco_for(k, E)
    stab[k] = {}
    for a in (0, 1, 2, 3, 4):
        sel = age >= a
        if sel.sum() == 0:
            continue
        stab[k][a] = {"n": int(sel.sum()),
                      "frac_refuted_by10": float(((f > 0) & (f <= 10))[sel].mean()) if k < 10 else None,
                      "frac_refuted_any_observed": float((f > 0)[sel].mean()),
                      "n_refuted_any": int((f > 0)[sel].sum())}
out["stable_false_acceptance"] = stab

# (3) drift census: per-order maxima strictly increasing on orders 8,9,10
inc = (M[:, 9] > M[:, 8]) & (M[:, 10] > M[:, 9])
nondec = (M[:, 9] >= M[:, 8]) & (M[:, 10] >= M[:, 9])
const = (M[:, 8] == M[:, 9]) & (M[:, 9] == M[:, 10])
dec = (M[:, 9] <= M[:, 8]) & (M[:, 10] <= M[:, 9]) & ~const
E10, age10 = feats(10)
f10 = fco_for(10, E10)
out["drift_census_orders_8_9_10"] = {
    "strictly_increasing": int(inc.sum()), "constant": int(const.sum()),
    "nonincreasing_not_constant": int(dec.sum()), "other": int(nT - inc.sum() - const.sum() - dec.sum()),
    "frac_refuted_beyond10_strictly_increasing": float((f10 > 0)[inc].mean()),
    "frac_refuted_beyond10_constant": float((f10 > 0)[const].mean()) if const.any() else None,
    "frac_refuted_beyond10_nonincreasing": float((f10 > 0)[dec].mean()) if dec.any() else None}

# (4) late counterexamples among stable conjectures accepted at k=10 (age>=2: unchanged since order <=8)
L = load("inv_large.bin").astype(np.int64)
tags = open(os.path.join(DATA, "large_meta.txt")).read().split()
late_stable = []
for i in np.where((age10 >= 2) & (f10 > 0))[0]:
    v = evaluate(T[i], L)
    w = np.where(v > E10[i])[0]
    fams = collections.Counter(tags[x].split(":")[0] for x in w)
    late_stable.append({"id": int(i), "statement_true_for_all_connected_n<=10": describe(T[i], int(E10[i])),
                        "record_age_at_k10": int(age10[i]), "first_refuting_order": int(f10[i]),
                        "witness_families_large_sample": dict(fams), "kind": T[i][0]})
late_stable.sort(key=lambda r: (-r["record_age_at_k10"], r["first_refuting_order"]))
json.dump(late_stable, open(os.path.join(RES, "late_counterexamples_stable.json"), "w"), indent=1)
fam_count = collections.Counter()
for r in late_stable:
    for fname in r["witness_families_large_sample"]:
        fam_count[fname] += 1
out["late_stable"] = {"n": len(late_stable), "by_age": dict(collections.Counter(r["record_age_at_k10"] for r in late_stable)),
                      "by_first_order": dict(collections.Counter(r["first_refuting_order"] for r in late_stable)),
                      "witness_family_frequency": dict(fam_count.most_common())}

# random-only vs structured-only falsification power at k=10 (all refuted beyond 10)
gnp = np.array([t.startswith("gnp") for t in tags])
v_gnp = np.zeros(nT, bool); v_str = np.zeros(nT, bool)
for i in np.where(f10 > 0)[0]:
    v = evaluate(T[i], L) > E10[i]
    v_gnp[i] = v[gnp].any(); v_str[i] = v[~gnp].any()
stable_ref = (age10 >= 2) & (f10 > 0)
out["falsifier_power_k10"] = {
    "refuted_total": int((f10 > 0).sum()),
    "by_gnp_sample": int(v_gnp.sum()), "by_structured_families": int(v_str.sum()),
    "stable_refuted_total": int(stable_ref.sum()),
    "stable_by_gnp": int((v_gnp & stable_ref).sum()), "stable_by_structured": int((v_str & stable_ref).sum()),
    "stable_only_structured": int((v_str & ~v_gnp & stable_ref).sum()),
    "stable_only_gnp": int((v_gnp & ~v_str & stable_ref).sum()),
    "stable_only_n11_slice": int((~v_gnp & ~v_str & stable_ref).sum())}

# (5) cross-validated hazard model: logistic in (age, drift3, gap) fit on k in {5,6}, tested on k in {7,8,9}
def design(k):
    E, age = feats(k)
    jj = np.array([k - 2, k - 1, k], float)
    sl = ((M[:, k - 2:k + 1] - M[:, k - 2:k + 1].mean(1, keepdims=True)) * (jj - jj.mean())).sum(1) / 2.0
    q = np.array([t[5] for t in T], float)
    gap = (E - M[:, k]) / q
    X = np.column_stack([np.ones(nT), np.log1p(age), sl / q, gap, (M[:, k] == E).astype(float)])
    y = (M[:, k + 1] > E).astype(float)
    return X, y


def fit_logit(X, y, l2=1e-3, iters=200):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


Xtr = np.vstack([design(k)[0] for k in (5, 6)]); ytr = np.concatenate([design(k)[1] for k in (5, 6)])
w = fit_logit(Xtr, ytr)
from analysis import auc
cv = {"weights[const,log1p(age),slope3/q,gap/q,attained_top]": w.tolist()}
for k in (7, 8, 9):
    X, y = design(k); p = 1 / (1 + np.exp(-X @ w))
    sel = feats(k)[1] >= 1  # stable stratum, where prediction is non-trivial
    cv[k] = {"AUC_next_order_all": auc(p, y), "AUC_next_order_stable(age>=1)": auc(p[sel], y[sel]),
             "n_pos_stable": int(y[sel].sum()), "n_stable": int(sel.sum()),
             "brier_all": float(((p - y) ** 2).mean()),
             "pred_mean_stable": float(p[sel].mean()), "obs_rate_stable": float(y[sel].mean())}
out["cv_hazard_model"] = cv

json.dump(out, open(os.path.join(RES, "summary2.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "late_stable"}, indent=1))
print(json.dumps(out["late_stable"], indent=1))
for r in late_stable[:25]:
    print(r)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
# (a) survival curves
for k in (5, 7, 9):
    E, age = feats(k); f = fco_for(k, E)
    Ks = list(range(k, 17))
    ax[0].plot(Ks, [1 - ((f > 0) & (f <= K)).mean() for K in Ks], marker="o", ms=3, label=f"accepted at k={k}: all")
    s = age >= 2
    ax[0].plot(Ks, [1 - ((f > 0) & (f <= K))[s].mean() for K in Ks], ls="--", marker="s", ms=3, label=f"k={k}: stable (age>=2)")
ax[0].axvline(10.5, color="grey", lw=0.8); ax[0].text(10.6, 0.05, "sampled\norders", fontsize=8)
ax[0].set_xlabel("largest order K checked"); ax[0].set_ylabel("fraction still unrefuted"); ax[0].set_title("(a) survival of k-sharp conjectures")
ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1.02)
# (b) hazard vs age
for k in (6, 7, 8, 9):
    a = [x for x in sorted(haz[k]) if haz[k][x]["n"] >= 20]
    ax[1].plot(a, [haz[k][x]["hazard"] for x in a], marker="o", ms=3, label=f"k={k}")
    ax[1].axhline(1 / k, ls=":", lw=0.7, color=ax[1].lines[-1].get_color())
ax[1].set_ylim(-0.02, 1.02); ax[1].set_xlabel("record age a at universe k (orders since bound last moved)")
ax[1].set_ylabel("P(refuted at order k+1)"); ax[1].set_title("(b) one-step hazard, cells n>=20\n(dotted: exchangeable-record null 1/k)"); ax[1].legend(fontsize=7)
# (c) false acceptance of stable conjectures vs k
for a in (0, 1, 2, 3):
    ks = [k for k in range(4, 11) if a in stab[k]]
    ax[2].plot(ks, [stab[k][a]["frac_refuted_any_observed"] for k in ks], marker="o", ms=3, label=f"age >= {a}")
ax[2].set_yscale("log"); ax[2].set_xlabel("universe k (all connected graphs, 2..k vertices)")
ax[2].set_ylabel("fraction refuted (observed, orders <=16)"); ax[2].set_title("(c) false acceptance vs universe size"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_survival_hazard.png"), dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
dc = out["drift_census_orders_8_9_10"]
labs = ["strictly\nincreasing", "constant", "non-increasing\n(not const.)", "other"]
ns = [dc["strictly_increasing"], dc["constant"], dc["nonincreasing_not_constant"], dc["other"]]
ax[0].bar(labs, ns, color=["#c0504d", "#4f81bd", "#9bbb59", "#999999"])
for x, (nn, r) in enumerate(zip(ns, [dc["frac_refuted_beyond10_strictly_increasing"], dc["frac_refuted_beyond10_constant"], dc["frac_refuted_beyond10_nonincreasing"], None])):
    ax[0].text(x, nn, f"n={nn}" + (f"\n{100*r:.1f}% refuted" if r is not None else ""), ha="center", va="bottom", fontsize=8)
ax[0].set_ylabel("templates"); ax[0].set_title("(a) shape of per-order maxima M(8..10)"); ax[0].set_ylim(0, max(ns) * 1.25)
fp = out["falsifier_power_k10"]
ax[1].bar(["all refuted\n(k=10)", "stable refuted\n(age>=2)"], [fp["by_structured_families"] / fp["refuted_total"], fp["stable_by_structured"] / fp["stable_refuted_total"]], width=0.35, label="structured families", align="edge")
ax[1].bar(["all refuted\n(k=10)", "stable refuted\n(age>=2)"], [fp["by_gnp_sample"] / fp["refuted_total"], fp["stable_by_gnp"] / fp["stable_refuted_total"]], width=-0.35, label="G(n,p) sample", align="edge")
ax[1].set_ylabel("fraction of refuted conjectures caught"); ax[1].set_title("(b) falsifier coverage, orders 11-16"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_drift_falsifiers.png"), dpi=150)
print("saved figure")
