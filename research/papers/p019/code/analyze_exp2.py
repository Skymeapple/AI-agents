"""Analysis of E2/E3/E4 outputs: tables (JSON + printed), figures, world-level bootstrap CIs."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rows = json.load(open(os.path.join(OUT, "exp2_rows.json")))
sweeps = {t: json.load(open(os.path.join(OUT, f"exp2_sweep_{t}.json"))) for t in ("noiseless", "noisy")}
depth = json.load(open(os.path.join(OUT, "exp2_depth_rows.json")))
rng = np.random.default_rng(7)
Q, ALPHA, B = 0.1, 0.05, 2000

world = np.array([r["world"] for r in rows])
true = np.array([r["true"] for r in rows])
selfp = np.array([r["self"] for r in rows])
rep = np.array([r["replicate"] for r in rows])
env = np.array([r["env"] for r in rows])
pint = np.array([r["p_int"] for r in rows]); pint2 = np.array([r["p_int2"] for r in rows])
dist = np.array([r["dist"] for r in rows])
rev = np.array([r["reverse"] for r in rows]); wsign = np.array([r["wrong_sign"] for r in rows])
cc = np.array([r["common_cause"] for r in rows]); lat = np.array([r["latent_shared"] for r in rows])
uw = np.unique(world); nW = len(uw)
widx = {w: np.nonzero(world == w)[0] for w in uw}


def bh(p, q):
    m = len(p); o = np.argsort(p); thr = q * np.arange(1, m + 1) / m
    ok = np.nonzero(p[o] <= thr)[0]
    acc = np.zeros(m, bool)
    if len(ok):
        acc[o[:ok.max() + 1]] = True
    return acc


def per_world_bh(p, mask, q):
    acc = np.zeros(len(p), bool)
    for w, ix in widx.items():
        ix2 = ix[mask[ix]]
        if len(ix2):
            acc[ix2] = bh(p[ix2], q)
    return acc


def stat(acc, sub=None):
    """FDR (pooled ratio over worlds), true discoveries per world, world-bootstrap 95% CI for FDR."""
    sub = np.ones(len(acc), bool) if sub is None else sub
    a = acc & sub
    fd = np.array([(a[widx[w]] & ~true[widx[w]]).sum() for w in uw])
    nd = np.array([a[widx[w]].sum() for w in uw])
    td = nd - fd
    f = fd.sum() / max(nd.sum(), 1)
    bs = []
    for _ in range(B):
        s = rng.integers(0, nW, nW)
        bs.append(fd[s].sum() / max(nd[s].sum(), 1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(fdr=float(f), lo=float(lo), hi=float(hi), disc_per_world=float(nd.mean()), true_per_world=float(td.mean()))


pipes = {
    "self-verification": selfp,
    "replicate (fresh data, same env)": selfp & rep,
}
for K in range(1, 5):
    pipes[f"triangulate across {K} env(s)"] = selfp & env[:, :K].all(1)
pipes["1 intervention, alpha=0.05"] = selfp & (pint < ALPHA)
pipes["2 interventions, alpha=0.05 each"] = selfp & (pint < ALPHA) & (pint2 < ALPHA)
pipes["1 intervention + BH(q=0.1) per world"] = per_world_bh(pint, selfp, Q)
pipes["env2 screen -> intervention + BH(q=0.1)"] = per_world_bh(pint, selfp & env[:, :2].all(1), Q)
res = {"n_worlds": int(nW), "n_candidates": int(len(rows)), "pi1_candidates": float(true.mean()),
       "pipelines": {k: stat(v) for k, v in pipes.items()}}
# experiments used per world by each intervention pipeline
res["interventions_per_world"] = {"all candidates": float(selfp.sum() / nW),
                                  "after env2 screen": float((selfp & env[:, :2].all(1)).sum() / nW)}

# ---- false-candidate anatomy ----
false = ~true
cls = np.full(len(rows), "true", dtype=object)
cls[false & (rev | wsign)] = "reverse/wrong-sign"
cls[false & ~(rev | wsign) & cc] = "observed common cause"
cls[false & ~(rev | wsign) & ~cc & lat] = "latent confounding only"
cls[false & ~(rev | wsign) & ~cc & ~lat] = "other (indirect)"
anat = {}
for c in ["true", "reverse/wrong-sign", "observed common cause", "latent confounding only", "other (indirect)"]:
    m = cls == c
    anat[c] = dict(share=float(m.mean()), n=int(m.sum()),
                   pass_env_K=[float(env[m, :K].all(1).mean()) if m.sum() else None for K in range(1, 5)],
                   pass_replicate=float(rep[m].mean()) if m.sum() else None,
                   pass_intervention=float((pint[m] < ALPHA).mean()) if m.sum() else None)
res["anatomy"] = anat

# ---- E3: predicting FDR of K-environment triangulation (calibrate on first half of worlds) ----
cal = np.isin(world, uw[: nW // 2]); hold = ~cal
shared = false & (cls != "latent confounding only") & (cls != "other (indirect)")
indep = false & ~shared


def predict(maskcal):
    pi1 = true[maskcal & selfp].mean(); pi0 = 1 - pi1
    p1 = env[maskcal & selfp & true, 0].mean()
    f = env[maskcal & selfp & false, 0].mean()
    piS = shared[maskcal & selfp].mean(); piI = indep[maskcal & selfp].mean()
    fS = env[maskcal & selfp & shared, 0].mean(); fI = env[maskcal & selfp & indep, 0].mean()
    naive, decomp, mixture = [], [], []
    for K in range(1, 5):
        naive.append(pi0 * f ** K / (pi0 * f ** K + pi1 * p1 ** K))
        fp = piS * fS + piI * fI ** K  # shared biases pass every environment together
        decomp.append(fp / (fp + pi1 * p1 ** K))
        fm = piS * fS ** K + piI * fI ** K  # class-specific rates, independent across environments
        mixture.append(fm / (fm + pi1 * p1 ** K))
    return dict(pi1=pi1, p=p1, f=f, piS=piS, piI=piI, fS=fS, fI=fI, naive=naive, decomposed=decomp,
                class_mixture=mixture)


pred = predict(cal)
obs = [stat(selfp & env[:, :K].all(1), hold) for K in range(1, 5)]
res["E3_prediction"] = dict(calibration=pred, heldout_observed=obs)

# ---- novelty bins ----
bins = {"near (dist=2)": dist == 2, "mid (dist=3)": dist == 3, "far (dist>=4 or none)": dist >= 4}
nov = {}
for b, m in bins.items():
    pi1 = true[m & selfp].mean()
    d = dict(share=float(m.mean()), pi1=float(pi1))
    for k in ["self-verification", "triangulate across 4 env(s)", "1 intervention, alpha=0.05",
              "2 interventions, alpha=0.05 each", "1 intervention + BH(q=0.1) per world"]:
        d[k] = stat(pipes[k], m)
    # Theorem-2 prediction of interventions needed for FDR<=0.1, using pooled single-intervention
    pw = (pint[selfp & true] < ALPHA).mean(); fa = (pint[selfp & false] < ALPHA).mean()
    lam = pw / fa
    odds = pi1 / (1 - pi1)
    d["K_star_interventions"] = int(max(0, np.ceil(np.log((1 - Q) / (Q * odds)) / np.log(lam))))
    d["lambda_intervention"] = float(lam)
    d["pred_fdr_K1"] = float((1 - pi1) * fa / ((1 - pi1) * fa + pi1 * pw))
    d["pred_fdr_K2"] = float((1 - pi1) * fa ** 2 / ((1 - pi1) * fa ** 2 + pi1 * pw ** 2))
    nov[b] = d
res["novelty"] = nov

# ---- sample-size sweeps ----
res["sweep"] = {}
for tag, sweep in sweeps.items():
    sw = {}
    for n, rs in sweep.items():
        t = np.array([r["true"] for r in rs]); s_ = np.array([r["self"] for r in rs])
        rp = np.array([r["replicate"] for r in rs]); e = np.array([r["env"] for r in rs])
        pi_ = np.array([r["p_int"] for r in rs])
        def f_(a):
            return float((a & ~t).sum() / max(a.sum(), 1)), float((a & t).sum() / max(t.sum(), 1)), float(a.sum() / len(set(r["world"] for r in rs)))
        sw[n] = {"self": f_(s_), "replicate": f_(s_ & rp), "env4": f_(s_ & e.all(1)),
                 "intervene": f_(s_ & (pi_ < ALPHA)), "pi1": float(t.mean())}
    res["sweep"][tag] = sw

# ---- depth-of-search (E5): originality via generation depth; Theorem-2 K* prediction ----
dw = np.array([r["world"] for r in depth]); duw = np.unique(dw)
dcal = np.isin(dw, duw[: len(duw) // 2]); dhold = ~dcal
drank = np.array([r["rank"] for r in depth]); dtrue = np.array([r["true"] for r in depth])
dself = np.array([r["self"] for r in depth])
dP = np.array([[r["p_int"], r["p_int2"], r["p_int3"]] for r in depth]) < ALPHA
dbins = {"ranks 1-40": drank < 40, "ranks 41-100": (drank >= 40) & (drank < 100), "ranks 101-200": drank >= 100}
dep = {}
for b, m in dbins.items():
    mc = m & dcal & dself; mh = m & dhold & dself
    pi1c = dtrue[mc].mean(); pw = dP[mc & dtrue, 0].mean(); fa = dP[mc & ~dtrue, 0].mean()
    lam = pw / fa; odds = pi1c / (1 - pi1c)
    kst = int(max(0, np.ceil(np.log((1 - Q) / (Q * odds)) / np.log(lam))))
    d = dict(pi1_cal=float(pi1c), pi1_hold=float(dtrue[mh].mean()), power_cal=float(pw), falsepass_cal=float(fa),
             lambda_cal=float(lam), K_star_pred=kst, n_hold=int(mh.sum()))
    for K in (1, 2, 3):
        acc = dP[mh, :K].all(1)
        d[f"pred_fdr_K{K}"] = float((1 - pi1c) * fa ** K / ((1 - pi1c) * fa ** K + pi1c * pw ** K))
        d[f"obs_fdr_K{K}"] = float((acc & ~dtrue[mh]).sum() / max(acc.sum(), 1))
        d[f"obs_true_K{K}"] = int((acc & dtrue[mh]).sum())
    smallest = next((K for K in (1, 2, 3) if d[f"obs_fdr_K{K}"] <= Q), None)
    d["K_smallest_obs_meeting_q"] = smallest
    dep[b] = d
res["depth"] = dep
json.dump(res, open(os.path.join(OUT, "exp2_summary.json"), "w"), indent=1)

# ---- print ----
print("candidates", len(rows), "worlds", nW, "pi1", round(true.mean(), 3))
for k, v in res["pipelines"].items():
    print(f"{k:45s} FDR {v['fdr']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]  disc/world {v['disc_per_world']:.2f} true/world {v['true_per_world']:.2f}")
print("interventions/world", res["interventions_per_world"])
for c, v in anat.items():
    print(c, v)
print("E3 calibration", {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in pred.items()})
print("E3 observed", [(round(o["fdr"], 3), round(o["lo"], 3), round(o["hi"], 3)) for o in obs])
for b, d in nov.items():
    print(b, "share %.2f pi1 %.3f" % (d["share"], d["pi1"]), "K*", d["K_star_interventions"],
          "pred K1 %.3f K2 %.3f" % (d["pred_fdr_K1"], d["pred_fdr_K2"]),
          {k: round(v["fdr"], 3) for k, v in d.items() if isinstance(v, dict)})
for tag, sw in res["sweep"].items():
    for n, d in sw.items():
        print(tag, "n", n, {k: (np.round(v, 3).tolist() if isinstance(v, tuple) else round(v, 3)) for k, v in d.items()})
for b, d in dep.items():
    print(b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})

# ---- figures ----
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (tag, sw) in zip(axs, res["sweep"].items()):
  ns = sorted(sw, key=int); x = [int(n) for n in ns]
  for k, c, lab in [("self", "#d62728", "self-verification"), ("replicate", "#ff7f0e", "fresh replication, same env"),
                  ("env4", "#1f77b4", "4 environments (resampled confounding)"), ("intervene", "#2ca02c", "1 intervention (n=300)")]:
      ax.plot(x, [sw[n][k][0] for n in ns], "o-", color=c, label=lab)
  ax.set_xscale("log"); ax.set_xlabel("observational units per dataset")
  ax.set_ylim(0, 1); ax.set_title(f"SEM worlds ({tag} measurement): FDR vs n", fontsize=9)
axs[0].set_ylabel("FDR of accepted original claims"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_sem_sweep.png"), dpi=150)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
K = np.arange(1, 5)
ax.errorbar(K, [o["fdr"] for o in obs], yerr=[[o["fdr"] - o["lo"] for o in obs], [o["hi"] - o["fdr"] for o in obs]],
            fmt="ko", label="held-out observed (95% world-bootstrap CI)")
ax.plot(K, pred["naive"], "s--", color="#9467bd", label="naive independent-bias prediction")
ax.plot(K, pred["decomposed"], "^-", color="#1f77b4", label="decomposition: shared class perfectly correlated")
ax.plot(K, pred["class_mixture"], "v-", color="#ff7f0e", label="decomposition: class-specific rates")
ax.set_xticks(K); ax.set_xlabel("number of environments K"); ax.set_ylabel("FDR"); ax.set_ylim(0, 1)
ax.legend(fontsize=7); ax.set_title("E3: predicting triangulation FDR from calibration worlds", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_e3_prediction.png"), dpi=150)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
bl = list(nov); xx = np.arange(len(bl)); wdt = 0.2
for o, (k, c) in enumerate([("self-verification", "#d62728"), ("1 intervention, alpha=0.05", "#2ca02c"),
                            ("2 interventions, alpha=0.05 each", "#8c564b"), ("1 intervention + BH(q=0.1) per world", "#17becf")]):
    ax.bar(xx + (o - 1.5) * wdt, [nov[b][k]["fdr"] for b in bl], wdt, color=c, label=k)
ax.set_xticks(xx); ax.set_xticklabels([f"{b}\npi1={nov[b]['pi1']:.2f}" for b in bl], fontsize=7)
ax.axhline(Q, color="k", ls=":", lw=0.8); ax.set_ylabel("FDR"); ax.legend(fontsize=6.5)
ax.set_title("Originality tax: FDR by distance from the literature graph", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_novelty.png"), dpi=150)

# ---- share of false candidates whose bias is shared across observational environments, by novelty bin ----
with open(os.path.join(OUT, "novelty_bias_composition.txt"), "w") as fh:
    for b, m in bins.items():
        mm = m & false
        line = f"{b}: n_false={int(mm.sum())} shared-bias share among false {np.mean((rev | wsign | cc)[mm]):.3f}"
        print(line); fh.write(line + "\n")
