"""Main empirical analysis: pass-rate metrics vs Delegation-Value on METR TH1.1 runs.

Usage:  python3 code/analysis.py  [--boot 300]
Writes results/*.json, results/*.csv and results/fig_*.png
"""
import argparse, json, os, collections, itertools
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from dv import net_saving, fit_logistic, horizon, stop_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data/eval-analysis-public/reports/time-horizon-1-1/data/raw/runs.jsonl")
DATES = os.path.join(ROOT, "data/eval-analysis-public/data/external/release_dates.yaml")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- load
rows = [json.loads(l) for l in open(DATA)]
rows = [r for r in rows if r["alias"] != "human"]
tasks = sorted({r["task_id"] for r in rows})
tix = {t: i for i, t in enumerate(tasks)}
T = len(tasks)
info = {}
for r in rows:
    info[r["task_id"]] = (r["human_minutes"], r["human_cost"], r["task_family"],
                          r["task_source"], r["invsqrt_task_weight"])
hmin = np.array([info[t][0] for t in tasks])
Hcost = np.array([info[t][1] for t in tasks])          # USD at METR's $143.6/h
WAGE_PER_MIN = float(np.median(Hcost / hmin))           # recovered wage
family = np.array([info[t][2] for t in tasks])
source = np.array([info[t][3] for t in tasks])
fams = sorted(set(family))
fidx = np.array([fams.index(f) for f in family])

models = sorted({r["alias"] for r in rows})
M = len(models)
succ = collections.defaultdict(list)
toks = collections.defaultdict(list)
gcost = collections.defaultdict(list)
for r in rows:
    k = (r["alias"], tix[r["task_id"]])
    succ[k].append(r["score_binarized"])
    if r["tokens_count"]:
        toks[k].append(r["tokens_count"])
    if r["generation_cost"]:
        gcost[k].append(r["generation_cost"])

P = np.full((M, T), np.nan)       # empirical success prob
TOK = np.zeros((M, T)); GC = np.full((M, T), np.nan)
NRUN = np.zeros((M, T), int)
for mi, m in enumerate(models):
    for i in range(T):
        k = (m, i)
        if k in succ:
            P[mi, i] = np.mean(succ[k]); NRUN[mi, i] = len(succ[k])
            TOK[mi, i] = np.mean(toks[k]) if toks[k] else 0.0
            if gcost[k]:
                GC[mi, i] = np.mean(gcost[k])
# tasks missing for a model are treated as failures (p=0): conservative, few cases
missing = int(np.isnan(P).sum())
P = np.nan_to_num(P, nan=0.0)

dates = yaml.safe_load(open(DATES))["date"]
rdate = {}
for m in models:
    d = dates.get(m) or dates.get(m.replace(" (Inspect)", ""))
    rdate[m] = str(d) if d else None

# ---------------------------------------------------------------- params
BASE = dict(rho=0.2, r0=2.0, s=1.0, delta=1.0, price=10.0)   # price: USD per 1M tokens
RHOS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def costs(mi, prm, idx=None):
    idx = np.arange(T) if idx is None else idx
    V = WAGE_PER_MIN * (prm["rho"] * hmin[idx] + prm["r0"])
    K = np.where(np.isnan(GC[mi, idx]), TOK[mi, idx] * prm["price"] / 1e6, GC[mi, idx])
    if prm["price"] == 0:
        K = np.zeros_like(K)
    return V, K


def crossfit_phat(mi, idx, fold_of):
    """Predict p for tasks in idx from a logistic in log2(minutes) fit on other folds."""
    x = np.log2(hmin[idx]); y = P[mi, idx]
    ph = np.zeros(len(idx))
    for f in np.unique(fold_of):
        tr = fold_of != f; te = ~tr
        a, b = fit_logistic(x[tr], y[tr])
        ph[te] = 1 / (1 + np.exp(-(a + b * x[te])))
    return ph


def evaluate(idx, prm, omega=None, phats=None, phat_fam=None):
    """Return dict of metrics per model for task multiset idx.
    phats: list (over fold seeds) of arrays [M, len(idx)] of cross-fit length-based p-hat.
    phat_fam: array [M, len(idx)] of leave-one-task-out family-based p-hat (optional)."""
    omega = np.ones(len(idx)) if omega is None else omega
    H = Hcost[idx]; tot = np.sum(omega * H)
    out = {}
    for mi, m in enumerate(models):
        p = P[mi, idx]
        V, K = costs(mi, prm, idx)
        S = net_saving(p, H, V, K, prm["s"], prm["delta"])
        dvi, fdi = [], []
        for ph in phats:
            Sh = net_saving(ph[mi], H, V, K, prm["s"], prm["delta"])
            dvi.append(np.sum(omega * S * (Sh > 0)) / tot)
            fdi.append(np.sum(omega * H * (Sh > 0)) / tot)
        d = dict(
            pass_eq=float(np.sum(omega * p) / np.sum(omega)),
            pass_val=float(np.sum(omega * H * p) / tot),
            dv_all=float(np.sum(omega * S) / tot),
            dv_oracle=float(np.sum(omega * np.clip(S, 0, None)) / tot),
            dv_informed=float(np.mean(dvi)),
            frac_deleg_informed=float(np.mean(fdi)),
        )
        if phat_fam is not None:
            Sf = net_saving(phat_fam[mi], H, V, K, prm["s"], prm["delta"])
            d["dv_family"] = float(np.sum(omega * S * (Sf > 0)) / tot)
        out[m] = d
    return out


def folds_by_family(idx, k=5, seed=0):
    r = np.random.default_rng(seed)
    fam_here = np.unique(fidx[idx])
    perm = r.permutation(fam_here)
    fmap = {f: j % k for j, f in enumerate(perm)}
    return np.array([fmap[f] for f in fidx[idx]])


ALL = np.arange(T)
NSEED = 10
PHATS = [np.array([crossfit_phat(mi, ALL, folds_by_family(ALL, seed=sd)) for mi in range(M)]) for sd in range(NSEED)]
# family-experience policy: p-hat = mean success of the model on the *other* tasks of the same family;
# singleton families fall back to the length-based cross-fit estimate (seed 0)
PFAM = np.zeros((M, T))
for mi in range(M):
    for i in range(T):
        same = np.where((fidx == fidx[i]) & (np.arange(T) != i))[0]
        PFAM[mi, i] = P[mi, same].mean() if len(same) else PHATS[0][mi, i]


def ev_all(prm, omega=None):
    return evaluate(ALL, prm, omega=omega, phats=PHATS, phat_fam=PFAM)


# ---------------------------------------------------------------- horizons
hz = {}
for mi, m in enumerate(models):
    a, b = fit_logistic(np.log2(hmin), P[mi])
    hz[m] = dict(h50=float(horizon(a, b, 0.5)), h80=float(horizon(a, b, 0.8)), a=float(a), b=float(b))

# ---------------------------------------------------------------- main table
main = ev_all(BASE)
for m in models:
    main[m].update(hz[m]); main[m]["release"] = rdate[m]

# rho sweep (oracle and informed), r0 kept at base
sweep = {}
for rho in RHOS:
    prm = dict(BASE, rho=rho)
    sweep[rho] = ev_all(prm)

# pure-theory case: r0=0, price=0, s=1 -> oracle DV equals stop-loss transform
theory = {}
grid = np.linspace(0, 1, 101)
for mi, m in enumerate(models):
    theory[m] = stop_loss(P[mi], Hcost, grid)
# check identity numerically
chk = ev_all(dict(rho=0.3, r0=0.0, s=1.0, delta=1.0, price=0.0))
ident_err = max(abs(chk[m]["dv_oracle"] - theory[m][30]) for m in models)

# pairwise icx-order analysis on value-weighted p distribution
pairs = []
for (i, a), (j, b) in itertools.combinations(enumerate(models), 2):
    d = theory[a] - theory[b]
    if np.all(d >= -1e-12) or np.all(d <= 1e-12):
        cross = False
    else:
        cross = True
    # ranking by equal-weight pass rate vs oracle DV at each rho
    pe = main[a]["pass_eq"] - main[b]["pass_eq"]
    flips_eq = [float(r) for r, dd in zip(grid, d) if np.sign(dd) != np.sign(pe) and abs(dd) > 1e-9]
    pairs.append(dict(a=a, b=b, icx_cross=cross, sign_mean=float(np.sign(d[0])),
                      rho_where_val_rank_flips=[float(grid[k]) for k in range(1, len(grid)) if np.sign(d[k]) != np.sign(d[0]) and abs(d[k]) > 1e-9][:1],
                      n_rho_disagree_with_passeq=len(flips_eq)))
n_pairs = len(pairs)
n_cross = sum(p["icx_cross"] for p in pairs)

# rank agreement across rho
rank_stats = {}
pe = np.array([main[m]["pass_eq"] for m in models])
pv = np.array([main[m]["pass_val"] for m in models])
h50 = np.array([main[m]["h50"] for m in models])
for rho in RHOS:
    dvo = np.array([sweep[rho][m]["dv_oracle"] for m in models])
    dvi = np.array([sweep[rho][m]["dv_informed"] for m in models])
    rank_stats[rho] = dict(
        tau_passeq_dvinf=float(kendalltau(pe, dvi)[0]),
        tau_passval_dvinf=float(kendalltau(pv, dvi)[0]),
        tau_h50_dvinf=float(kendalltau(h50, dvi)[0]),
        tau_passeq_dvor=float(kendalltau(pe, dvo)[0]),
        top_passeq=models[int(np.argmax(pe))], top_dvinf=models[int(np.argmax(dvi))],
        top_dvor=models[int(np.argmax(dvo))],
        n_models_dvinf_positive=int(np.sum(dvi > 0)),
    )

# ---------------------------------------------------------------- sensitivity grid
sens = []
for r0 in [0.0, 2.0, 10.0]:
    for s, delta in [(1.0, 1.0), (0.9, 2.0), (0.8, 3.0), (0.8, 5.0)]:
        for price in [0.0, 10.0, 50.0]:
            for rho in [0.1, 0.2, 0.4]:
                prm = dict(rho=rho, r0=r0, s=s, delta=delta, price=price)
                ev = ev_all(prm)
                dvi = np.array([ev[m]["dv_informed"] for m in models])
                dvo = np.array([ev[m]["dv_oracle"] for m in models])
                sens.append(dict(prm, best_inf=float(dvi.max()), best_or=float(dvo.max()),
                                 top_inf=models[int(np.argmax(dvi))],
                                 tau_passeq=float(kendalltau(pe, dvi)[0]),
                                 n_pos=int(np.sum(dvi > 0))))

# portfolio reweighting: METR invsqrt (diversity) weights
omega_div = np.array([info[t][4] for t in tasks]); omega_div = omega_div / omega_div.mean()
main_div = ev_all(BASE, omega=omega_div)

# ---------------------------------------------------------------- bootstrap (family clusters)
boot = {m: collections.defaultdict(list) for m in models}
boot_tau = []
boot_top = collections.Counter()
for b in range(args.boot):
    fsel = rng.choice(len(fams), len(fams), replace=True)
    idx_parts, fold_parts = [], []
    for c, f in enumerate(fsel):
        ii = np.where(fidx == f)[0]
        idx_parts.append(ii); fold_parts.append(np.full(len(ii), c % 5))
    idx = np.concatenate(idx_parts); fo = np.concatenate(fold_parts)
    fo = rng.permutation(5)[fo]
    ph = np.array([crossfit_phat(mi, idx, fo) for mi in range(M)])
    ev = evaluate(idx, BASE, phats=[ph], phat_fam=PFAM[:, idx])
    for m in models:
        for k, v in ev[m].items():
            boot[m][k].append(v)
    dvi = np.array([ev[m]["dv_informed"] for m in models])
    pe_b = np.array([ev[m]["pass_eq"] for m in models])
    boot_tau.append(kendalltau(pe_b, dvi)[0])
    boot_top[models[int(np.argmax(dvi))]] += 1
ci = {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot[m].items()} for m in models}

# ---------------------------------------------------------------- retries (failure concentration)
from math import comb


def allfail_unbiased(n, c, j):
    """Unbiased estimator of (1-p)^j from c successes in n iid runs (j<=n)."""
    return comb(n - c, j) / comb(n, j)


retry = {}
for mi, m in enumerate(models):
    num = den = 0.0; marg_num = marg_den = 0.0
    p3 = wsum = 0.0
    for i in range(T):
        o = succ.get((m, i), [])
        n = len(o)
        if n < 3:
            continue
        c = int(np.sum(o)); f = n - c
        if f > 0:
            num += f * c / (n - 1); den += f
        marg_num += c; marg_den += n
        p3 += Hcost[i] * (1 - allfail_unbiased(n, c, 3)); wsum += Hcost[i]
    pv_ = main[m]["pass_val"]
    retry[m] = dict(p_succ_after_fail=num / den if den else np.nan, p_marginal=marg_num / marg_den,
                    val_w_pass3_unbiased=p3 / wsum, val_w_pass3_pooled_iid=1 - (1 - pv_) ** 3,
                    val_w_pass1=pv_)


def dv_retry(mi, nmax, prm):
    """Oracle task selection; on each delegated task run up to nmax verified attempts
    (choose best nmax per task). Uses unbiased estimators of (1-p)^j."""
    tot = 0.0
    V_all, K_all = costs(mi, prm)
    for i in range(T):
        o = succ.get((models[mi], i), [])
        n = len(o)
        if n == 0:
            continue
        c = int(np.sum(o)); V = V_all[i]; K = K_all[i]
        best = 0.0
        for na in range(1, min(nmax, n) + 1):
            ps = 1 - allfail_unbiased(n, c, na)
            exp_att = sum(allfail_unbiased(n, c, j) for j in range(na))
            best = max(best, ps * Hcost[i] - exp_att * (V + K))
        tot += best
    return tot / Hcost.sum()


for mi, m in enumerate(models):
    retry[m]["dv_oracle_1"] = dv_retry(mi, 1, BASE)
    retry[m]["dv_oracle_3"] = dv_retry(mi, 3, BASE)

# ---------------------------------------------------------------- save
summary = dict(n_tasks=T, n_families=len(fams), n_models=M, n_runs=len(rows), missing_cells=missing,
               wage_per_hour=WAGE_PER_MIN * 60, base_params=BASE, total_portfolio_cost=float(Hcost.sum()),
               identity_check_max_abs_err=float(ident_err), n_pairs=n_pairs, n_pairs_icx_cross=n_cross,
               boot=args.boot, boot_tau_passeq_dvinf_ci=[float(np.percentile(boot_tau, 2.5)), float(np.percentile(boot_tau, 97.5))],
               boot_top_dvinf=dict(boot_top))
json.dump(dict(summary=summary, main=main, main_div=main_div, ci=ci, rank_stats={str(k): v for k, v in rank_stats.items()},
               sweep={str(k): v for k, v in sweep.items()}, pairs=pairs, retry=retry, sens=sens),
          open(os.path.join(RES, "results.json"), "w"), indent=1, default=float)

with open(os.path.join(RES, "main_table.csv"), "w") as f:
    cols = ["model", "release", "pass_eq", "pass_val", "h50", "h80", "dv_all", "dv_oracle", "dv_informed",
            "dv_informed_lo", "dv_informed_hi", "dv_family", "frac_deleg_informed", "dv_informed_divweights"]
    f.write(",".join(cols) + "\n")
    for m in sorted(models, key=lambda m: -main[m]["dv_informed"]):
        d = main[m]
        f.write(",".join(str(x) for x in [m, d["release"], round(d["pass_eq"], 4), round(d["pass_val"], 4),
                                          round(d["h50"], 1), round(d["h80"], 1), round(d["dv_all"], 4),
                                          round(d["dv_oracle"], 4), round(d["dv_informed"], 4),
                                          round(ci[m]["dv_informed"][0], 4), round(ci[m]["dv_informed"][1], 4), round(d["dv_family"], 4),
                                          round(d["frac_deleg_informed"], 4), round(main_div[m]["dv_informed"], 4)]) + "\n")

with open(os.path.join(RES, "sensitivity.csv"), "w") as f:
    keys = list(sens[0].keys()); f.write(",".join(keys) + "\n")
    for s_ in sens:
        f.write(",".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in s_.values()) + "\n")

with open(os.path.join(RES, "retry.csv"), "w") as f:
    keys = list(next(iter(retry.values())).keys()); f.write("model," + ",".join(keys) + "\n")
    for m in models:
        f.write(m + "," + ",".join(str(round(retry[m][k], 4)) for k in keys) + "\n")

# ---------------------------------------------------------------- figures
short = {m: m.replace(" (Inspect)", "") for m in models}
order = sorted(models, key=lambda m: main[m]["pass_eq"])
cmap = plt.get_cmap("viridis")
colors = {m: cmap(i / (M - 1)) for i, m in enumerate(order)}

# Fig 1: DV informed vs rho
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, key, ttl in [(axes[0], "dv_oracle", "Oracle delegation"), (axes[1], "dv_informed", "Length-informed delegation (cross-fit)")]:
    for m in order:
        ax.plot(RHOS, [sweep[r][m][key] for r in RHOS], color=colors[m], lw=1.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"verification cost ratio $\rho$ (review time / human time)")
    ax.set_title(ttl)
axes[0].set_ylabel("Delegation-Value Rate (share of portfolio labour cost)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(pe), max(pe)))
fig.colorbar(sm, ax=axes, label="equal-weight pass rate")
fig.savefig(os.path.join(RES, "fig_dv_vs_rho.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# Fig 2: pass rate vs DV informed (base)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for m in models:
    x = main[m]["pass_eq"]; y = main[m]["dv_informed"]; lo, hi = ci[m]["dv_informed"]
    ax.errorbar(x, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]], fmt="o", color=colors[m], ms=6, capsize=2)
    ax.annotate(short[m], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("equal-weight pass rate (benchmark-style score)")
ax.set_ylabel(r"DVR, length-informed ($\rho$=0.2, $r_0$=2 min)")
ax.axhline(0, color="k", lw=0.6)
fig.savefig(os.path.join(RES, "fig_pass_vs_dv.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# Fig 3: stop-loss curves for crossing pairs (top 6 by mean)
fig, ax = plt.subplots(figsize=(7.5, 5))
top = sorted(models, key=lambda m: -main[m]["pass_val"])[:6]
for m in top:
    ax.plot(grid, theory[m], label=short[m], lw=1.5)
ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\pi(\rho)=E_\mu[(P-\rho)_+]$ (oracle DVR, $r_0$=0, $K$=0)")
ax.legend(fontsize=8); ax.set_xlim(0, 1)
fig.savefig(os.path.join(RES, "fig_stoploss.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# Fig 4: over time
fig, ax = plt.subplots(figsize=(8, 4.8))
dm = [m for m in models if rdate[m]]
xs = np.array([np.datetime64(rdate[m]) for m in dm])
o = np.argsort(xs)
for key, lab in [("pass_eq", "equal-weight pass rate"), ("pass_val", "value-weighted pass rate"),
                 ("dv_oracle", "DVR oracle"), ("dv_informed", "DVR length-informed")]:
    ax.plot(xs[o], [main[dm[k]][key] for k in o], "o-", ms=3, label=lab)
ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=8); ax.set_ylabel("score / share of labour cost")
ax.set_xlabel("model release date")
fig.savefig(os.path.join(RES, "fig_time.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# Fig 5: retries
fig, ax = plt.subplots(figsize=(6.5, 5))
for m in models:
    ax.scatter(retry[m]["p_marginal"], retry[m]["p_succ_after_fail"], color=colors[m])
ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="independence")
ax.set_xlabel("marginal per-run success"); ax.set_ylabel("P(success | previous attempt on same task failed)")
ax.legend(); ax.set_xlim(0.35, 0.9); ax.set_ylim(0, 0.9)
fig.savefig(os.path.join(RES, "fig_retry.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

print(json.dumps(summary, indent=1))
for rho in RHOS:
    print(rho, rank_stats[rho])
