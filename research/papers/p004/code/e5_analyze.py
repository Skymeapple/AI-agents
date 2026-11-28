"""Analyze E5 (tiny transformer): storage law, multipliers, invariance, coverage."""
import json, os, numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import e7_threshold_storage as T
from e5_tinylm_retention import make_world

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = json.load(open(os.path.join(OUT, "e5_tinylm.json")))
runs, cfg = R["runs"], R["config"]
V, NT, NF = cfg["V"], cfg["T"], cfg["NP"] * cfg["NA"]
g = 1.0 / V
T.KCLS = V
rep = {}

def zipf_p(alpha):
    w = np.arange(1, NF + 1, dtype=float) ** (-alpha); return w / w.sum()

def plain_curve(alpha):
    Ns = sorted(set(x["N"] for x in runs if x["kind"] == "plain" and x["alpha"] == alpha))
    return np.array(Ns, float), {N: [x["acc_weighted"] for x in runs if x["kind"] == "plain" and x["alpha"] == alpha and x["N"] == N] for N in Ns}

def invert(Ns, curve, acc):
    c = np.maximum.accumulate(curve)
    if acc <= c[0] or acc >= c[-1]: return np.nan
    return float(np.exp(np.interp(acc, c, np.log(Ns))))

# ---- (1) storage law from plain runs ------------------------------------------
def nll(params, c, k, model, m=1):
    s = 1 / (1 + np.exp(-params[0]))
    S = 1 - (1 - s) ** c if model == "geom" else stats.binom.sf(m - 1, c, s)
    P = np.clip(g + (1 - g) * S, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(P) + (NT - k) * np.log(1 - P))

storage = []
for alpha in [1.0, 0.6]:
    for N in [2000, 4000, 8000, 16000]:
        rs = [x for x in runs if x["kind"] == "plain" and x["alpha"] == alpha and x["N"] == N]
        c = np.concatenate([np.array(x["counts"]) for x in rs]) * cfg["E"]   # exposures = occurrences x epochs
        k = np.concatenate([np.round(np.array(x["fact_acc"]) * NT) for x in rs])
        row = dict(alpha=alpha, N=N, n_facts=len(c))
        for name, model, m in [("geom", "geom", 1), ("thr2", "thr", 2), ("thr3", "thr", 3), ("thr4", "thr", 4), ("thr6", "thr", 6)]:
            best = min((optimize.minimize(nll, [x0], args=(c, k, model, m), method="Nelder-Mead") for x0 in [-4, -2, 0]), key=lambda o: o.fun)
            row[name] = dict(s=float(1 / (1 + np.exp(-best.x[0]))), nll=float(best.fun))
        # empirical recall by exposure count
        row["recall_by_occ"] = {int(o): float(np.mean(k[c == o * cfg["E"]] / NT)) for o in range(0, 9) if np.sum(c == o * cfg["E"]) >= 20}
        storage.append(row)
rep["storage"] = storage

# ---- (2) multipliers ------------------------------------------------------------
mult = []
rng = np.random.default_rng(0)
for alpha in [1.0, 0.6]:
    Ns, pc = plain_curve(alpha)
    mean_curve = np.array([np.mean(pc[N]) for N in Ns])
    p = zipf_p(alpha)
    for (kind, N0, r) in sorted(set((x["kind"], x["N"], x["r"]) for x in runs if x["kind"] != "plain" and x["alpha"] == alpha)):
        accs = [x["acc_weighted"] for x in runs if x["kind"] == kind and x["N"] == N0 and x["r"] == r and x["alpha"] == alpha]
        M = invert(Ns, mean_curve, np.mean(accs)) / N0
        boots = []
        for _ in range(2000):   # bootstrap over seeds of both the plain curve and the amplified runs
            cb = np.array([np.mean(rng.choice(pc[N], len(pc[N]))) for N in Ns])
            boots.append(invert(Ns, cb, np.mean(rng.choice(accs, len(accs)))) / N0)
        lo, hi = np.nanpercentile(boots, [5, 95])
        # coverage ceiling: every fact with >=1 human occurrence answered, others guessed
        cov = []
        for x in runs:
            if x["kind"] == kind and x["N"] == N0 and x["r"] == r and x["alpha"] == alpha:
                _, p = make_world(x["seed"], alpha)      # world_seed == seed in e5
                seen = np.array(x["counts"]) > 0
                cov.append(np.sum(p * seen) + g * np.sum(p * ~seen))
        mult.append(dict(alpha=alpha, kind=kind, N0=N0, r=r, acc=float(np.mean(accs)), acc_sd=float(np.std(accs, ddof=1)),
                         M=M, M_ci90=[float(lo), float(hi)], coverage_ceiling=float(np.mean(cov)),
                         plain_acc_N0=float(np.mean(pc[N0])) if N0 in pc else None, bound_1_plus_r=1 + r))
rep["multipliers"] = mult

# ---- (3) model predictions for M, using storage fits at N0 ---------------------
preds = []
for row in mult:
    if row["kind"] != "paraphrase": continue
    st = [s for s in storage if s["alpha"] == row["alpha"] and s["N"] == row["N0"]][0]
    p = zipf_p(row["alpha"]); E_ = cfg["E"]
    out = dict(alpha=row["alpha"], N0=row["N0"], r=row["r"], M_measured=row["M"], M_ci90=row["M_ci90"])
    for name, m in [("geom", 1), ("thr2", 2), ("thr3", 3), ("thr4", 4), ("thr6", 6)]:
        s = st[name]["s"]
        # each corpus occurrence = E exposures; paraphrase copies are also seen E times.
        # exposures per human occurrence: E*(1+r) Bernoulli(s) trials; plain: E trials.
        def acc_plain(N):
            return 1 - (1 - g) * np.sum(p * T.p_unstored_amp(N * p, s, m, E_ - 1))
        acc_amp = 1 - (1 - g) * np.sum(p * T.p_unstored_amp(row["N0"] * p, s, m, E_ * (1 + row["r"]) - 1))
        f = lambda lg: acc_plain(np.exp(lg)) - acc_amp
        try: Neq = float(np.exp(optimize.brentq(f, np.log(1), np.log(1e9))))
        except ValueError: Neq = np.nan
        out[name] = Neq / row["N0"]
    preds.append(out)
rep["predictions"] = preds
json.dump(rep, open(os.path.join(OUT, "e5_analysis.json"), "w"), indent=1)

for s in storage:
    print(s["alpha"], s["N"], {k: (round(v["s"], 4), round(v["nll"])) for k, v in s.items() if isinstance(v, dict) and "nll" in v})
    print("   recall by #occurrences:", {k: round(v, 3) for k, v in s["recall_by_occ"].items()})
for m in mult: print(m)
for p_ in preds: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in p_.items()})

# ---- figure -----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for alpha, mk in [(1.0, "o"), (0.6, "s")]:
    Ns, pc = plain_curve(alpha)
    ax[0].plot(Ns, [np.mean(pc[N]) for N in Ns], mk + "-", label=f"plain, alpha={alpha}")
    for kind, ls in [("paraphrase", "--"), ("repetition", ":")]:
        pts = sorted([(m["N0"] * (1 + m["r"]), m["acc"]) for m in mult if m["alpha"] == alpha and m["kind"] == kind and m["N0"] == 2000])
        if pts: ax[0].plot(*zip(*pts), mk + ls, mfc="none", label=f"{kind} of N0=2000 (x = rows)")
ax[0].set_xscale("log"); ax[0].set_xlabel("corpus sentences (N, or N0(1+r))"); ax[0].set_ylabel("Zipf-weighted recall"); ax[0].legend(fontsize=6)
ax[0].set_title("(a) tiny transformer: recall vs corpus size")
for s in storage:
    if s["alpha"] != 1.0: continue
    o = sorted(s["recall_by_occ"]); ax[1].plot(o, [s["recall_by_occ"][k] for k in o], "o-", label=f"N={s['N']}")
oo = np.arange(0, 9); st = [x for x in storage if x["alpha"] == 1.0 and x["N"] == 4000][0]
ax[1].plot(oo, g + (1 - g) * (1 - (1 - st["geom"]["s"]) ** (oo * cfg["E"])), "k--", label="geometric fit (N=4000)")
ax[1].set_xlabel("human occurrences of the fact"); ax[1].set_ylabel("recall"); ax[1].legend(fontsize=6); ax[1].set_title("(b) recall vs exposure count (alpha=1)")
pp = [x for x in preds]
lab = [f"a{x['alpha']}\nN0={x['N0']}\nr={x['r']}" for x in pp]
xs = np.arange(len(pp))
ax[2].errorbar(xs, [x["M_measured"] for x in pp], yerr=np.array([[x["M_measured"] - x["M_ci90"][0], x["M_ci90"][1] - x["M_measured"]] for x in pp]).T, fmt="ko", label="measured (90% CI)")
ax[2].plot(xs, [x["geom"] for x in pp], "b^", label="geometric (Prop. 2)")
best = min(["thr2", "thr3", "thr4", "thr6"], key=lambda n: sum(s[n]["nll"] for s in storage))
ax[2].plot(xs, [x[best] for x in pp], "rs", mfc="none", label=f"threshold ({best})")
ax[2].set_xticks(xs); ax[2].set_xticklabels(lab, fontsize=6); ax[2].set_ylabel("multiplier M"); ax[2].legend(fontsize=7)
ax[2].set_title("(c) paraphrase multiplier: measured vs storage models")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_tinylm.png"), dpi=150)
