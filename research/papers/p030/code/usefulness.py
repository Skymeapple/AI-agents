"""Delegation-surplus ("usefulness") analysis of the public METR time-horizon runs.

Run:  python3 code/usefulness.py            (from the paper directory, ~2-4 min CPU)
Outputs: results/*.json, results/*.csv, results/*.png
All numbers in paper.md Sections 5-6 come from these outputs.
"""
import json, os, sys, collections, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data_cache/metr/reports/time-horizon-1-1/data/raw/runs.jsonl")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
SEED = 20261128
M_PRIOR = 2.0          # pseudo-count shrinking per-task history toward the length model
N_SPLITS = 20          # random half-splits of runs for the task-history information structure
N_BOOT = 300           # cluster bootstrap replicates (task families)
N_SPLITS_BOOT = 4
KGRID = np.round(np.linspace(0, 0.95, 20), 2)
THETA0 = dict(rho=0.2, h=5.0, c=0.05)   # reference deployment context
RHOS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
HS = [1.0, 2.0, 5.0, 20.0]
CS = [0.0, 0.05, 0.2]


# ----------------------------------------------------------------------------- data
def load():
    rows = [json.loads(l) for l in open(DATA)]
    rows = [r for r in rows if r["alias"] != "human"]
    models = sorted(set(r["alias"] for r in rows))
    out = {}
    for m in models:
        d = collections.defaultdict(list)
        meta = {}
        for r in rows:
            if r["alias"] == m:
                d[r["task_id"]].append(int(r["score_binarized"]))
                meta[r["task_id"]] = (r["human_minutes"], r["task_family"], r["task_source"])
        tids = sorted(d)
        out[m] = dict(
            tid=tids,
            t=np.array([meta[k][0] for k in tids]),
            fam=np.array([meta[k][1] for k in tids]),
            src=np.array([meta[k][2] for k in tids]),
            runs=[np.array(d[k]) for k in tids],
        )
    return out


# ----------------------------------------------------------------------------- model pieces
def fit_logistic(x, s, n, ridge=1e-2):
    """Binomial logistic regression of successes s out of n on [1, x]; Newton with small ridge."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(50):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        g = X.T @ (s - n * p) - ridge * b
        H = (X * (n * p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def crossfit_len(t, fam, s, n, nfold=5):
    """Leave-family-fold-out prediction of success probability from log2(human minutes)."""
    x = np.log2(t)
    ufam = sorted(set(fam))
    fold_of = {f: i % nfold for i, f in enumerate(ufam)}
    fold = np.array([fold_of[f] for f in fam])
    q = np.zeros(len(t))
    for k in range(nfold):
        te = fold == k
        if te.sum() == 0:
            continue
        b = fit_logistic(x[~te], s[~te], n[~te])
        q[te] = 1 / (1 + np.exp(-np.clip(b[0] + b[1] * x[te], -30, 30)))
    return q


def saving(q, y, rho, h, c):
    """Realised fractional labour saving of the policy that acts optimally on belief q,
    when the realised success frequency is y. Actions: do-it-yourself (0),
    verify (y-rho-c), blind accept (1-c-(1-y)h)."""
    ev = q - rho - c
    eb = 1 - c - (1 - q) * h
    act = np.where((ev <= 0) & (eb <= 0), 0, np.where(eb > ev, 2, 1))
    return np.where(act == 0, 0.0, np.where(act == 1, y - rho - c, 1 - c - (1 - y) * h))


def stoploss(q, y, w, k):
    """pi(k) = E_w[ 1{q>k} (y-k) ]  -- honest value of the threshold-k delegation policy."""
    return float(np.sum(w * (q > k) * (y - k)) / np.sum(w))


def identity_U(pi_fn, rho, h, c):
    """Proposition 1: usefulness from the stop-loss curve alone."""
    k1 = rho + c
    if h <= 1 + 1e-12:
        return pi_fn(c)                      # blind acceptance weakly dominates verification
    k2 = 1 - rho / (h - 1)
    if k2 >= k1:
        return pi_fn(k1) + (h - 1) * pi_fn(k2)
    return h * pi_fn(1 - (1 - c) / h)


def analyse(md, gamma, rng, n_splits, idx=None):
    """Return per-information-structure (q, y, w) triples for one model.
    idx: optional array of task indices (bootstrap resample, may repeat)."""
    t, fam, runs = md["t"], md["fam"], md["runs"]
    if idx is not None:
        t, fam = t[idx], fam[idx]
        runs = [runs[i] for i in idx]
    n = np.array([len(r) for r in runs], float)
    s = np.array([r.sum() for r in runs], float)
    y = s / n
    w = t ** gamma
    acc = float(y.mean())                          # headline: equal-task-weight success rate
    aw = float(np.sum(w * y) / np.sum(w))          # value-weighted accuracy
    qlen = crossfit_len(t, fam, s, n)
    info = {}
    info["none"] = [(np.full(len(t), aw), y, w)]
    info["length"] = [(qlen, y, w)]
    # task history: belief from half the runs (shrunk to the length model), evaluated on the other half
    trip = []
    for _ in range(n_splits):
        qa, ya, qb, yb = (np.zeros(len(t)) for _ in range(4))
        for i, r in enumerate(runs):
            if len(r) < 2:
                qa[i] = qb[i] = qlen[i]; ya[i] = yb[i] = y[i]; continue
            perm = rng.permutation(len(r)); hlf = len(r) // 2
            A, B = r[perm[:hlf]], r[perm[hlf:]]
            qa[i] = (A.sum() + M_PRIOR * qlen[i]) / (len(A) + M_PRIOR); ya[i] = B.mean()
            qb[i] = (B.sum() + M_PRIOR * qlen[i]) / (len(B) + M_PRIOR); yb[i] = A.mean()
        trip += [(qa, ya, w), (qb, yb, w)]
    info["task"] = trip
    # oracle: outcome of each run known in advance -> value is linear in y with q in {0,1}
    info["oracle"] = "oracle"
    return dict(acc=acc, aw=aw, y=y, w=w, info=info, qlen=qlen, t=t)


def U_of(an, I, rho, h, c):
    if I == "oracle":
        return an["aw"] * (1 - c)
    return float(np.mean([np.sum(w * saving(q, y, rho, h, c)) / np.sum(w) for q, y, w in an["info"][I]]))


def pi_of(an, I, k):
    if I == "oracle":
        return an["aw"] * (1 - k)
    return float(np.mean([stoploss(q, y, w, k) for q, y, w in an["info"][I]]))


def kendall(a, b):
    a, b = np.asarray(a), np.asarray(b)
    conc = disc = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        sgn = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        conc += sgn > 0; disc += sgn < 0
    return (conc - disc) / (len(a) * (len(a) - 1) / 2), int(disc)


INFOS = ["none", "length", "task", "oracle"]


def main():
    data = load()
    models = list(data)
    out = {"models": models, "theta0": THETA0, "n_tasks": {m: len(data[m]["tid"]) for m in models},
           "n_runs": {m: int(sum(len(r) for r in data[m]["runs"])) for m in models}}
    rng = np.random.default_rng(SEED)

    # ---------------- point estimates, three value weightings
    main_an = {}
    for gamma in [1.0, 0.5, 0.0]:
        G = {}
        for m in models:
            an = analyse(data[m], gamma, rng, N_SPLITS)
            if gamma == 1.0:
                main_an[m] = an
            G[m] = dict(acc=an["acc"], aw=an["aw"],
                        U={I: U_of(an, I, **THETA0) for I in INFOS},
                        pi={I: [pi_of(an, I, k) for k in KGRID] for I in INFOS})
        out[f"gamma_{gamma}"] = G

    # ---------------- calibration of the cross-fitted length model (gamma irrelevant)
    cal = {}
    for m in models:
        an = main_an[m]
        cal[m] = dict(brier_len=float(np.mean((an["qlen"] - an["y"]) ** 2)),
                      brier_const=float(np.mean((an["acc"] - an["y"]) ** 2)),
                      mean_q=float(an["qlen"].mean()), mean_y=float(an["y"].mean()))
    out["calibration"] = cal

    # ---------------- within-task run-to-run inconsistency
    cons = {}
    for m in models:
        y = main_an[m]["y"]; w = main_an[m]["w"]
        mixed = (y > 0) & (y < 1)
        cons[m] = dict(frac_tasks_mixed=float(mixed.mean()),
                       timeweighted_frac_mixed=float(np.sum(w * mixed) / np.sum(w)))
    out["consistency"] = cons

    # ---------------- concentration of the time weighting
    m0 = max(models, key=lambda m: len(data[m]["tid"]))
    t0 = data[m0]["t"]; src0 = data[m0]["src"]
    out["time_weight"] = dict(model_used=m0, n_tasks=int(len(t0)), total_hours=float(t0.sum() / 60),
                              effective_n=float(t0.sum() ** 2 / np.sum(t0 ** 2)),
                              share_top10_tasks=float(np.sort(t0)[-10:].sum() / t0.sum()),
                              share_tasks_over_4h=float(t0[t0 > 240].sum() / t0.sum()),
                              n_tasks_over_4h=int((t0 > 240).sum()),
                              median_minutes=float(np.median(t0)),
                              share_by_source={s_: float(t0[src0 == s_].sum() / t0.sum()) for s_ in sorted(set(src0))},
                              n_by_source={s_: int((src0 == s_).sum()) for s_ in sorted(set(src0))},
                              n_families=int(len(set(data[m0]["fam"]))))
    # accuracy restricted to tasks over 4 human-hours, and under 15 minutes
    out["acc_by_length"] = {m: dict(under15=float(main_an[m]["y"][main_an[m]["t"] < 15].mean()),
                                    over240=float(main_an[m]["y"][main_an[m]["t"] > 240].mean()))
                            for m in models}

    # ---------------- Proposition 1 check on data: U == identity(pi)
    maxerr = 0.0
    for m in models:
        an = main_an[m]
        for I in ["none", "length", "task"]:
            for rho, h, c in itertools.product(RHOS, HS, CS):
                u = U_of(an, I, rho, h, c)
                u2 = identity_U(lambda k: pi_of(an, I, k), rho, h, c)
                maxerr = max(maxerr, abs(u - u2))
    out["prop1_max_abs_error"] = maxerr

    # ---------------- context grid: rank agreement between accuracy and usefulness
    acc = np.array([main_an[m]["acc"] for m in models])
    grid = []
    for I in ["none", "length", "task"]:
        for rho, h, c in itertools.product(RHOS, HS, CS):
            u = np.array([U_of(main_an[m], I, rho, h, c) for m in models])
            pos = u > 1e-9
            tau, disc = kendall(acc, u)
            ties = int(sum(1 for i, j in itertools.combinations(range(len(u)), 2) if abs(u[i] - u[j]) < 1e-12))
            grid.append(dict(info=I, rho=rho, h=h, c=c, tau=tau, discordant=disc, tied_pairs=ties,
                             n_positive=int(pos.sum()), best_acc=models[int(acc.argmax())],
                             best_U=models[int(u.argmax())], max_U=float(u.max()),
                             same_top=bool(u.argmax() == acc.argmax())))
    out["grid"] = grid
    with open(os.path.join(RES, "context_grid.csv"), "w") as f:
        keys = list(grid[0].keys()); f.write(",".join(keys) + "\n")
        for g in grid:
            f.write(",".join(str(g[k]) for k in keys) + "\n")
    summ = {}
    for I in ["none", "length", "task"]:
        gg = [g for g in grid if g["info"] == I]
        summ[I] = dict(n_contexts=len(gg),
                       median_tau=float(np.median([g["tau"] for g in gg])),
                       min_tau=float(min(g["tau"] for g in gg)),
                       mean_discordant=float(np.mean([g["discordant"] for g in gg])),
                       max_discordant=int(max(g["discordant"] for g in gg)),
                       mean_tied=float(np.mean([g["tied_pairs"] for g in gg])),
                       frac_top_differs=float(np.mean([not g["same_top"] for g in gg])),
                       frac_contexts_no_model_useful=float(np.mean([g["n_positive"] == 0 for g in gg])),
                       mean_n_positive=float(np.mean([g["n_positive"] for g in gg])))
    out["grid_summary"] = summ

    # ---------------- dominance (increasing convex order) between model pairs
    dom = {}
    for I in ["length", "task"]:
        P = {m: np.array([pi_of(main_an[m], I, k) for k in KGRID]) for m in models}
        n_pairs = n_dom = 0; fails = []; MAT = [0.005, 0.01, 0.02]; mat = {t_: 0 for t_ in MAT}
        for a, b in itertools.combinations(models, 2):
            hi, lo = (a, b) if main_an[a]["acc"] >= main_an[b]["acc"] else (b, a)
            n_pairs += 1
            d = P[hi] - P[lo]
            for tol in MAT:
                mat[tol] += int(d.min() < -tol)
            if np.all(d >= -1e-12):
                n_dom += 1
            else:
                j = int(np.argmin(d))
                fails.append(dict(higher_acc=hi, lower_acc=lo, acc_gap=main_an[hi]["acc"] - main_an[lo]["acc"],
                                  worst_k=float(KGRID[j]), pi_gap=float(d[j])))
        dom[I] = dict(n_pairs=n_pairs, n_dominated=n_dom, n_crossing=n_pairs - n_dom,
                      n_crossing_material={str(k_): v_ for k_, v_ in mat.items()},
                      crossings=sorted(fails, key=lambda z: z["pi_gap"])[:15],
                      max_acc_gap_among_crossings=max([f["acc_gap"] for f in fails], default=0.0))
    out["dominance"] = dom

    # ---------------- U as a function of rho (h=5,c=0.05), for the figure/table
    rho_line = np.round(np.linspace(0, 0.6, 13), 2)
    out["rho_sweep"] = dict(rho=list(map(float, rho_line)),
                            U={I: {m: [U_of(main_an[m], I, r, THETA0["h"], THETA0["c"]) for r in rho_line]
                                   for m in models} for I in INFOS})

    # ---------------- cluster bootstrap over task families (gamma=1, theta0)
    brng = np.random.default_rng(SEED + 1)
    boot = {m: {k: [] for k in ["acc", "aw", "none", "length", "task", "oracle"]} for m in models}
    taus = {I: [] for I in ["none", "length", "task"]}
    fams_all = sorted(set(itertools.chain.from_iterable(data[m]["fam"] for m in models)))
    for b in range(N_BOOT):
        pick = brng.choice(len(fams_all), len(fams_all), replace=True)
        cnt = collections.Counter(fams_all[i] for i in pick)
        accs = []; us = {I: [] for I in taus}
        for m in models:
            fam = data[m]["fam"]
            idx = np.concatenate([np.repeat(np.where(fam == f)[0], c_) for f, c_ in cnt.items()]).astype(int)
            an = analyse(data[m], 1.0, brng, N_SPLITS_BOOT, idx=idx)
            boot[m]["acc"].append(an["acc"]); boot[m]["aw"].append(an["aw"]); accs.append(an["acc"])
            for I in INFOS:
                u = U_of(an, I, **THETA0); boot[m][I].append(u)
                if I in us: us[I].append(u)
        for I in taus:
            taus[I].append(kendall(accs, us[I])[0])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["bootstrap"] = dict(n_boot=N_BOOT, ci={m: {k: ci(v) for k, v in boot[m].items()} for m in models},
                            tau_ci={I: ci(v) for I, v in taus.items()},
                            tau_point={I: kendall(acc, [U_of(main_an[m], I, **THETA0) for m in models])[0] for I in taus})

    json.dump(out, open(os.path.join(RES, "usefulness.json"), "w"), indent=1)

    # ---------------- main table
    order = sorted(models, key=lambda m: -main_an[m]["acc"])
    G = out["gamma_1.0"]
    with open(os.path.join(RES, "table_main.csv"), "w") as f:
        f.write("model,n_tasks,n_runs,accuracy,acc_lo,acc_hi,time_weighted_accuracy,aw_lo,aw_hi,"
                "U_none,U_length,len_lo,len_hi,U_task,task_lo,task_hi,U_oracle,frac_tasks_mixed,brier_len,brier_const\n")
        for m in order:
            c_ = out["bootstrap"]["ci"][m]
            f.write(f"{m},{out['n_tasks'][m]},{out['n_runs'][m]},{G[m]['acc']:.3f},{c_['acc'][0]:.3f},{c_['acc'][1]:.3f},"
                    f"{G[m]['aw']:.3f},{c_['aw'][0]:.3f},{c_['aw'][1]:.3f},{G[m]['U']['none']:.3f},"
                    f"{G[m]['U']['length']:.3f},{c_['length'][0]:.3f},{c_['length'][1]:.3f},"
                    f"{G[m]['U']['task']:.3f},{c_['task'][0]:.3f},{c_['task'][1]:.3f},{G[m]['U']['oracle']:.3f},"
                    f"{cons[m]['frac_tasks_mixed']:.3f},{cal[m]['brier_len']:.3f},{cal[m]['brier_const']:.3f}\n")
    with open(os.path.join(RES, "table_gamma.csv"), "w") as f:
        f.write("model,accuracy," + ",".join(f"aw_g{g},Ulen_g{g},Utask_g{g}" for g in [1.0, 0.5, 0.0]) + "\n")
        for m in order:
            f.write(f"{m},{G[m]['acc']:.3f}," + ",".join(
                f"{out[f'gamma_{g}'][m]['aw']:.3f},{out[f'gamma_{g}'][m]['U']['length']:.3f},{out[f'gamma_{g}'][m]['U']['task']:.3f}"
                for g in [1.0, 0.5, 0.0]) + "\n")

    # ---------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        a = [G[m]["acc"] for m in order]
        for I, mk in [("oracle", "v"), ("task", "s"), ("length", "o"), ("none", "x")]:
            ax[0].plot(a, [G[m]["U"][I] for m in order], mk, label=f"info: {I}", ms=5)
        ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax[0].set_xlabel("headline accuracy (equal task weight)"); ax[0].set_ylabel("usefulness U (share of human hours saved)")
        ax[0].set_title("rho=0.2, h=5, c=0.05, time-weighted"); ax[0].legend(fontsize=8)
        top = order[0]; mid = order[len(order) // 2]
        for m, ls in [(top, "-"), (mid, "--")]:
            for I, col in [("none", "C3"), ("length", "C0"), ("task", "C2"), ("oracle", "C4")]:
                ax[1].plot(KGRID, G[m]["pi"][I], ls, color=col, label=f"{m.replace(' (Inspect)','')} / {I}")
        ax[1].set_xlabel("threshold k"); ax[1].set_ylabel("stop-loss curve pi(k)"); ax[1].legend(fontsize=6)
        ax[1].set_title("delegation-surplus curves")
        for m in order[:6]:
            ax[2].plot(out["rho_sweep"]["rho"], out["rho_sweep"]["U"]["length"][m], label=m.replace(" (Inspect)", ""))
        ax[2].set_xlabel("verification cost ratio rho"); ax[2].set_ylabel("U (info: length)"); ax[2].legend(fontsize=7)
        ax[2].set_title("h=5, c=0.05")
        plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_usefulness.png"), dpi=130)
    except Exception as e:  # figures are optional
        print("figure skipped:", e)

    print(open(os.path.join(RES, "table_main.csv")).read())
    print(json.dumps(out["grid_summary"], indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "crossings"} for k, v in dom.items()}, indent=1))
    print("prop1 max err", maxerr, "tau", out["bootstrap"]["tau_point"], out["bootstrap"]["tau_ci"])


if __name__ == "__main__":
    main()
