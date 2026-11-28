"""Round-3 analysis of Exp. H: 5 seeds (0,1 from transformer_runs.jsonl; 2,3,4 from
transformer_runs_r3.jsonl) and the optimisation control (transformer_converge.jsonl).

Delta_m = L_unified,m - L_specialist,m, test nats/token, each model at that modality's own
best-validation checkpoint ('own') or the unified model's single joint checkpoint ('joint';
specialists always at their own).  Per cell: mean over seeds, seed SE, t-based 95% CI
(df = n_seeds-1), number of seeds with Delta>0.  Overall: paired sign counts with exact
two-sided binomial p-values; per-seed slope of Delta against log2(parameters).
Test-set sampling uncertainty: for seeds 2-4 the per-window test losses are stored; the
160 held-out windows are identical for the unified model and the specialist, so we bootstrap
windows (paired) with the seeds held fixed.
Writes ../results/transformer_summary_r3.json and ../results/figH_transformer_r3.png
"""
import json, os
from collections import defaultdict
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(__file__), "..", "results")
load = lambda f: [json.loads(l) for l in open(os.path.join(R, f))] if os.path.exists(os.path.join(R, f)) else []
runs = load("transformer_runs.jsonl") + load("transformer_runs_r3.jsonl")
idx = {(tuple(r["mods"]), r["D"], r["d_model"], r["seed"]): r for r in runs}
SEEDS, DS, DMS = [0, 1, 2, 3, 4], [25000, 100000], [16, 32, 64, 128]
order = ["image", "audio", "text", "fashion"]


def stat(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2:
        return dict(delta=float(d.mean()) if n else None, n_seeds=n)
    se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return dict(delta=d.mean(), se=se, ci95=[d.mean() - t * se, d.mean() + t * se], n_pos=int((d > 0).sum()),
                n_seeds=n, per_seed=d.tolist())


def sign(d):
    d = np.asarray(d); k, n = int((d > 0).sum()), len(d)
    return dict(n_pos=k, n=n, p_two_sided=float(stats.binomtest(k, n, 0.5).pvalue) if n else None, mean=float(d.mean()) if n else None)


def window_boot(u, s, m, nb=4000, seed=0):
    """paired bootstrap over test windows; u, s lists of runs (same seeds)."""
    a = [np.array(x["test_windows_per_mod_stop"][m]) - np.array(y["test_windows_per_mod_stop"][m])
         for x, y in zip(u, s) if "test_windows_per_mod_stop" in x and "test_windows_per_mod_stop" in y]
    if not a:
        return None
    dw = np.mean(a, 0); rng = np.random.default_rng(seed)
    bs = dw[rng.integers(0, len(dw), (nb, len(dw)))].mean(1)
    return dict(delta_these_seeds=float(dw.mean()), ci95=np.percentile(bs, [2.5, 97.5]).tolist(), n_seeds=len(a), n_windows=len(dw))


out = dict(n_runs=len(runs), seeds=SEEDS)
grid = {}; alld = defaultdict(list); slopes = defaultdict(list)
for D in DS:
    for dm in DMS:
        ent = {}
        for m in ["image", "audio"]:
            own, joint, U, Sp, Ls = [], [], [], [], []
            for sd in SEEDS:
                u, s = idx.get((("image", "audio"), D, dm, sd)), idx.get(((m,), D, dm, sd))
                if u is None or s is None:
                    continue
                own.append(u["test_per_mod_stop"][m] - s["test_per_mod_stop"][m])
                joint.append(u["test"][m] - s["test_per_mod_stop"][m]); U.append(u); Sp.append(s); Ls.append(s["test_per_mod_stop"][m])
            if not own:
                continue
            ent[m] = dict(own=stat(own), joint=stat(joint), L_s=float(np.mean(Ls)), window_bootstrap_own=window_boot(U, Sp, m))
            alld[(m, "own")] += own; alld[(m, "joint")] += joint; alld[(m, "own", D)] += own
            ent["n_params"] = U[0]["n_params"]
        if ent:
            grid[f"D={D},d={dm}"] = ent
out["grid"] = grid
out["sign_tests"] = {"|".join(map(str, k)): sign(v) for k, v in alld.items()}
# grand mean over the 8 cells per seed (seed is the unit) and size trend per seed
tr = {}
for m in ["image", "audio"]:
    for D in DS:
        sl = []
        for sd in SEEDS:
            ys = [grid[f"D={D},d={dm}"][m]["own"]["per_seed"] for dm in DMS if f"D={D},d={dm}" in grid and m in grid[f"D={D},d={dm}"]]
            if len(ys) == 4 and all(len(y) == len(SEEDS) for y in ys):
                x = np.log2([grid[f"D={D},d={dm}"]["n_params"] for dm in DMS])
                sl.append(np.polyfit(x, [y[SEEDS.index(sd)] for y in ys], 1)[0])
        if len(sl) > 1:
            tr[f"{m},D={D}"] = dict(slope_per_doubling_of_params=stat(sl))
    gm = []
    for i, sd in enumerate(SEEDS):
        c = [grid[k][m]["own"]["per_seed"][i] for k in grid if m in grid[k] and len(grid[k][m]["own"]["per_seed"]) == len(SEEDS)]
        if len(c) == 8:
            gm.append(np.mean(c))
    if len(gm) > 1:
        tr[f"{m},grand_mean_over_8_cells"] = stat(gm)
out["trends"] = tr
# how many image models stop near the cap
img = [r["step_per_mod_stop"]["image"] for r in runs if "image" in r["mods"]]
out["image_best_step_ge_2400"] = dict(n=int(np.sum(np.array(img) >= 2400)), of=len(img))

# P4
B = {}
for Mn in [2, 3, 4]:
    mods = tuple(order[:Mn]); ent = {}; per_seed_mean = []
    for sd in SEEDS:
        u = idx.get((mods, 25000, 32, sd))
        ss = [idx.get(((m,), 25000, 32, sd)) for m in mods]
        if u is None or any(s is None for s in ss):
            continue
        ds = {m: u["test_per_mod_stop"][m] - s["test_per_mod_stop"][m] for m, s in zip(mods, ss)}
        for m in mods:
            ent.setdefault(m, []).append(ds[m])
        per_seed_mean.append(np.mean(list(ds.values())))
    if ent:
        B[str(Mn)] = {m: stat(v) for m, v in ent.items()}; B[str(Mn)]["mean_delta"] = stat(per_seed_mean)
# paired changes when modalities are added (seed-paired)
ch = {}
for m, a, b in [("image", "2", "4"), ("image", "2", "3"), ("audio", "2", "4"), ("text", "3", "4"), ("mean_delta", "2", "4")]:
    if a in B and b in B and m in B[a] and m in B[b] and B[a][m]["n_seeds"] == B[b][m]["n_seeds"] > 1:
        ch[f"{m}: M={b} minus M={a}"] = stat(np.array(B[b][m]["per_seed"]) - np.array(B[a][m]["per_seed"]))
out["budget"] = B; out["budget_changes"] = ch

# optimisation control
cv = load("transformer_converge.jsonl")
if cv:
    cidx = {(tuple(r["mods"]), r["seed"], r["lr"]): r for r in cv}
    LRS = sorted({r["lr"] for r in cv}); C = dict(lrs=LRS, maxsteps=cv[0]["maxsteps"])
    for m in ["image", "audio"]:
        sel, fixed, short, chosen_u, chosen_s, steps_u, steps_s, Lsel = [], [], [], [], [], [], [], []
        for sd in sorted({r["seed"] for r in cv}):
            us = [cidx.get((("image", "audio"), sd, lr)) for lr in LRS]; ss = [cidx.get(((m,), sd, lr)) for lr in LRS]
            if any(x is None for x in us + ss):
                continue
            bu = min(us, key=lambda r: r["val_per_mod_stop"][m]); bs_ = min(ss, key=lambda r: r["val_per_mod_stop"][m])
            sel.append(bu["test_per_mod_stop"][m] - bs_["test_per_mod_stop"][m]); Lsel.append(bs_["test_per_mod_stop"][m])
            chosen_u.append(bu["lr"]); chosen_s.append(bs_["lr"]); steps_u.append(bu["step_per_mod_stop"][m]); steps_s.append(bs_["step_per_mod_stop"][m])
            u2, s2 = cidx[(("image", "audio"), sd, 2e-3)], cidx[((m,), sd, 2e-3)]
            fixed.append(u2["test_per_mod_stop"][m] - s2["test_per_mod_stop"][m])
            u0, s0 = idx.get((("image", "audio"), 25000, 32, sd)), idx.get(((m,), 25000, 32, sd))
            if u0 and s0:
                short.append(u0["test_per_mod_stop"][m] - s0["test_per_mod_stop"][m])
        if sel:
            C[m] = dict(lr_selected_9000=stat(sel), lr2em3_9000=stat(fixed), lr2em3_3000_same_seeds=stat(short),
                        L_s_selected=float(np.mean(Lsel)), lr_unified=chosen_u, lr_specialist=chosen_s,
                        best_step_unified=steps_u, best_step_specialist=steps_s)
    C["image_best_step_ge_8400"] = dict(n=int(sum(r["step_per_mod_stop"]["image"] >= 8400 for r in cv if "image" in r["mods"])),
                                        of=int(sum("image" in r["mods"] for r in cv)))
    out["converge"] = C
json.dump(out, open(os.path.join(R, "transformer_summary_r3.json"), "w"), indent=1, default=float)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, m in zip(axs[:2], ["image", "audio"]):
        for D, c in [(25000, "C0"), (100000, "C3")]:
            ks = [f"D={D},d={dm}" for dm in DMS if f"D={D},d={dm}" in grid and m in grid[f"D={D},d={dm}"]]
            xs = [grid[k]["n_params"] for k in ks]; ys = [grid[k][m]["own"]["delta"] for k in ks]
            lo = [y - grid[k][m]["own"]["ci95"][0] for k, y in zip(ks, ys)]; hi = [grid[k][m]["own"]["ci95"][1] - y for k, y in zip(ks, ys)]
            ax.errorbar(np.array(xs) * (1.0 if D == 25000 else 1.08), ys, yerr=[lo, hi], fmt="o-", color=c, ms=4, capsize=2, label=f"D={D // 1000}k tokens/modality")
        ax.axhline(0, color="k", lw=.8, ls="--"); ax.set_xscale("log"); ax.set_xlabel("parameters")
        ax.set_ylabel("L_unified - L_specialist (nats/token)"); ax.set_title(f"{m}: mean, 95% t-CI over 5 seeds", fontsize=9); ax.legend(fontsize=7)
    ax = axs[2]
    for m, c in zip(order, ["C0", "C1", "C2", "C4"]):
        xs = [int(k) for k in B if m in B[k]]
        if xs:
            ys = [B[str(k)][m]["delta"] for k in xs]; es = [B[str(k)][m].get("se", 0) for k in xs]
            ax.errorbar(xs, ys, yerr=es, fmt="o-", color=c, capsize=2, label=m)
    ax.axhline(0, color="k", lw=.8, ls="--"); ax.set_xlabel("number of modalities M (d_model=32, D=25k)"); ax.set_xticks([2, 3, 4])
    ax.set_ylabel("L_unified - L_specialist (mean ± SE)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figH_transformer_r3.png"), dpi=150)
except ImportError:
    print("matplotlib missing: figure skipped")
f = lambda s: "%+.3f±%.3f [%+.3f,%+.3f] %d/%d" % (s["delta"], s["se"], *s["ci95"], s["n_pos"], s["n_seeds"]) if s.get("se") is not None else str(s)
for k, e in grid.items():
    for m in ["image", "audio"]:
        if m in e:
            print(k, m, "own", f(e[m]["own"]), "| joint", f(e[m]["joint"]), "| Ls %.3f" % e[m]["L_s"], "| win", e[m]["window_bootstrap_own"])
print(json.dumps({k: out[k] for k in ["sign_tests", "trends", "image_best_step_ge_2400", "budget_changes"]}, indent=0, default=float))
for k, e in B.items():
    print("M", k, {m: f(v) for m, v in e.items()})
print(json.dumps(out.get("converge"), default=float))
