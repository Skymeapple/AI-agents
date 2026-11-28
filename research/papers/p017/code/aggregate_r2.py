"""Aggregate the round-2 experiments into results/summary_r2.md + figures fig6-fig9."""
import glob, json, os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
L = ["# Round-2 summary (all numbers computed by code/aggregate_r2.py from raw outputs)\n"]


def ms(x):
    x = np.asarray(x, float); return f"{x.mean():.3f} ± {x.std():.3f}"


# ---------- exact ceilings ----------
ex = json.load(open(os.path.join(RES, "exact_ceiling.json")))
L.append("## 1. Exact finite-time ceiling (review M4)\n")
L.append("| world | seed | Acc*_1 | Acc*_2 | Acc*_4 | Acc*_8 | Acc*_12 | Acc*_16 | Acc*_24 | max truncation excess (all encoders, k) |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
allexc = []
for key, r in ex.items():
    c = r["exact_curve"]
    exc = [v for m in r["models"].values() if "trunc_excess" in m for v in m["trunc_excess"].values()]
    allexc += exc
    L.append(f"| {key.split('_s')[0]} | {key.split('_s')[1]} | " + " | ".join(f"{c[k]:.3f}" for k in [1, 2, 4, 8, 12, 16, 24]) + f" | {max(exc):+.4f} |")
L.append(f"\nAll truncation cells: n = {len(allexc)}, max excess of truncated-probe accuracy over exact Acc*_k = {max(allexc):+.4f}, "
         f"number of cells with excess > 0.005: {sum(e > 0.005 for e in allexc)}\n")
L.append("k_eff against the exact ceiling (main-table models):\n")
for key, r in ex.items():
    L.append(f"- {key}: " + ", ".join(f"{n}: {m['k_eff_exact']}" for n, m in r["models"].items()))
L.append("\nTraining-dynamics checkpoints (epoch, probe acc, exact k_eff):\n")
for key, r in ex.items():
    if "dyn_k_eff_exact" in r:
        L.append(f"- {key}: " + "; ".join(f"ep{e}: {p:.3f}/{k}" for e, p, k in r["dyn_k_eff_exact"]))

# ---------- GRU seeds ----------
L.append("\n## 2. Trained GRU, 10 seeds per world and budget (one-cycle schedule; review M5)\n")
seeds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "seeds", "gru_*.json")))]
L.append("| world | epochs | n | next-obs acc | probe acc | k_eff (per seed) | #seeds k_eff ≥ 12 | CRA probe | CRA random | CRA full | CRA none |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for mode in ["balanced2", "landmark"]:
    for ep in [10, 30]:
        rs = sorted([s for s in seeds if s["mode"] == mode and s["epochs"] == ep], key=lambda s: s["seed"])
        if not rs: continue
        L.append(f"| {mode} | {ep} | {len(rs)} | {ms([r['obs_acc'] for r in rs])} | {ms([r['probe_acc'] for r in rs])} | "
                 f"{', '.join(str(r['k_eff_exact']) for r in rs)} | {sum(r['k_eff_exact'] >= 12 for r in rs)} | "
                 f"{ms([r['cra']['probe_subspace'] for r in rs])} | {ms([r['cra']['random_subspace'] for r in rs])} | "
                 f"{ms([r['cra']['full_swap'] for r in rs])} | {ms([r['cra']['none'] for r in rs])} |")
L.append("\nPer-seed (probe acc / k_eff / CRA probe / CRA random / next-obs / landmark recall):\n")
for s in sorted(seeds, key=lambda s: (s["mode"], s["epochs"], s["seed"])):
    L.append(f"- {s['mode']} e{s['epochs']} s{s['seed']}: {s['probe_acc']:.3f} / {s['k_eff_exact']} / {s['cra']['probe_subspace']:.3f} / "
             f"{s['cra']['random_subspace']:.3f} / {s['obs_acc']:.3f} / {s.get('landmark_recall', float('nan')):.3f}")
# run-level correlation of intervention gain with probe accuracy (independent runs only)
if seeds:
    pa = np.array([s["probe_acc"] for s in seeds]); gain = np.array([s["cra"]["probe_subspace"] - s["cra"]["random_subspace"] for s in seeds])
    r = np.corrcoef(pa, gain)[0, 1]
    rng = np.random.default_rng(0); bs = []
    for _ in range(10000):
        j = rng.integers(0, len(pa), len(pa))
        if np.std(pa[j]) > 0 and np.std(gain[j]) > 0: bs.append(np.corrcoef(pa[j], gain[j])[0, 1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    from scipy.stats import spearmanr
    rho = spearmanr(pa, gain).correlation
    L.append(f"\nIndependent runs (n = {len(pa)}; each a separately trained GRU): Pearson r(probe acc, CRA gain) = {r:.3f}, "
             f"95% bootstrap CI [{lo:.3f}, {hi:.3f}], Spearman ρ = {rho:.3f}. "
             f"Runs with probe < 0.4: {int((pa < 0.4).sum())}, of which gain ≤ 0: {int(((pa < 0.4) & (gain <= 0)).sum())}. "
             f"Runs with probe ≥ 0.75: {int((pa >= 0.75).sum())}, of which gain ≥ 0.17: {int(((pa >= 0.75) & (gain >= 0.17)).sum())}; "
             f"min gain among them: {gain[pa >= 0.75].min():.3f}.")
    L.append(f"CRA(probe) range over runs with k_eff ≥ 12: {min(s['cra']['probe_subspace'] for s in seeds if s['k_eff_exact'] >= 12):.3f}–"
             f"{max(s['cra']['probe_subspace'] for s in seeds if s['k_eff_exact'] >= 12):.3f}; CRA(random) over the same runs: "
             f"{min(s['cra']['random_subspace'] for s in seeds if s['k_eff_exact'] >= 12):.3f}–{max(s['cra']['random_subspace'] for s in seeds if s['k_eff_exact'] >= 12):.3f}")
    lm = [s for s in seeds if s["mode"] == "landmark"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for mode, mk in [("balanced2", "o"), ("landmark", "s")]:
        for ep, c in [(10, "#1f5fa8"), (30, "#d9822b")]:
            ss = [s for s in seeds if s["mode"] == mode and s["epochs"] == ep]
            ax.scatter([s["probe_acc"] for s in ss], [s["cra"]["probe_subspace"] - s["cra"]["random_subspace"] for s in ss],
                       marker=mk, color=c, s=22, label=f"{mode}, {ep} ep")
    ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("linear state-probe accuracy"); ax.set_ylabel("CRA(probe) − CRA(random)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(RES, "fig6_seeds_gain.png"), dpi=160); plt.close()

# ---------- reservoir sweep ----------
sw = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "sweep", "res_*.json")))]
if sw:
    L.append("\n## 3. Untrained reservoir sweep, d = 512 (review M1). Mean over 3 seeds.\n")
    L.append("| world | λ | W | leak | input gain | next-obs | linear probe | k_eff (lin) | MLP probe | k_eff (MLP) | δ(8) | trunc acc k=8 / Acc*_8 | CRA probe | CRA random | CRA full |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mode in ["balanced2", "landmark"]:
        rs = [s for s in sw if s["mode"] == mode]
        if not rs: continue
        n = min(len(s["configs"]) for s in rs)
        for i in range(n):
            cs = [s["configs"][i] for s in rs]; c0 = cs[0]
            L.append(f"| {mode} | {c0['lam']} | {c0.get('wtype', 'orth' if c0['orth'] else 'norm')} | {c0['leak']} | {c0['in_scale']} | "
                     f"{np.mean([c['obs_acc'] for c in cs]):.3f} | {ms([c['probe_lin'] for c in cs])} | {','.join(str(c['k_eff_lin']) for c in cs)} | "
                     f"{ms([c['probe_mlp'] for c in cs])} | {','.join(str(c['k_eff_mlp']) for c in cs)} | "
                     f"{np.mean([c['trunc']['8']['delta'] for c in cs]):.2g} | {np.mean([c['trunc']['8']['trunc_acc'] for c in cs]):.3f} / {np.mean([c['trunc']['8']['acc_star'] for c in cs]):.3f} | "
                     f"{np.mean([c['cra']['probe_subspace'] for c in cs]):.3f} | {np.mean([c['cra']['random_subspace'] for c in cs]):.3f} | {np.mean([c['cra']['full_swap'] for c in cs]):.3f} |")
    exc = [c["trunc"][k]["trunc_acc"] - c["trunc"][k]["acc_star"] for s in sw for c in s["configs"] for k in c["trunc"]]
    L.append(f"\nSweep truncation cells: n = {len(exc)}, max excess over exact Acc*_k = {max(exc):+.4f}")
    best = {}
    for s in sw:
        for c in s["configs"]:
            for p in ["lin", "mlp"]:
                key = (s["mode"], p)
                if key not in best or c[f"probe_{p}"] > best[key][0]:
                    best[key] = (round(c[f"probe_{p}"], 3), c[f"k_eff_{p}"], c["lam"], c["in_scale"], c.get("wtype"), s["seed"])
    L.append("Best single untrained reservoir per world/probe (acc, k_eff, λ, gain, orth, leak, seed): " + "; ".join(f"{k}: {v}" for k, v in best.items()))

# ---------- calibration ----------
cf = os.path.join(RES, "calib_window.json")
if os.path.exists(cf):
    cal = json.load(open(cf))
    L.append("\n## 4. Learned window decoders vs exact ceiling (calibration for Othello)\n")
    L.append("| world | seed | " + " | ".join(f"k={k}" for k in [1, 2, 3, 4, 6, 8, 12, 16]) + " |")
    L.append("|---|---|" + "---|" * 8)
    gaps = []
    for key, r in cal.items():
        L.append(f"| {key.split('_s')[0]} | {key.split('_s')[1]} | " + " | ".join(f"{r['mlp'][str(k)]:.3f}/{r['exact'][str(k)]:.3f}" for k in [1, 2, 3, 4, 6, 8, 12, 16] if str(k) in r['mlp']) + " |")
        gaps += [r["exact"][k] - r["mlp"][k] for k in r["mlp"]]
    L.append(f"\n(decoder/exact). Exact − decoder: mean {np.mean(gaps):+.4f}, max {np.max(gaps):+.4f}, min {np.min(gaps):+.4f}")

# ---------- transformer windows ----------
tfs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "tf", "tf_*.json")))]
if tfs:
    L.append("\n## 5. Transformers (3 seeds): attention-window truncation (Corollary 2)\n")
    L.append("| world | seed | model | probe acc | k_eff | next-obs (own head) | w=1 | w=2 | w=4 | w=8 | w=16 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    viol = []
    for r in tfs:
        for name, m in r["models"].items():
            cells = []
            for w in ["1", "2", "4", "8", "16"]:
                v = m["window"][w]; cells.append(f"{v['refit']:.3f}/{v['acc_star']:.3f}")
            for v in m["window"].values():
                viol.append(max(v["same"], v["refit"]) - v["acc_star"])
            L.append(f"| {r['mode']} | {r['seed']} | {name} | {m['probe_acc']:.3f} | {m['k_eff_exact']} | {m['obs_acc_own_head']:.3f} | " + " | ".join(cells) + " |")
    L.append(f"\n(cells: refitted-probe accuracy of the windowed model / exact Acc*_k for k = ceil(receptive field/2)). "
             f"Max over all {len(viol)} cells of (windowed accuracy − Acc*_k), same or refitted probe: {max(viol):+.4f}")

open(os.path.join(RES, "summary_r2.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
