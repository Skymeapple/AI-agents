"""Figures for the paper. Reads results/*.json, writes paper/figures/*.png.
Usage: python -m wu.plots [--quick]"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FIGURES, RESULTS, load_json

C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
GRAY, INK, INK2 = "#9c9c9c", "#191919", "#666666"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 2, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
SFX = "_quick" if "--quick" in sys.argv else ""


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def have(name):
    return (RESULTS / f"{name}{SFX}.json").exists()


def mean_sd(vals, axis=0):
    a = np.asarray(vals, float)
    return a.mean(axis), a.std(axis)


# ----------------------------------------------------------------------------- E1
def fig_e1():
    R = [load_json(f.stem) for f in sorted(RESULTS.glob(f"e1_seed[0-9]{SFX}.json"))]
    r0 = R[0]
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2))
    # (a) code length
    a = ax[0, 0]
    keys = ["uniform", "best_ngram", "model", "bayes"]
    labels = ["Uniform", "Best\nn-gram", "Trans-\nformer", "Bayes\nfilter"]
    m, s = mean_sd([[r["nll"][k] for k in keys] for r in R])
    mo, so = mean_sd([[r["nll"]["model_ood"], r["nll"]["bayes_ood"]] for r in R])
    a.bar(np.arange(4), m, yerr=s, color=[GRAY, C[1], C[0], C[2]], width=0.62, capsize=2)
    a.bar([2.0 + 0.0, 3.0], [0, 0])
    a.scatter([2, 3], mo, marker="D", s=22, color=INK, zorder=5, label="policy-shift test")
    for i, v in enumerate(m):
        a.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    a.set_xticks(range(4), labels, fontsize=7.5)
    a.set_ylabel("nats / observation")
    a.set_title("(a) Prediction loss on observations")
    a.legend(loc="upper right")
    # (b) per-timestep
    a = ax[0, 1]
    t = np.arange(1, len(r0["per_t"]["model"]) + 1)
    mm, _ = mean_sd([r["per_t"]["model"] for r in R])
    a.plot(t, mm, color=C[0], label="Transformer")
    a.plot(t, r0["per_t"]["bayes"], color=C[2], ls="--", label="Bayes filter")
    a2 = a.twinx()
    a2.plot(t, r0["per_t"]["belief_entropy"], color=GRAY, lw=1.2, label="belief entropy")
    a2.set_ylabel("posterior entropy over states (nats)", color=INK2, fontsize=7.5)
    a2.spines["right"].set_visible(True)
    a2.grid(False)
    a.set_xlabel("step in trajectory")
    a.set_ylabel("nats / observation")
    a.set_title("(b) Loss falls as hidden state is inferred")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper right")
    # (c) probes by layer
    a = ax[0, 2]
    L = np.arange(len(r0["probes"]))
    for key, col, lab in [("acc_state", C[0], "state (trained)"),
                          ("acc_state_identified", C[4], "state, belief collapsed"), ("acc_cell", C[6], "cell (trained)"),
                          ("acc_heading", C[3], "heading (trained)")]:
        mm, ss = mean_sd([[p[key] for p in r["probes"]] for r in R])
        a.errorbar(L, mm, yerr=ss, color=col, marker="o", label=lab, capsize=2)
    mm, _ = mean_sd([[p["acc_state"] for p in r["probes_untrained"]] for r in R])
    a.plot(L, mm, color=C[0], ls=":", marker="o", label="state (untrained)")
    a.axhline(np.mean([r["bayes_map_acc"] for r in R]), color=C[2], ls="--", lw=1.2,
              label="Bayes MAP ceiling")
    a.axhline(1 / r0["S"], color=GRAY, lw=1, label="chance (state)")
    a.set_xticks(L, ["emb"] + [f"L{i}" for i in L[1:]])
    a.set_ylim(0, 1.02)
    a.set_ylabel("linear-probe accuracy")
    a.set_title("(c) Hidden state is linearly decodable")
    a.legend(loc="upper left", fontsize=6, ncol=2)
    # (d) interventions
    a = ax[1, 0]
    layers = [iv["layer"] for iv in r0["interventions"]]
    x = np.arange(len(layers))
    for j, (k, col, lab) in enumerate([("patch", C[0], "state-swap patch"), ("random", GRAY, "random patch, same norm"),
                                       ("none", C[1], "no patch")]):
        mm, ss = mean_sd([[iv[f"{k}_acc_target"] for iv in r["interventions"]] for r in R])
        a.bar(x + (j - 1) * 0.27, mm, 0.25, yerr=ss, color=col, label=lab, capsize=2)
    a.set_xticks(x, [f"after L{l}" for l in layers])
    a.set_ylim(0, 1)
    a.set_ylabel("P(prediction = target state's view)")
    a.set_title("(d) Editing the state edits the prediction")
    a.legend(loc="upper left", fontsize=6.5)
    # (e) PCA of cell means
    a = ax[1, 1]
    P = np.asarray(r0["cell_pca"])
    xy = np.asarray(r0["cell_xy"])
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if np.abs(xy[i] - xy[j]).sum() == 1:
                a.plot(P[[i, j], 0], P[[i, j], 1], color="#cccccc", lw=0.8, zorder=1)
    sc = a.scatter(P[:, 0], P[:, 1], c=xy[:, 1], cmap="Blues", s=30, edgecolor=INK2, lw=0.4, zorder=2)
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
    a.set_title(f"(e) Cell representations (RSA ρ={np.mean([r['rsa_spearman'] for r in R]):.2f})")
    cb = fig.colorbar(sc, ax=a, fraction=0.04)
    cb.set_label("grid column", fontsize=7)
    a.grid(False)
    # (f) dynamics
    a = ax[1, 2]
    for i, r in enumerate(R):
        d = r["dynamics"]
        a.plot([q["excess_nll"] for q in d], [q["acc_state"] for q in d], marker="o", color=C[i],
               label=f"seed {r['seed']}")
    a.set_xscale("log")
    a.invert_xaxis()
    a.set_xticks([0.5, 0.2, 0.1, 0.05, 0.02], ["0.5", "0.2", "0.1", "0.05", "0.02"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("excess loss over Bayes (nats, log; training →)")
    a.set_ylabel("best-layer state-probe accuracy")
    a.set_title("(f) World model emerges as loss falls")
    a.legend()
    fig.tight_layout()
    save(fig, "fig_e1_world_model")


# ----------------------------------------------------------------------------- E2
def floor_fit(ns, e):
    """Fit e(n) = a n^-beta + f (power law plus an optimisation/noise floor) in log space.
    Returns beta."""
    from scipy.optimize import curve_fit
    ns, e = np.asarray(ns, float), np.asarray(e, float)
    f = lambda n, la, b, lf: np.log(np.exp(la) * n ** (-b) + np.exp(lf))
    p, _ = curve_fit(f, ns, np.log(e), p0=[np.log(e[0] * 128 ** 0.5), 1.0, np.log(e.min() / 2)], maxfev=20000)
    return float(p[1])


def e2_analysis(R):
    """Exponent per d from a power-law-plus-floor fit (per seed), and the c/d fit."""
    out = {}
    for d in R["dims"]:
        E = np.asarray(R["err"][str(d)])
        b = [floor_fit(R["ns"], E[:, j]) for j in range(E.shape[1])]
        out[d] = dict(beta=float(np.mean(b)), beta_sd=float(np.std(b)), beta_seeds=b,
                      global_=-R["fits"][str(d)]["slope"])
    ds = list(R["dims"])
    inv = np.array([1 / d for d in ds])
    sl = np.array([out[d]["beta"] for d in ds])
    c = float(inv @ sl / (inv @ inv))
    r2 = float(1 - ((sl - c * inv) ** 2).sum() / ((sl - sl.mean()) ** 2).sum())
    return out, ds, c, r2


def fig_e2():
    R = load_json("e2_manifold" + SFX)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    a = ax[0]
    for i, d in enumerate(R["dims"]):
        e = np.asarray(R["err"][str(d)])
        a.errorbar(R["ns"], e.mean(1), yerr=e.std(1), color=C[i], marker="o", capsize=2,
                   label=f"d={d}")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("training samples n")
    a.set_ylabel("normalised test MSE")
    a.set_title(f"(a) Learning curves, ambient D={R['D']}")
    a.legend()
    a = ax[1]
    an, ds, c, r2 = e2_analysis(R)
    dims = np.array(ds, float)
    sl = np.array([an[d]["beta"] for d in ds])
    sd = np.array([an[d]["beta_sd"] for d in ds])
    a.errorbar(1 / dims, sl, yerr=sd, fmt="o", color=C[0], capsize=2, label="β from power law + floor")
    a.plot(1 / dims, [an[d]["global_"] for d in ds], "x", color=GRAY, label="naive log-log slope")
    xx = np.linspace(0, 0.55, 50)
    a.plot(xx, c * xx, color=C[1], ls="--", label=f"{c:.1f}/d  (R²={r2:.2f})")
    a.plot(xx, 4 * xx, color=C[2], ls=":", lw=1.2, label="4/d (Sharma & Kaplan)")
    for d, s_ in zip(dims, sl):
        a.annotate(f"d={int(d)}", (1 / d, s_), textcoords="offset points", xytext=(5, -10), fontsize=7)
    a.set_xlabel("1 / intrinsic dimension")
    a.set_ylabel("learning-curve exponent  −slope")
    a.set_title("(b) Exponent scales with 1/d")
    a.legend(loc="upper left", fontsize=6.5)
    a = ax[2]
    est = [np.mean(R["twonn"][str(d)]) for d in R["dims"]]
    a.plot([0, 9], [0, 9], color=GRAY, lw=1)
    a.plot(R["dims"], est, "o", color=C[0], label="synthetic manifolds (D=64)")
    amb = R["ambient_control"]
    for j, (D, v) in enumerate(amb.items()):
        a.plot(4, v["twonn"], marker="s", color=C[2 + j], ls="none", label=f"d=4 embedded in D={D}")
    a.set_xlabel("true intrinsic dimension")
    a.set_ylabel("TwoNN estimate")
    a.set_title(f"(c) Dimension recovered; digits ≈ {R['digits_twonn']:.1f}")
    a.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_e2_manifold")


# ----------------------------------------------------------------------------- E3
def fig_e3():
    R = load_json("e3_zipf_scaling" + SFX)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    P = np.array(R["params"])
    for i, a_ in enumerate(R["alphas"]):
        c = R["capacity"][str(a_)]
        e = np.asarray(c["err"])
        ax[0].errorbar(P, e.mean(1), yerr=e.std(1), color=C[i], marker="o", capsize=2,
                       label=f"α={a_}: {c['slope']:.2f} (theory {c['predicted']:.2f})")
        d = R["data"][str(a_)]
        e = np.asarray(d["err"])
        ax[1].errorbar(R["Ds"], e.mean(1), yerr=e.std(1), color=C[i], marker="o", capsize=2,
                       label=f"α={a_}: {d['slope']:.2f} (theory {d['predicted']:.2f})")
        ax[1].plot(R["Ds"], d["theory"], color=C[i], ls=":", lw=1.2)
    for a, xl, t in [(ax[0], "parameters N", "(a) Capacity scaling"), (ax[1], "training samples D", "(b) Data scaling")]:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(xl)
        a.set_ylabel("Zipf-weighted error")
        a.set_title(t)
        a.legend(fontsize=6.5, title="fitted slope (asymptotic theory)" if a is ax[0] else
                 "fitted slope (asymptotic theory)\ndotted: exact finite-K unseen mass", title_fontsize=6.5,
                 loc="lower left")
    a = ax[2]
    a_ = R["alphas"][1]
    rc = R["capacity"][str(a_)]["rank_curves"]
    ws = sorted(rc, key=lambda k: int(k))
    cmap = plt.get_cmap("Blues")
    for j, w in enumerate(ws):
        y = np.asarray(rc[w])
        x = 1 + np.arange(len(y)) * 16 + 16
        a.plot(x, y, color=cmap(0.3 + 0.7 * j / max(1, len(ws) - 1)), lw=1.4, label=f"N={P[R['widths'].index(int(w))]:,}")
    a.set_xscale("log")
    a.set_xlabel("fact frequency rank k")
    a.set_ylabel("fraction learned (moving avg)")
    a.set_title(f"(c) Learned in order of frequency (α={a_})")
    a.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "fig_e3_scaling")


# ----------------------------------------------------------------------------- E4
def fig_e4():
    R = load_json("e4_superposition" + SFX)
    S = R["sparsities"]
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.9), gridspec_kw={"width_ratios": [1.3, 1, 1, 1.2]})
    a = ax[0]
    for kind, col, lab in [("relu", C[0], "ReLU readout"), ("linear", C[1], "linear readout")]:
        mm, ss = mean_sd([R[kind][str(s)]["represented"] for s in S], 1)
        a.errorbar(np.arange(len(S)), mm, yerr=ss, color=col, marker="o", capsize=2, label=lab)
    a.axhline(R["m"], color=GRAY, ls="--", lw=1, label=f"m = {R['m']} dims")
    a.axhline(R["n"], color=GRAY, ls=":", lw=1, label=f"n = {R['n']} features")
    a.set_xticks(range(len(S)), [str(s) for s in S], rotation=45, fontsize=7)
    a.set_xlabel("feature sparsity S")
    a.set_ylabel("features represented (‖W_i‖>0.5)")
    a.set_title("(a) Superposition grows with sparsity")
    a.legend(fontsize=6.5)
    for a, s, tag in [(ax[1], S[0], "(b) dense"), (ax[2], S[-2], "(c) sparse")]:
        G = np.asarray(R["relu"][str(s)]["gram"])
        a.imshow(G, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_title(f"{tag}: WᵀW, S={s}")
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    a = ax[3]
    cap = R["capacity"]["results"]
    ks = sorted(cap, key=float)
    a.plot([float(k) for k in ks], [cap[k]["per_dim"] for k in ks], marker="o", color=C[0])
    a.axhline(1, color=GRAY, ls="--", lw=1)
    a.set_xlabel("sparsity S")
    a.set_ylabel("features stored per dimension")
    a.set_title(f"(d) n={R['capacity']['n']}, m={R['capacity']['m']}")
    fig.tight_layout()
    save(fig, "fig_e4_superposition")


# ----------------------------------------------------------------------------- E5
def fig_e5():
    R = load_json("e5_grokking" + SFX)
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))
    a = ax[0]
    h = np.asarray(R["main"][0]["hist"])
    a.plot(h[:, 0], h[:, 2], color=C[1], label="train acc")
    a.plot(h[:, 0], h[:, 3], color=C[0], label="test acc")
    a2 = a.twinx()
    a2.plot(h[:, 0], h[:, 6], color=C[2], lw=1.3, ls="--", label="Fourier concentration")
    a2.set_ylim(0, 1)
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    a.set_xscale("log")
    a.set_xlabel("step")
    a.set_ylabel("accuracy")
    a.set_title("(a) Memorise, then generalise")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="center left", fontsize=6.5)
    a = ax[1]
    for i, r in enumerate(R["main"]):
        h = np.asarray(r["hist"])
        a.plot(h[:, 0], h[:, 5], color=C[i], lw=1.4, label=f"seed {r['seed']}")
    a.set_xscale("log")
    a.set_xlabel("step")
    a.set_ylabel("parameter norm")
    a.set_title("(b) Generalisation = norm collapse")
    a.legend(fontsize=6.5)
    a = ax[2]
    pw = np.asarray(R["main"][0]["power"])
    a.bar(np.arange(1, len(pw) + 1), pw, color=C[0], width=0.8)
    a.set_xlabel("frequency k")
    a.set_ylabel("share of embedding power")
    a.set_title("(c) Final embedding: few frequencies")
    a = ax[3]
    for key, sweep, col, lab in [("wd", R["wd_sweep"] + [dict(r, wd=1.0) for r in R["main"]], C[0], "weight decay (frac 0.3)")]:
        vals = sorted(set(r[key] for r in sweep))
        fin = [np.mean([r["final_test"] for r in sweep if r[key] == v]) for v in vals]
        a.plot(range(len(vals)), fin, marker="o", color=col, label=lab)
        a.set_xticks(range(len(vals)), [str(v) for v in vals])
    fr = R["frac_sweep"] + [r for r in R["main"]]
    vals = sorted(set(r["frac"] for r in fr))
    a3 = a.twiny()
    a3.plot(range(len(vals)), [np.mean([r["final_test"] for r in fr if r["frac"] == v]) for v in vals],
            marker="s", color=C[1], label="train fraction (wd 1.0)")
    a3.set_xticks(range(len(vals)), [str(v) for v in vals], fontsize=7)
    a3.grid(False)
    a.set_ylabel("final test accuracy")
    a.set_xlabel("weight decay")
    a.set_title("(d) Conditions for discovering the rule", pad=18)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a3.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="center right")
    fig.tight_layout()
    save(fig, "fig_e5_grokking")


# ----------------------------------------------------------------------------- E6
def fig_e6():
    R = load_json("e6_incontext" + SFX)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    k = np.arange(R["K"])
    runs = R["runs"]
    a = ax[0]
    inf = [r for r in runs if r["M"] == "inf"][0]["eval"]["fresh"]
    a.plot(k, inf["model"], color=C[0], marker="o", label="Transformer (M=∞)")
    a.plot(k, inf["ridge"], color=C[2], ls="--", label="Bayes (ridge, λ=σ²)")
    if "ols" in inf:
        a.plot(k[1:], inf["ols"][1:], color=C[1], ls=":", label="least squares (spikes at k=d)")
    a.axhline(R["sigma"] ** 2, color=GRAY, lw=1, label="noise floor σ²")
    a.set_yscale("log")
    a.set_ylim(0.05, 20)
    a.set_xlabel("in-context examples")
    a.set_ylabel("MSE on next y")
    a.set_title("(a) In-context inference tracks Bayes")
    a.legend(fontsize=6.5)
    a = ax[1]
    for i, r in enumerate(runs):
        a.plot(k, r["eval"]["fresh"]["model"], color=C[i], lw=1.5, label=f"M={r['M']}")
    a.plot(k, inf["ridge"], color=INK, ls="--", lw=1, label="ridge")
    a.set_yscale("log")
    a.set_xlabel("in-context examples")
    a.set_ylabel("MSE on unseen worlds")
    a.set_title("(b) Generalising to new worlds needs diversity")
    a.legend(fontsize=6.5, ncol=2)
    a = ax[2]
    fin = [r for r in runs if r["M"] != "inf"]
    Ms = [r["M"] for r in fin]
    tail = slice(R["K"] // 2, None)
    a.plot(Ms, [np.mean(r["eval"]["fresh"]["dist_to_ridge"][tail]) for r in fin], marker="o", color=C[2],
           label="distance to ridge (general rule)")
    a.plot(Ms, [np.mean(r["eval"]["fresh"]["dist_to_dmmse"][tail]) for r in fin], marker="s", color=C[1],
           label="distance to dMMSE (memorised pool)")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("number of training worlds M")
    a.set_ylabel("mean sq. distance of predictions")
    a.set_title("(c) Memorise → generalise transition")
    a.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_e6_incontext")


# ----------------------------------------------------------------------------- E7
def fig_e7():
    R = load_json("e7_shortcuts" + SFX)
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    a = ax[0]
    rh = R["rhos"]
    for key, col, lab in [("aligned", C[2], "watermark agrees"), ("clean", C[0], "no watermark"),
                          ("conflict", C[7], "watermark reversed")]:
        mm, ss = mean_sd([R["sweep"][str(r)][key] for r in rh], 1)
        a.errorbar(range(len(rh)), mm, yerr=ss, marker="o", color=col, capsize=2, label=lab)
    a.axhline(0.5, color=GRAY, lw=1, ls="--")
    a.set_xticks(range(len(rh)), [str(r) for r in rh])
    a.set_xlabel("training correlation ρ between watermark and label")
    a.set_ylabel("test accuracy")
    a.set_title("(a) Reliance on the shortcut grows with ρ")
    a.legend(fontsize=6.5, loc="lower left")
    a = ax[1]
    names = ["erm", "group_reweight", "randomise_watermark"]
    labs = ["standard\ntraining", "group\nreweighting", "randomise\nwatermark"]
    x = np.arange(len(names))
    for j, (key, col, lab) in enumerate([("clean", C[0], "no watermark"), ("conflict", C[7], "watermark reversed")]):
        mm, ss = mean_sd([R["remedies"][n][key] for n in names], 1)
        a.bar(x + (j - 0.5) * 0.36, mm, 0.34, yerr=ss, color=col, capsize=2, label=lab)
    a.set_xticks(x, labs, fontsize=7.5)
    a.set_ylim(0, 1.05)
    a.set_ylabel("test accuracy")
    a.set_title("(b) Remedies at ρ = 0.95")
    a.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_e7_shortcuts")


if __name__ == "__main__":
    for name, fn in [("e1_world_model", fig_e1), ("e2_manifold", fig_e2), ("e3_zipf_scaling", fig_e3),
                     ("e4_superposition", fig_e4), ("e5_grokking", fig_e5), ("e6_incontext", fig_e6),
                     ("e7_shortcuts", fig_e7)]:
        if have(name) or (name == "e1_world_model" and list(RESULTS.glob(f"e1_seed[0-9]{SFX}.json"))):
            try:
                fn()
            except Exception as ex:  # keep going so one broken figure doesn't block the rest
                import traceback
                traceback.print_exc()
        else:
            print("missing", name)
