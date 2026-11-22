"""Make all figures from saved results (no re-computation)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED = "#222222", "#777777"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "lines.linewidth": 2})


def fig1():
    cf = json.load(open(os.path.join(R, "e1_closed_form.json")))
    mc = json.load(open(os.path.join(R, "e1_montecarlo_summary.json")))
    n = np.array(cf["n"])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, k in enumerate(["1", "4", "16", "64"]):
        ax[0].plot(n, cf["lead_zero_floor"][k], color=C[i], label=f"k={k}")
        ax[1].plot(n, cf["lead_chance_floor_m4"][k], color=C[i], label=f"k={k}")
    ax[0].plot(n, cf["lead_k_inf"], color=INK, ls="--", lw=1.2, label="k→∞ limit")
    for i, k in enumerate([1, 4, 16]):
        rows = [r for r in mc if r["k"] == k and r["s"] == 0 and not r["kink"]]
        ax[0].errorbar([r["n"] for r in rows], [r["lead_detect_mean"] for r in rows],
                       yerr=[r["lead_detect_sd"] for r in rows], fmt="o", ms=5, color=C[i],
                       mec="white", mew=1, capsize=2, lw=1)
    for a, t in zip(ax, ["Exact match, zero floor (Thm 1)", "4-way multiple choice (Cor. 3)"]):
        a.set_xscale("log"); a.set_xlabel("benchmark size n (items)"); a.set_title(t, fontsize=9, color=INK)
    ax[0].set_ylabel("detection lead time (decades, β=2)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_detection_lead.png"), dpi=180); plt.close(fig)


def fig2():
    mc = json.load(open(os.path.join(R, "e1_montecarlo_summary.json")))
    scen = [(0.0, False, "homogeneous items, smooth latent"), (1.0, False, "item spread s=1, smooth latent"),
            (0.0, True, "homogeneous, hidden latent kink")]
    keys = [("lead_detect", "detection (model-free)"), ("leadfc_acc", "logistic fit to accuracy"),
            ("leadfc_proxy", "mean log-lik, correct family"), ("leadfc_proxy_lin", "mean log-lik, linear"),
            ("leadfc_proxy_item", "item-level log-lik")]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, (s, kink, title) in zip(axs, scen):
        rows = [r for r in mc if r["s"] == s and r["kink"] == kink and r["n"] == 1000]
        x = np.arange(len(rows)); w = 0.16
        for i, (kk, lab) in enumerate(keys):
            v = [r[kk + "_mean"] if kk != "lead_detect" else r["lead_detect_mean"] for r in rows]
            ax.bar(x + (i - 2) * w, v, w * 0.88, color=C[i], label=lab)
        ax.plot(x, [r["max_possible_lead"] for r in rows], "_", ms=28, color=INK, mew=1.5)
        ax.set_xticks(x); ax.set_xticklabels([f"k={r['k']}" for r in rows]); ax.set_title(title, fontsize=9, color=INK)
    axs[0].set_ylabel("lead time (decades of compute), n=1000")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_forecaster_lead.png"), dpi=180); plt.close(fig)


def fig3():
    p = os.path.join(R, "e2_teacher_student.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    cv = d["curves"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, k in enumerate(["4", "8", "16"]):
        for j, c in enumerate(cv):
            ax[0].plot(c["x"], c["acc"][k], color=C[i], lw=1.2, alpha=0.6, label=f"k={k}" if j == 0 else None)
            ax[1].plot(c["x"], c["logp"][k], color=C[i], lw=1.2, alpha=0.6, label=f"k={k}" if j == 0 else None)
    ax[0].axhline(0.5, color=MUTED, lw=1, ls=":")
    ax[0].set_ylabel("expected pass@1 (exact match)"); ax[1].set_ylabel("mean log-lik of correct answer")
    for a in ax:
        a.set_xlabel("log10 training steps"); a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_teacher_student_curves.png"), dpi=180); plt.close(fig)


def fig4():
    p = os.path.join(R, "e3_parity_n30.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    runs = [r for r in res if not r["control"]][:4]
    ctrl = [r for r in res if r["control"]][:3]
    panels = [("acc", "test accuracy"), ("logp", "mean log p(correct)"), ("lwS", "log support/off-support\nweight-norm ratio (ΔlwS)")]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, (ind, lab) in zip(axs, panels):
        for i, r in enumerate(runs):
            st = np.array(r["step"]); v = np.array(r[ind]); i0 = np.searchsorted(st, 100)
            y = v - v[i0] if ind == "lwS" else v
            ax.plot(st, y, color=C[i], lw=1.5, label=f"seed {r['seed']}")
        for c in ctrl:
            st = np.array(c["step"]); v = np.array(c[ind]); i0 = np.searchsorted(st, 100)
            y = v - v[i0] if ind == "lwS" else v
            ax.plot(st, y, color=MUTED, lw=1, ls="--", label="control" if c is ctrl[0] else None)
        ax.set_xscale("log"); ax.set_xlabel("training step"); ax.set_title(lab, fontsize=9, color=INK)
        if ind == "lwS":
            ax.set_ylim(-0.02, 0.06)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig4_parity_indicators.png"), dpi=180); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("figures written")
