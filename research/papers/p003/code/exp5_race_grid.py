"""E5: collapse probability of GRPO on the sequence task over a grid of
honest discrimination J = alpha - beta and exploit base mass h0, for factorised and AR1
policies.  Also records the offline balanced accuracy of the verifier on base samples
and the number of steps to reach 90% true accuracy (learning speed)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import exp4_sequence as E

R = os.path.join(os.path.dirname(__file__), "..", "results")
BETA, ETA = 0.15, 0.95
Js = [0.10, 0.20, 0.30, 0.45, 0.60, 0.80]
SUFFIX_P = [0.08, 0.15, 0.25]
SEEDS = 8; STEPS = 3000

if __name__ == "__main__":
    rows = []
    for ps in SUFFIX_P:
        E.P0 = np.array([0.6] * 5 + [0.5] * 3 + [ps] * 2)
        p0 = E.Factorised().probs(); h0 = float(p0 @ E.EXPLOIT); c0 = float(p0 @ E.CORRECT)
        for J in Js:
            alpha = BETA + J
            f = E.verifier(alpha, BETA, ETA)
            wrong = ~E.CORRECT
            fpr_off = float(p0[wrong] @ f[wrong] / p0[wrong].sum())
            ba = 0.5 * (alpha + 1 - fpr_off)
            for pname, cls in [("factorised", E.Factorised), ("AR1", E.AR1)]:
                col, peaks, t90 = [], [], []
                for sd in range(SEEDS):
                    h, _ = E.train(cls(), f, steps=STEPS, seed=sd, log_every=10)
                    col.append(h[-1, 2] > 0.5); peaks.append(h[:, 1].max())
                    hit = np.where(h[:, 1] >= 0.9)[0]
                    t90.append(float(h[hit[0], 0]) if len(hit) else np.nan)
                row = {"suffix_p": ps, "h0": h0, "c0": c0, "J": J, "alpha": alpha, "policy": pname,
                       "offline_FPR": fpr_off, "offline_balanced_acc": ba,
                       "collapse_frac": float(np.mean(col)), "peak_A_mean": float(np.mean(peaks)),
                       "steps_to_A90_median": float(np.nanmedian(t90)) if np.any(~np.isnan(t90)) else None}
                rows.append(row); print(row); sys.stdout.flush()
    json.dump(rows, open(os.path.join(R, "exp5.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for k, pname in enumerate(["factorised", "AR1"]):
        for i, ps in enumerate(SUFFIX_P):
            rr = [r for r in rows if r["policy"] == pname and r["suffix_p"] == ps]
            ax[k].plot([r["J"] for r in rr], [r["collapse_frac"] for r in rr], "-o",
                       label=f"h0={rr[0]['h0']:.4f}")
        ax[k].set_xlabel("honest discrimination J = α − β"); ax[k].set_ylabel(f"collapse fraction ({SEEDS} seeds)")
        ax[k].set_title(f"{pname} policy: exploit takeover"); ax[k].legend(fontsize=8)
    rr = [r for r in rows if r["policy"] == "factorised"]
    sc = ax[2].scatter([r["offline_balanced_acc"] for r in rr], [r["collapse_frac"] for r in rr],
                       c=[np.log10(r["h0"]) for r in rr], cmap="viridis")
    plt.colorbar(sc, ax=ax[2], label="log10 h0")
    ax[2].set_xlabel("offline balanced accuracy of verifier (on π0 samples)")
    ax[2].set_ylabel("collapse fraction (factorised)"); ax[2].set_title("Offline accuracy vs RL fate")
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig5_race_grid.png"), dpi=150)
