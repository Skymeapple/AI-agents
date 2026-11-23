"""E2: finite-sample GRPO on a small tabular (realizable) mode universe.

Part A: KL-regularised GRPO converges to the exponential tilt of the verifier score.
        With mean-centred advantages the tilt is t = 1/lam; with std-normalised (GRPO)
        advantages it is t = 1/(lam * sigma*), sigma* = sqrt(fbar*(1-fbar*)) at the fixed point.
Part B: unregularised dynamics, sweep of exploit mass h0.  Compares
        (1) the replicator / tilted family (theory; = natural-gradient flow),
        (2) the deterministic expected-gradient vanilla softmax PG flow,
        (3) finite-sample GRPO (vanilla softmax PG with group-normalised advantages)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theory import tilted_accuracy, tilted_kl

R = os.path.join(os.path.dirname(__file__), "..", "results")


def make_u20(h0, c0=0.2, seed=0):
    rng = np.random.default_rng(seed)
    fc = np.array([0.55, 0.60, 0.65, 0.70]); mc = rng.dirichlet(np.ones(4)) * c0
    fw = rng.uniform(0.05, 0.45, 15); mw = rng.dirichlet(np.ones(15)) * (1 - c0 - h0)
    f = np.concatenate([fc, fw, [0.95]]); w = np.concatenate([mc, mw, [h0]])
    y = np.zeros(20, bool); y[:4] = True
    return w, y, f


def softmax(th):
    z = th - th.max(); e = np.exp(z); return e / e.sum()


def grpo(w0, y, f, lam=0.0, steps=10000, G=16, B=8, lr=1.0, seed=0, norm=True, log_every=50):
    rng = np.random.default_rng(seed)
    M = len(w0); th = np.log(w0).copy(); lw0 = np.log(w0); hist = []
    for s in range(steps + 1):
        p = softmax(th)
        if s % log_every == 0:
            hist.append((s, p @ y, p @ (np.log(p + 1e-300) - lw0), p[-1]))
        a = rng.choice(M, size=(B, G), p=p)
        r = (rng.random((B, G)) < f[a]).astype(float)
        adv = r - r.mean(1, keepdims=True)
        if norm:
            sd = r.std(1, keepdims=True); adv = np.where(sd > 1e-8, adv / np.maximum(sd, 1e-8), 0.0)
        g = np.zeros(M); np.add.at(g, a.ravel(), adv.ravel() / (G * B))
        if lam > 0:
            l = np.log(p + 1e-300) - lw0; g -= lam * p * (l - p @ l)
        th += lr * g
    return np.array(hist)


def vanilla_flow(w0, y, f, T=10000, lr=1.0, norm=True, log_every=50):
    """Deterministic expected-gradient softmax PG (population limit of GRPO)."""
    th = np.log(w0).copy(); lw0 = np.log(w0); hist = []
    for s in range(T + 1):
        p = softmax(th); fb = p @ f
        if s % log_every == 0:
            hist.append((s, p @ y, p @ (np.log(p + 1e-300) - lw0), p[-1]))
        sc = np.sqrt(max(fb * (1 - fb), 1e-8)) if norm else 1.0
        th += lr * p * (f - fb) / sc
    return np.array(hist)


def selfconsistent_t(w, y, f, lam):
    """Fixed point of t = 1/(lam*sigma(t)), sigma(t)=sqrt(fbar_t(1-fbar_t))."""
    t = 1.0 / lam
    for _ in range(200):
        lw = np.log(w) + t * f; p = np.exp(lw - lw.max()); p /= p.sum(); fb = p @ f
        t = 0.5 * t + 0.5 / (lam * np.sqrt(fb * (1 - fb)))
    return t


if __name__ == "__main__":
    os.makedirs(R, exist_ok=True); out = {}
    # ---------------- Part A ----------------
    w, y, f = make_u20(0.02)
    lams = [1.0, 0.5, 0.25, 0.15, 0.1, 0.07]
    partA = []
    for lam in lams:
        row = {"lam": lam}
        for norm in [False, True]:
            accs = [grpo(w, y, f, lam=lam, steps=15000, seed=s, norm=norm, log_every=500)[-6:, 1].mean()
                    for s in range(3)]
            t = selfconsistent_t(w, y, f, lam) if norm else 1.0 / lam
            key = "std" if norm else "centred"
            row[key] = {"t_pred": float(t), "A_pred": float(tilted_accuracy(f, y, w, t)[0]),
                        "A_grpo_mean": float(np.mean(accs)), "A_grpo_sd": float(np.std(accs))}
        partA.append(row); print(row); sys.stdout.flush()
    out["partA_kl_regularised"] = partA

    # ---------------- Part B ----------------
    h0s = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    STEPS = 20000
    partB = []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for h0 in h0s:
        w, y, f = make_u20(h0)
        ts = np.linspace(0, 400, 8001); At = tilted_accuracy(f, y, w, ts)
        vf = vanilla_flow(w, y, f, T=STEPS, log_every=5)
        gr = [grpo(w, y, f, steps=STEPS, seed=s, log_every=5) for s in range(3)]
        row = {"h0": h0, "theory_peak_A": float(At.max()), "theory_final_A(t=400)": float(At[-1]),
               "vanilla_flow_peak_A": float(vf[:, 1].max()), "vanilla_flow_final_A": float(vf[-1, 1]),
               "vanilla_flow_final_exploit": float(vf[-1, 3]),
               "grpo_peak_A": [float(g[:, 1].max()) for g in gr],
               "grpo_final_A": [float(g[-1, 1]) for g in gr],
               "grpo_final_exploit": [float(g[-1, 3]) for g in gr]}
        partB.append(row); print(row); sys.stdout.flush()
        if h0 in (1e-3, 3e-2):
            k = 0 if h0 == 1e-3 else 1
            KLt = tilted_kl(f, w, ts)
            ax[k].plot(KLt, At, "k--", lw=2, label="tilt / replicator (theory)")
            ax[k].plot(vf[:, 2], vf[:, 1], "C1", lw=2, label="expected-gradient softmax PG")
            for j, g in enumerate(gr):
                ax[k].plot(g[:, 2], g[:, 1], "C0", alpha=0.6, label="finite GRPO" if j == 0 else None)
            ax[k].set_xlabel("KL(π‖π0)"); ax[k].set_ylabel("true accuracy")
            ax[k].set_title(f"exploit mass h0={h0:g}"); ax[k].legend(fontsize=8)
    hh = np.array(h0s)
    ax[2].semilogx(hh, [r["theory_peak_A"] for r in partB], "k--o", label="theory peak A*")
    ax[2].semilogx(hh, [r["vanilla_flow_final_A"] for r in partB], "C1-s", label=f"softmax-PG flow, A at step {STEPS}")
    ax[2].semilogx(hh, [np.mean(r["grpo_final_A"]) for r in partB], "C0-^", label=f"GRPO, A at step {STEPS}")
    ax[2].semilogx(hh, [r["theory_final_A(t=400)"] for r in partB], "k:", label="theory A(t=400)")
    ax[2].set_xlabel("exploit base mass h0"); ax[2].set_ylabel("true accuracy"); ax[2].legend(fontsize=7)
    ax[2].set_title("Fate vs exploit mass")
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig2_tabular_grpo.png"), dpi=150)
    out["partB_h0_sweep"] = partB
    json.dump(out, open(os.path.join(R, "exp2.json"), "w"), indent=2)
