"""E6: does a KL penalty to the reference policy protect against a rare exploit?
On the sequence task (math-like verifier, alpha=.75, beta=.15, eta=.95, h0=0.009)
unregularised GRPO starves the exploit.  The KL-regularised objective's optimum is the
exponential tilt of pi0, which keeps exploit mass alive.  We train GRPO with a score-function
KL penalty for several lam and compare final accuracy with the tilt prediction
A(t_eff), t_eff = 1/(lam*sigma*) (self-consistent, std-normalised advantages)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import exp4_sequence as E
from theory import tilted_accuracy

R = os.path.join(os.path.dirname(__file__), "..", "results")


def train_kl(pol, f, lam, steps=8000, G=16, B=8, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    lp0 = np.log(pol.probs()); hist = []
    for s in range(steps + 1):
        p = pol.probs(); lp = np.log(p + 1e-300)
        if s % 50 == 0:
            hist.append((s, p @ E.CORRECT, p @ E.EXPLOIT, p @ (lp - lp0)))
        idx = rng.choice(len(E.X), size=B * G, p=p)
        r = (rng.random(B * G) < f[idx]).astype(float).reshape(B, G)
        sd = r.std(1, keepdims=True)
        adv = np.where(sd > 1e-8, (r - r.mean(1, keepdims=True)) / np.maximum(sd, 1e-8), 0).ravel()
        kl_term = lp[idx] - lp0[idx]; kl_term = kl_term - kl_term.mean()  # baseline
        d = lr * ((adv - lam * kl_term)[:, None] * pol.grad_logp(idx)).mean(0)
        if hasattr(pol, "step"): pol.step(d)
        else: pol.th += d
    return np.array(hist)


def t_eff(w, f, lam):
    t = 1.0 / lam
    for _ in range(300):
        lw = np.log(w) + t * f; p = np.exp(lw - lw.max()); p /= p.sum(); fb = p @ f
        t = 0.5 * t + 0.5 / (lam * np.sqrt(fb * (1 - fb)))
    return t


if __name__ == "__main__":
    f = E.verifier(0.75, 0.15, 0.95)
    p0 = E.Factorised().probs()
    rows = []
    for lam in [0.0, 0.5, 0.3, 0.2, 0.1, 0.05]:
        te = t_eff(p0, f, lam) if lam > 0 else np.inf
        pred = float(tilted_accuracy(f, E.CORRECT, p0, te)[0]) if lam > 0 else None
        row = {"lam": lam, "t_eff": None if lam == 0 else float(te), "tilt_pred_A": pred}
        for pname, cls in [("tabular", E.Tabular), ("factorised", E.Factorised)]:
            hs = [train_kl(cls(), f, lam, seed=s) for s in range(3)]
            row[pname] = {"final_A": [float(h[-1, 1]) for h in hs],
                          "final_exploit": [float(h[-1, 2]) for h in hs],
                          "final_KL": [float(h[-1, 3]) for h in hs],
                          "A_last20pct_mean": float(np.mean([h[-len(h)//5:, 1].mean() for h in hs]))}
        rows.append(row); print(row); sys.stdout.flush()
    json.dump(rows, open(os.path.join(R, "exp6.json"), "w"), indent=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    lr_ = [r for r in rows if r["lam"] > 0]
    x = [r["lam"] for r in lr_]
    ax.plot(x, [r["tilt_pred_A"] for r in lr_], "k--o", label="tilt prediction A(t_eff)")
    for j, pn in enumerate(["tabular", "factorised"]):
        ax.plot(x, [np.mean(r[pn]["final_A"]) for r in lr_], f"C{j}-s", label=f"KL-GRPO {pn}")
        ax.axhline(np.mean(rows[0][pn]["final_A"]), color=f"C{j}", ls=":", label=f"λ=0 {pn}")
    ax.set_xscale("log"); ax.set_xlabel("KL coefficient λ"); ax.set_ylabel("true accuracy after 8000 steps")
    ax.set_title("KL penalty vs rare exploit (math-like verifier)"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig6_kl_paradox.png"), dpi=150)
