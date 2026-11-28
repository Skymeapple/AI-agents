"""E4: GRPO on a sequence policy with parameter sharing (exploits are *features*).

Responses are binary strings of length L=10.  Gold: correct iff x[0:5] are all 1.
Exploit ('master key'): x[0]==0 and x[8:10]==(1,1)  -- a non-attempt with a judge-fooling
suffix.  Verifier acceptance: alpha if correct, eta if exploit, beta otherwise.
Policies: factorised Bernoulli (L params), first-order autoregressive (2L params) and a
full tabular softmax over all 2^L strings.  All metrics computed exactly by enumeration.
Compared with the theory: exponential tilt of the initial distribution."""
import json, os, sys, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theory import tilted_accuracy, tilted_kl

R = os.path.join(os.path.dirname(__file__), "..", "results")
L = 10
X = np.array(list(itertools.product([0, 1], repeat=L)), dtype=np.int8)  # 1024 x L
CORRECT = X[:, :5].all(1)
EXPLOIT = (X[:, 0] == 0) & (X[:, 8] == 1) & (X[:, 9] == 1)
P0 = np.array([0.6] * 5 + [0.5] * 3 + [0.15] * 2)
PREV = np.concatenate([np.zeros((len(X), 1), np.int8), X[:, :-1]], 1)


def sig(z): return 1 / (1 + np.exp(-z))


def verifier(alpha, beta, eta):
    return np.where(CORRECT, alpha, np.where(EXPLOIT, eta, beta))


class Factorised:
    def __init__(self): self.th = np.log(P0 / (1 - P0))
    def probs(self):
        q = sig(self.th); return np.exp((X * np.log(q) + (1 - X) * np.log(1 - q)).sum(1))
    def grad_logp(self, idx):  # (n, params)
        return X[idx] - sig(self.th)[None, :]


class AR1:
    def __init__(self): self.th = np.tile(np.log(P0 / (1 - P0))[:, None], (1, 2))  # [pos, prev]
    def probs(self):
        q = sig(self.th[np.arange(L)[None, :], PREV])
        return np.exp((X * np.log(q) + (1 - X) * np.log(1 - q)).sum(1))
    def grad_logp(self, idx):
        q = sig(self.th[np.arange(L)[None, :], PREV[idx]])
        g = np.zeros((len(idx), L, 2))
        g[np.arange(len(idx))[:, None], np.arange(L)[None, :], PREV[idx]] = X[idx] - q
        return g.reshape(len(idx), -1)
    def step(self, d): self.th += d.reshape(L, 2)


class Tabular:
    def __init__(self):
        self.th = np.log(Factorised().probs())
    def probs(self):
        z = self.th - self.th.max(); e = np.exp(z); return e / e.sum()
    def grad_logp(self, idx):
        g = -np.tile(self.probs(), (len(idx), 1)); g[np.arange(len(idx)), idx] += 1; return g


def train(pol, f, steps=3000, G=16, B=8, lr=0.5, seed=0, log_every=10):
    rng = np.random.default_rng(seed)
    p0 = pol.probs(); lp0 = np.log(p0); hist = []
    for s in range(steps + 1):
        p = pol.probs()
        if s % log_every == 0:
            hist.append((s, p @ CORRECT, p @ EXPLOIT, p @ (np.log(p + 1e-300) - lp0), p @ f))
        idx = rng.choice(len(X), size=B * G, p=p)
        r = (rng.random(B * G) < f[idx]).astype(float).reshape(B, G)
        sd = r.std(1, keepdims=True)
        adv = np.where(sd > 1e-8, (r - r.mean(1, keepdims=True)) / np.maximum(sd, 1e-8), 0).ravel()
        d = lr * (adv[:, None] * pol.grad_logp(idx)).mean(0)
        if hasattr(pol, "step"): pol.step(d)
        else: pol.th += d
    return np.array(hist), p0


if __name__ == "__main__":
    os.makedirs(R, exist_ok=True)
    settings = {"math-like (α=.75, β=.15, η=.95)": (0.75, 0.15, 0.95),
                "noisy judge (α=.60, β=.40, η=.95)": (0.60, 0.40, 0.95),
                "no exploit (α=.60, β=.40, η=β)": (0.60, 0.40, 0.40)}
    out = {"c0": float(Factorised().probs() @ CORRECT), "h0": float(Factorised().probs() @ EXPLOIT)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, (name, (a, b, e)) in enumerate(settings.items()):
        f = verifier(a, b, e); ax = axes[k]
        p0 = Factorised().probs()
        ts = np.linspace(0, 200, 4001)
        At = tilted_accuracy(f, CORRECT, p0, ts); Kt = tilted_kl(f, p0, ts)
        res = {"theory_peak_A": float(At.max()), "theory_KL_at_peak": float(Kt[np.argmax(At)]),
               "theory_A_at_t200": float(At[-1])}
        ax.plot(Kt, At, "k--", lw=2.5, label="tilt (theory)")
        for j, (pname, cls) in enumerate([("factorised", Factorised), ("AR1", AR1), ("tabular", Tabular)]):
            peaks, finals, fex, klp = [], [], [], []
            for sd in range(3):
                h, _ = train(cls(), f, seed=sd)
                peaks.append(h[:, 1].max()); finals.append(h[-1, 1]); fex.append(h[-1, 2])
                klp.append(h[np.argmax(h[:, 1]), 3])
                ax.plot(h[:, 3], h[:, 1], color=f"C{j}", alpha=0.6, label=pname if sd == 0 else None)
            res[pname] = {"peak_A": [float(x) for x in peaks], "final_A": [float(x) for x in finals],
                          "final_exploit_mass": [float(x) for x in fex], "KL_at_peak": [float(x) for x in klp]}
            print(name, pname, res[pname]); sys.stdout.flush()
        out[name] = res
        ax.set_xscale("symlog", linthresh=0.05); ax.set_xlabel("KL(π‖π0) (nats)"); ax.set_ylabel("true accuracy")
        ax.set_title(name, fontsize=10); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig4_sequence.png"), dpi=150)
    json.dump(out, open(os.path.join(R, "exp4.json"), "w"), indent=2)
