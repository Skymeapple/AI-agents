"""Experiment 6 (added in revision): learned conditional defectors in the regime that
matters, alpha in [1e-6, 1e-3] (alpha ~ c/N for N ~ 1e3..1e6).

Plain Monte Carlo cannot measure alpha ~ 1e-6 with 4e5 samples, and Exp. 1 therefore
excluded alpha <= 1e-4. Here every rate is estimated with a *defensive importance
sampler*: draw x from q = (E + D)/2 and weight by dE/dq = 2/(1+LR) and dD/dq = 2LR/(1+LR),
where LR = dD/dE is known exactly in both synthetic environments. Weights are bounded
by 2, so relative error at alpha ~ 1e-6 is controlled whenever the policy's defection
region has non-negligible D-mass (which it must, for beta >> alpha).

Two trainers per configuration, identical except for the evaluation-term gradient:
  plain : E-term gradient from samples of E (as in Exp. 1)
  IS    : E-term gradient from the defensive IS sampler (lower variance in the tail)
Lambda is annealed geometrically 1 -> lambda over 70% of 3000 steps (as in Exp. 1).
The Bayes frontier in the tail is computed exactly (Gaussian) or by the same IS
estimator applied to the exact log-likelihood ratio with 4e6 draws (XOR).
Run: OMP_NUM_THREADS=1 python3 exp6_lowalpha.py   (~6 min on 4 cores)
"""
import json, os
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import GaussianShift, XORCue, MLP, Adam, RESULTS

LAMS = [64, 128, 256, 512, 1024, 2048]
SEEDS = [0, 1, 2]
CONFIGS = [("gaussian", 0), ("gaussian", 16), ("xor", 8), ("xor", 32)]


def make_env(name):
    return GaussianShift(8, 1.5, 0) if name == "gaussian" else XORCue(8, 1.2)


def log_lr(env, x):
    if isinstance(env, GaussianShift):
        return x @ env.mu - 0.5 * env.dprime ** 2
    return env.log_lr(x)


def sample_q(env, n, rng):
    """Defensive mixture q = (E + D)/2 and IS weights for E and D."""
    k = rng.random(n) < 0.5
    x = np.where(k[:, None], env.sample(n, "D", rng), env.sample(n, "E", rng))
    ll = np.clip(log_lr(env, x), -700, 700)
    lr = np.exp(ll)
    wE = 2.0 / (1.0 + lr)
    wD = 2.0 * lr / (1.0 + lr)
    return x, wE, wD


def is_rates(model, env, rng, n=1_000_000, chunks=4):
    a = b = 0.0
    a2 = 0.0
    for _ in range(chunks):
        x, wE, wD = sample_q(env, n // chunks, rng)
        p = model.prob(x)
        a += np.sum(p * wE); b += np.sum(p * wD); a2 += np.sum((p * wE) ** 2)
    a /= n; b /= n
    se_a = np.sqrt(max(a2 / n - a * a, 0) / n)
    return float(a), float(b), float(se_a)


def train(env, hidden, lam_final, rng, mode, steps=3000, batch=4096, lr=1e-2, warm=0.7):
    m = MLP(env.k, hidden, rng)
    opt = Adam(m.params, lr=lr)
    for t in range(steps):
        lam = lam_final ** min(1.0, t / (warm * steps))
        xd = env.sample(batch, "D", rng)
        g = m.grad_mean_prob(xd, np.full(batch, 1.0 / batch))
        if mode == "plain":
            xe = env.sample(batch, "E", rng)
            ge = m.grad_mean_prob(xe, np.full(batch, -lam / batch))
        else:
            xq, wE, _ = sample_q(env, batch, rng)
            ge = m.grad_mean_prob(xq, -lam * wE / batch)
        g = {k: g[k] + ge[k] for k in g}
        opt.ascend(m.params, g)
    return m


def job(args):
    envname, h, lam, s, mode = args
    env = make_env(envname)
    rng = np.random.default_rng(90001 * s + 7 * int(np.log2(lam)) + h + (0 if envname == "gaussian" else 3)
                                + (0 if mode == "plain" else 50000))
    m = train(env, h, lam, rng, mode)
    a, b, se = is_rates(m, env, np.random.default_rng(777 + s))
    return dict(env=envname, width=h, lam=lam, seed=s, mode=mode, alpha=a, beta=b, alpha_se=se)


def xor_tail_frontier(n=4_000_000, seed=11):
    env = XORCue(8, 1.2)
    rng = np.random.default_rng(seed)
    x, wE, wD = sample_q(env, n, rng)
    ll = env.log_lr(x)
    o = np.argsort(-ll)
    A = np.cumsum(wE[o]) / n
    B = np.cumsum(wD[o]) / n
    return A, B


def main():
    jobs = [(e, h, l, s, mode) for (e, h) in CONFIGS for l in LAMS for s in SEEDS for mode in ["plain", "IS"]]
    with Pool(4) as p:
        pts = p.map(job, jobs)
    A, B = xor_tail_frontier()
    g = GaussianShift(8, 1.5, 0)

    def frontier(env, a):
        a = max(a, 1e-9)
        if env == "gaussian":
            return float(g.bayes_roc(a))
        return float(np.interp(np.log(a), np.log(np.maximum(A, 1e-300)), B))

    for q in pts:
        q["frontier"] = frontier(q["env"], q["alpha"])
        q["efficiency"] = q["beta"] / q["frontier"] if q["frontier"] > 0 else None
        q["excess"] = q["beta"] - q["frontier"]
        q["rel_se_alpha"] = q["alpha_se"] / q["alpha"] if q["alpha"] > 0 else None

    bins = [(1e-7, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3)]
    summary = {}
    for (e, h) in CONFIGS:
        for mode in ["plain", "IS"]:
            sel = [q for q in pts if q["env"] == e and q["width"] == h and q["mode"] == mode]
            row = {}
            for lo, hi in bins:
                s2 = [q for q in sel if lo < q["alpha"] <= hi]
                eff = np.array([q["efficiency"] for q in s2])
                row[f"{lo:.0e}-{hi:.0e}"] = dict(
                    n=len(s2),
                    eff_mean=float(eff.mean()) if len(s2) else None,
                    eff_median=float(np.median(eff)) if len(s2) else None,
                    eff_min=float(eff.min()) if len(s2) else None,
                    n_collapsed=int(np.sum(eff < 0.1)) if len(s2) else 0)
            row["n_alpha_le_1e-7"] = int(sum(q["alpha"] <= 1e-7 for q in sel))
            row["max_excess"] = float(max(q["excess"] for q in sel))
            summary[f"{e}_w{h}_{mode}"] = row
    out = dict(points=pts, summary=summary,
               max_excess_all=float(max(q["excess"] for q in pts)),
               max_rel_se_alpha=float(max(q["rel_se_alpha"] for q in pts if q["alpha"] > 1e-8)),
               xor_tail=dict(alpha=[float(np.interp(b_, B, A)) for b_ in []]))
    out["xor_frontier_at"] = {f"{a0:.0e}": frontier("xor", a0) for a0 in [1e-6, 1e-5, 1e-4, 1e-3]}
    out["gauss_frontier_at"] = {f"{a0:.0e}": frontier("gaussian", a0) for a0 in [1e-6, 1e-5, 1e-4, 1e-3]}
    json.dump(out, open(os.path.join(RESULTS, "exp6_lowalpha.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("max excess", out["max_excess_all"], "max rel se", out["max_rel_se_alpha"])
    print(out["xor_frontier_at"], out["gauss_frontier_at"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    al = np.logspace(-7, -2, 200)
    for k, (e, widths) in enumerate([("gaussian", [0, 16]), ("xor", [8, 32])]):
        ax[k].plot(al, [frontier(e, a) for a in al], "k-", lw=1.5, label="Bayes frontier")
        ax[k].plot(al, al, ":", color="gray", lw=1)
        for h, mk in zip(widths, ["o", "s"]):
            for mode, col in [("plain", "#1f77b4"), ("IS", "#d62728")]:
                sel = [q for q in pts if q["env"] == e and q["width"] == h and q["mode"] == mode]
                ax[k].scatter([max(q["alpha"], 1e-7) for q in sel], [max(q["beta"], 1e-7) for q in sel],
                              marker=mk, s=18, color=col, alpha=0.7,
                              label=f"{'linear' if h == 0 else 'width ' + str(h)}, {mode} gradient")
        ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlim(1e-7, 1e-2); ax[k].set_ylim(1e-7, 1)
        ax[k].set_xlabel("evaluation defection rate α (IS estimate)")
        ax[k].set_title(f"({'ab'[k]}) {'Gaussian d′=1.5' if e == 'gaussian' else 'XOR cue'}: low-α regime")
        ax[k].legend(fontsize=7)
    ax[0].set_ylabel("deployment defection rate β")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig6_lowalpha.png"), dpi=150)


if __name__ == "__main__":
    main()
