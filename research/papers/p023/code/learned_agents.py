"""Experiment L: learned (small-MLP) agents on a compositional task.

Tests (i) whether the two-parameter beta-binomial model (p, rho), fit on a small
pilot, predicts held-out majority-vote scaling; (ii) whether manipulating the
diversity of agents' training data moves rho and, through it, the
redundancy-vs-division crossover hand-off error eps*, as Theorem 2 predicts.

Run: python3 code/learned_agents.py   (about 2-4 minutes on 4 CPU cores)
"""
import json, os, sys, time
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from theory import vote_acc, ceiling

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
K, d, H = 4, 8, 32
N_POOL, N_TEST = 400, 2000
N_AGENTS = 41           # per type (generalist) / per subtask (specialist)
PILOT = 8               # agents used to fit (p, rho)
EPOCHS, LR = 400, 0.01
LABEL_NOISE = 0.1
SEED = int(os.environ.get("SEED", "0"))
master = np.random.default_rng(7 + 1000 * SEED)

# ---- teacher: subtask k label depends only on block k ----
Wt1 = master.normal(size=(K, d, 16)) / np.sqrt(d)
Wt2 = master.normal(size=(K, 16))
bt = master.normal(size=(K, 16)) * 0.5


def gen(n, rng, noise=LABEL_NOISE):
    X = rng.normal(size=(n, K * d))
    Y = np.zeros((n, K))
    for k in range(K):
        z = np.tanh(X[:, k * d:(k + 1) * d] @ Wt1[k] + bt[k]) @ Wt2[k]
        Y[:, k] = z > 0
    Y = Y.astype(float)
    if noise > 0:
        flip = rng.random(Y.shape) < noise
        Y = np.where(flip, 1 - Y, Y)
    return X, Y


def train_mlp(X, Y, rng):
    """1-hidden-layer tanh MLP, sigmoid outputs, BCE, Adam, full batch, weight decay."""
    n, din = X.shape; dout = Y.shape[1]
    W1 = rng.normal(size=(din, H)) / np.sqrt(din); b1 = np.zeros(H)
    W2 = rng.normal(size=(H, dout)) / np.sqrt(H); b2 = np.zeros(dout)
    P = [W1, b1, W2, b2]; M = [np.zeros_like(q) for q in P]; V = [np.zeros_like(q) for q in P]
    for t in range(1, EPOCHS + 1):
        h = np.tanh(X @ W1 + b1); o = 1 / (1 + np.exp(-(h @ W2 + b2)))
        g_o = (o - Y) / n
        gW2 = h.T @ g_o + 1e-4 * W2; gb2 = g_o.sum(0)
        g_h = (g_o @ W2.T) * (1 - h ** 2)
        gW1 = X.T @ g_h + 1e-4 * W1; gb1 = g_h.sum(0)
        for i, g in enumerate([gW1, gb1, gW2, gb2]):
            M[i] = 0.9 * M[i] + 0.1 * g; V[i] = 0.999 * V[i] + 0.001 * g * g
            P[i] -= LR * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
    return lambda Z: (1 / (1 + np.exp(-(np.tanh(Z @ W1 + b1) @ W2 + b2)))) > 0.5


# fixed shared pool and clean test set (labels thresholded at 0 for consistency)
Xpool, Ypool = gen(N_POOL, np.random.default_rng(1 + 1000 * SEED))
Xte, Yte = gen(N_TEST, np.random.default_rng(2 + 1000 * SEED), noise=0.0)


def agent_data(cond, rng):
    if cond == "shared":        # identical data, different init
        return Xpool, Ypool
    if cond == "bootstrap":     # bootstrap resample of shared pool
        idx = rng.integers(0, N_POOL, N_POOL); return Xpool[idx], Ypool[idx]
    if cond == "independent":   # fresh i.i.d. data per agent
        return gen(N_POOL, rng)
    raise ValueError


def run_condition(cond, seed):
    rng = np.random.default_rng(seed)
    corrG = np.zeros((N_AGENTS, N_TEST, K), bool)
    corrS = np.zeros((N_AGENTS, N_TEST, K), bool)
    for a in range(N_AGENTS):
        X, Y = agent_data(cond, rng)
        f = train_mlp(X, Y, rng)
        corrG[a] = f(Xte) == Yte.astype(bool)
        for k in range(K):
            sl = slice(k * d, (k + 1) * d)
            fk = train_mlp(X[:, sl], Y[:, k:k + 1], rng)
            corrS[a, :, k] = fk(Xte[:, sl])[:, 0] == Yte[:, k].astype(bool)
    return corrG, corrS


def fit_p_rho(C):
    """Method of moments on an agents x items 0/1 matrix."""
    na = C.shape[0]; m = C.mean(0); p = m.mean()
    v = m.var(ddof=1)
    rho = (v / (p * (1 - p)) - 1 / na) / (1 - 1 / na)
    return float(p), float(np.clip(rho, 1e-4, 0.99))


def observed_vote_curve(C, ns, rng, reps=200):
    """Average majority-vote accuracy over random agent subsets of size n."""
    out = []
    for n in ns:
        acc = 0
        for _ in range(reps):
            idx = rng.choice(C.shape[0], n, replace=False)
            acc += (C[idx].mean(0) > 0.5).mean()
        out.append(acc / reps)
    return np.array(out)


def team_curve(corr, ns, rng, reps=200):
    """Fraction of items with all K subtasks correct under per-subtask majority vote.
    corr: agents x items x K."""
    out = []
    for n in ns:
        acc = 0
        for _ in range(reps):
            idx = rng.choice(corr.shape[0], n, replace=False)
            acc += (corr[idx].mean(0) > 0.5).all(1).mean()
        out.append(acc / reps)
    return np.array(out)


if __name__ == "__main__":
    t0 = time.time()
    ns = [1, 3, 5, 9, 15, 21, 33]
    cal = np.arange(N_TEST) < N_TEST // 2          # items for pilot fit
    hold = ~cal                                    # held-out items for evaluation
    R = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ci, cond in enumerate(["shared", "bootstrap", "independent"]):
        corrG, corrS = run_condition(cond, 100 + ci + 1000 * SEED)
        rng = np.random.default_rng(5)
        pilot, rest = slice(0, PILOT), slice(PILOT, N_AGENTS)
        res = {"subtasks": {}}
        maeB, maeI = [], []
        for typ, C in [("G", corrG), ("S", corrS)]:
            for k in range(K):
                p, rho = fit_p_rho(C[pilot][:, cal, k])
                obs = observed_vote_curve(C[rest][:, hold, k], ns, rng)
                predB = np.array([vote_acc(n, p, rho) for n in ns])
                predI = np.array([vote_acc(n, p, 0.0) for n in ns])
                p_full, rho_full = fit_p_rho(C[:, :, k])
                maeB.append(np.abs(obs - predB).mean()); maeI.append(np.abs(obs - predI).mean())
                res["subtasks"][f"{typ}{k}"] = dict(p_pilot=p, rho_pilot=rho, p_all=p_full, rho_all=rho_full,
                                                   obs=obs.tolist(), pred_beta=predB.tolist(), pred_iid=predI.tolist(),
                                                   ceiling_beta=ceiling(p, rho),
                                                   frac_items_theta_gt_half_all41=float((C[:, :, k].mean(0) > 0.5).mean()))
                if k == 0:
                    ax = axes[ci]
                    col = "C0" if typ == "G" else "C1"
                    ax.plot(ns, obs, "o", color=col, label=f"{typ}: observed")
                    ax.plot(ns, predB, "-", color=col, label=f"{typ}: beta-binomial (pilot fit)")
                    ax.plot(ns, predI, ":", color=col, label=f"{typ}: i.i.d. prediction")
        res["MAE_beta"] = float(np.mean(maeB)); res["MAE_iid"] = float(np.mean(maeI))
        # cross-subtask error correlation within generalists (independence assumption check)
        eG = ~corrG[:, hold, :]
        cc = [np.corrcoef(eG[:, :, i].ravel(), eG[:, :, j].ravel())[0, 1] for i in range(K) for j in range(i + 1, K)]
        res["generalist_cross_subtask_error_corr_mean"] = float(np.mean(cc))
        # team-level crossover eps*: observed vs predicted
        AG = team_curve(corrG[rest][:, hold], ns, rng)
        AD = team_curve(corrS[rest][:, hold], ns, rng)
        eps_obs = 1 - np.clip(AG / AD, 0, None) ** (1 / (K - 1))
        predG = np.array([np.prod([vote_acc(n, *fit_p_rho(corrG[pilot][:, cal, k])) for k in range(K)]) for n in ns])
        predD = np.array([np.prod([vote_acc(n, *fit_p_rho(corrS[pilot][:, cal, k])) for k in range(K)]) for n in ns])
        eps_pred = 1 - (predG / predD) ** (1 / (K - 1))
        res.update(ns=ns, team_G_obs=AG.tolist(), team_D_obs_eps0=AD.tolist(), team_G_pred=predG.tolist(),
                   team_D_pred=predD.tolist(), eps_star_obs=eps_obs.tolist(), eps_star_pred=eps_pred.tolist(),
                   rho_mean_G=float(np.mean([res["subtasks"][f"G{k}"]["rho_all"] for k in range(K)])),
                   rho_mean_S=float(np.mean([res["subtasks"][f"S{k}"]["rho_all"] for k in range(K)])),
                   p_mean_G=float(np.mean([res["subtasks"][f"G{k}"]["p_all"] for k in range(K)])),
                   p_mean_S=float(np.mean([res["subtasks"][f"S{k}"]["p_all"] for k in range(K)])))
        R[cond] = res
        axes[ci].set_title(f"{cond} data: rho_G={res['rho_mean_G']:.2f}, rho_S={res['rho_mean_S']:.2f}")
        axes[ci].set_xlabel("agents voting (n)"); axes[ci].set_ylabel("subtask-0 vote accuracy (held-out)")
        print(cond, "pG=%.3f pS=%.3f rhoG=%.3f rhoS=%.3f MAEbeta=%.4f MAEiid=%.4f xcorr=%.3f" % (
            res["p_mean_G"], res["p_mean_S"], res["rho_mean_G"], res["rho_mean_S"], res["MAE_beta"], res["MAE_iid"],
            res["generalist_cross_subtask_error_corr_mean"]), "eps_obs", np.round(eps_obs, 3), "eps_pred", np.round(eps_pred, 3),
            f"t={time.time()-t0:.0f}s", flush=True)
    axes[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig6_learned_vote_seed{SEED}.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for ci, cond in enumerate(R):
        ax.plot(ns, R[cond]["eps_star_obs"], "o-", color=f"C{ci}", label=f"{cond}: observed")
        ax.plot(ns, R[cond]["eps_star_pred"], "--", color=f"C{ci}", label=f"{cond}: predicted (pilot)")
    ax.set_xlabel("agents per subtask (n)"); ax.set_ylabel("break-even hand-off error eps*")
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig7_learned_eps_star_seed{SEED}.png"), dpi=150); plt.close()
    json.dump(R, open(os.path.join(OUT, f"learned_results_seed{SEED}.json"), "w"), indent=1)
    print("done", time.time() - t0)
