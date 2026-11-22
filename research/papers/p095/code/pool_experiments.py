"""Replay automated-research protocols on the real candidate pools built by build_pool.py.

For each dataset:
  1. estimate prior spread tau, per-run evaluation noise s, and the upper-tail shape
     (generalised Pareto shape xi of the true-value distribution);
  2. replay "propose N, evaluate each with m runs, keep the argmax" for many (B, m) and
     compare realised true / measured gains with the Gaussian closed form (Prop. 1);
  3. audit gain estimators: naive measured gain vs reliability-corrected gain;
  4. multi-round research with a stale incumbent score vs re-measured incumbent.

Every number is written to results/*.csv; figures to results/fig4-6*.png.
"""
import numpy as np, csv, json, sys
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/user/AI-agents/research/papers/p095/code")
from theory import e_max, gaussian_GMW

R = "/home/user/AI-agents/research/papers/p095/results"
NV = 500            # validation examples per evaluation run
EVAL_SEEDS = 4      # seeds 0..3 are used for evaluation, 4..7 define the true value
rng = np.random.default_rng(99)


class Pool:
    def __init__(self, name, subset="all"):
        z = np.load(f"{R}/pool_{name}.npz")
        val, test = z["val_correct"], z["test_acc"]
        true = test[:, EVAL_SEEDS:].mean(1)
        if subset == "top25":
            # a 'competent proposer': its ideas are drawn from the best quarter of the space
            keep = true >= np.quantile(true, 0.75)
            val, test, true = val[keep], test[keep], true[keep]
        self.val, self.test, self.true = val, test, true   # (C,S,V) bool, (C,S), (C,)
        self.C, self.S, self.V = self.val.shape
        # per-(config, seed) cumulative sums allow fast random-subset accuracy via sampling indices
        self.valf = self.val.astype(np.float32)

    def draw(self, cfg, seeds):
        """One evaluation run per (cfg[i], seeds[i]): accuracy on a fresh random NV-subset."""
        idx = rng.integers(0, self.V, size=(len(cfg), NV))   # sampling with replacement from val pool
        return self.valf[cfg[:, None], seeds[:, None], idx].mean(1)


def noise_and_prior(P):
    # per-run noise: variance of draws around each config's mean over eval seeds & subsets
    reps = 16
    cfg = np.repeat(np.arange(P.C), reps)
    seeds = rng.integers(0, EVAL_SEEDS, size=cfg.size)
    y = P.draw(cfg, seeds).reshape(P.C, reps)
    s2 = y.var(1, ddof=1).mean()
    # signal: true values. remove residual noise of the 4-seed test mean (small)
    test_noise = P.test[:, EVAL_SEEDS:].var(1, ddof=1).mean() / (P.S - EVAL_SEEDS)
    tau2 = P.true.var(ddof=1) - test_noise
    # validation-vs-test agreement of config means
    corr = np.corrcoef(y.mean(1), P.true)[0, 1]
    return np.sqrt(s2), np.sqrt(tau2), corr, test_noise


def gpd_tail(x, q=0.8, nboot=500):
    u = np.quantile(x, q)
    exc = x[x > u] - u
    xi, loc, sc = stats.genpareto.fit(exc, floc=0)
    bs = []
    for _ in range(nboot):
        e = rng.choice(exc, exc.size, replace=True)
        bs.append(stats.genpareto.fit(e, floc=0)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return xi, lo, hi, exc.size, u, x.max()


def replay(P, s, tau, name):
    rows, est_rows = [], []
    mu = P.true.mean()
    for m in [1, 2, 4]:
        for B in [4, 8, 16, 32, 64, 128, 256, 400]:
            N = B // m
            if N < 1 or N > P.C // 2:   # cap to limit finite-pool truncation
                continue
            reps = 3000
            G, M, Wsh, naive_err, corr_err, holdout_err = [], [], [], [], [], []
            for _ in range(reps):
                cfg = rng.choice(P.C, N, replace=False)
                seeds = np.stack([rng.permutation(EVAL_SEEDS)[:m] for _ in range(N)])
                y = P.draw(np.repeat(cfg, m), seeds.reshape(-1)).reshape(N, m)
                ybar = y.mean(1)
                k = ybar.argmax()
                g_true = P.true[cfg[k]] - P.true[cfg].mean()   # gain over average proposed idea
                G.append(P.true[cfg[k]] - mu)
                M.append(ybar[k] - mu)
                if m >= 2 and N >= 4:
                    naive = ybar[k] - ybar.mean()
                    s2_hat = y.var(1, ddof=1).mean() / m
                    vy = ybar.var(ddof=1)
                    rel = np.clip(1 - s2_hat / vy, 0, 1)
                    corrected = rel * naive
                    # gold standard with equal extra cost: m fresh runs of the winner and of a random idea
                    naive_err.append(naive - g_true)
                    corr_err.append(corrected - g_true)
            G, M = np.array(G), np.array(M)
            gth, mth, _ = gaussian_GMW(N, m, tau, s)
            rows.append([name, m, B, N, G.mean(), G.std() / np.sqrt(reps), gth,
                         M.mean(), M.std() / np.sqrt(reps), mth, (M.mean() - G.mean()) / M.mean()])
            if naive_err:
                ne, ce = np.array(naive_err), np.array(corr_err)
                est_rows.append([name, m, B, N, ne.mean(), np.sqrt((ne**2).mean()),
                                 ce.mean(), np.sqrt((ce**2).mean())])
    return rows, est_rows


def rounds(P, s, tau, name, R_=40, k=8, reps=1000, gaussian=False):
    """Iterated research. Each round: k new ideas, 1 run each; best challenger vs incumbent.
    policy 'stale': incumbent keeps the score it had when accepted.
    policy 'remeasure': incumbent gets one fresh run each round; compare fresh numbers.
    Returns mean trajectories of true value and reported score of the incumbent."""
    out = {}
    for pol in ["stale", "remeasure"]:
        T = np.zeros((reps, R_ + 1)); Rep = np.zeros((reps, R_ + 1)); acc = np.zeros(reps)
        for r in range(reps):
            if gaussian:
                inc_true = 0.0; inc_rep = inc_true + rng.normal(0, s)
            else:
                c0 = rng.integers(P.C)
                inc_true = P.true[c0]; inc_cfg = c0
                inc_rep = P.draw(np.array([c0]), rng.integers(0, EVAL_SEEDS, 1))[0]
            T[r, 0], Rep[r, 0] = inc_true, inc_rep
            for t in range(1, R_ + 1):
                if gaussian:
                    d = rng.normal(0, tau, k); y = d + rng.normal(0, s, k)
                    j = y.argmax(); ch_true, ch_y = d[j], y[j]
                    inc_now = inc_true + rng.normal(0, s)
                else:
                    cfg = rng.choice(P.C, k, replace=False)
                    y = P.draw(cfg, rng.integers(0, EVAL_SEEDS, k))
                    j = y.argmax(); ch_true, ch_y = P.true[cfg[j]], y[j]
                    inc_now = P.draw(np.array([inc_cfg]), rng.integers(0, EVAL_SEEDS, 1))[0]
                comp = inc_rep if pol == "stale" else inc_now
                if ch_y > comp:
                    inc_true, inc_rep = ch_true, ch_y
                    if not gaussian:
                        inc_cfg = cfg[j]
                    acc[r] += 1
                elif pol == "remeasure":
                    inc_rep = inc_now
                T[r, t], Rep[r, t] = inc_true, inc_rep
        out[pol] = (T.mean(0), Rep.mean(0), acc.mean(), T[:, -1].std() / np.sqrt(reps))
    return out


def main():
    summary = {}
    all_rows, all_est = [], []
    fig4, ax4 = plt.subplots(2, 2, figsize=(11, 8)); ax4 = ax4.ravel()
    fig6, ax6 = plt.subplots(1, 5, figsize=(22, 4))
    for di, (dname, sub) in enumerate([("letter", "all"), ("letter", "top25"), ("magic", "all"), ("magic", "top25")]):
        P = Pool(dname, sub); name = f"{dname}-{sub}"
        s, tau, corr, tn = noise_and_prior(P)
        xi, lo, hi, nexc, u, xmax = gpd_tail(P.true)
        skew = stats.skew(P.true)
        summary[name] = dict(configs=int(P.C), seeds=int(P.S), per_run_noise_s=float(s), prior_sd_tau=float(tau),
                             kappa=float(tau**2 / s**2), val_test_corr_16draw=float(corr),
                             true_mean=float(P.true.mean()), true_max=float(xmax),
                             true_median=float(np.median(P.true)), skewness=float(skew),
                             gpd_xi=float(xi), gpd_xi_ci=[float(lo), float(hi)], gpd_n_exceed=int(nexc),
                             gpd_threshold=float(u),
                             implied_endpoint=float(u - stats.genpareto.fit(P.true[P.true > u] - u, floc=0)[2] / xi) if xi < 0 else None)
        print(name, summary[name])
        rows, est = replay(P, s, tau, name)
        all_rows += rows; all_est += est
        a = ax4[di]
        for m, c in zip([1, 2, 4], ["C0", "C1", "C2"]):
            rr = [x for x in rows if x[1] == m]
            Ns = [x[3] for x in rr]
            a.errorbar(Ns, [x[4] for x in rr], yerr=[2 * x[5] for x in rr], fmt="o-", color=c, ms=3, label=f"true gain, m={m}")
            a.plot(Ns, [x[6] for x in rr], ":", color=c, label=f"Gaussian theory, m={m}")
            a.plot(Ns, [x[7] for x in rr], "s--", color=c, ms=3, alpha=0.5, label=f"measured gain, m={m}")
        a.axhline(P.true.max() - P.true.mean(), color="k", lw=0.8, ls="-.", label="pool best")
        a.set_xscale("log"); a.set_xlabel("candidates evaluated N"); a.set_ylabel("accuracy gain over mean idea")
        a.set_title(f"{name}: replay on real pool"); a.legend(fontsize=6)
        # rounds
        rd = rounds(P, s, tau, name)
        summary[name]["rounds"] = {pol: dict(final_true=float(v[0][-1]), final_reported=float(v[1][-1]),
                                             gap=float(v[1][-1] - v[0][-1]), accepts=float(v[2]), se_true=float(v[3]))
                                   for pol, v in rd.items()}
        for pol, c in zip(["stale", "remeasure"], ["C3", "C0"]):
            ax6[di].plot(rd[pol][0], "-", color=c, label=f"{pol}: true")
            ax6[di].plot(rd[pol][1], "--", color=c, label=f"{pol}: reported")
        ax6[di].set_title(f"{name}: 40 rounds, 8 ideas/round"); ax6[di].set_xlabel("round"); ax6[di].set_ylabel("accuracy"); ax6[di].legend(fontsize=7)
    # gaussian rounds for reference (tau=1, s=2)
    rd = rounds(None, 2.0, 1.0, "gauss", gaussian=True)
    summary["gaussian_rounds_tau1_s2"] = {pol: dict(final_true=float(v[0][-1]), final_reported=float(v[1][-1]),
                                                     gap=float(v[1][-1] - v[0][-1]), accepts=float(v[2]), se_true=float(v[3]))
                                          for pol, v in rd.items()}
    for pol, c in zip(["stale", "remeasure"], ["C3", "C0"]):
        ax6[4].plot(rd[pol][0], "-", color=c, label=f"{pol}: true")
        ax6[4].plot(rd[pol][1], "--", color=c, label=f"{pol}: reported")
    ax6[4].set_title("Gaussian model tau=1, s=2"); ax6[4].set_xlabel("round"); ax6[4].legend(fontsize=7)
    fig4.tight_layout(); fig4.savefig(f"{R}/fig4_pool_replay.png", dpi=150)
    fig6.tight_layout(); fig6.savefig(f"{R}/fig6_rounds.png", dpi=150)
    with open(f"{R}/pool_replay.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "m", "B", "N", "G_true", "G_se", "G_gauss_theory", "M_measured", "M_se", "M_gauss_theory", "illusory_share"])
        w.writerows(all_rows)
    with open(f"{R}/estimator_audit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "m", "B", "N", "naive_bias", "naive_rmse", "corrected_bias", "corrected_rmse"])
        w.writerows(all_est)
    # tail figure
    fig5, ax5 = plt.subplots(1, 2, figsize=(10, 3.6))
    for di, name in enumerate(["letter", "magic"]):
        P = Pool(name); name = f"{name}-all"
        ax5[di].hist(P.true, bins=40, color="C0")
        ax5[di].set_title(f"{name}: true accuracy of 400 proposed ideas\nGPD tail shape xi={summary[name]['gpd_xi']:.2f} "
                          f"[{summary[name]['gpd_xi_ci'][0]:.2f},{summary[name]['gpd_xi_ci'][1]:.2f}]", fontsize=9)
        ax5[di].set_xlabel("test accuracy (mean of 4 held-out seeds)")
    fig5.tight_layout(); fig5.savefig(f"{R}/fig5_prior_tails.png", dpi=150)
    with open(f"{R}/pool_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
