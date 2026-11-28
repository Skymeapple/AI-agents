"""Simulation studies S1-S3.  Run:  python3 sim.py   (~5-10 min CPU)"""
import json, itertools, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import *

OUT = "../results/"
rng0 = np.random.default_rng(2026)


def generate(K=6, R=30, M=100, per_item=3, tau=0.5, persistent=True,
             sd_item=0.6, sd_lenient=0.5, sd_noise=0.8, likert=True, rng=None,
             theta=None):
    """Score model y = theta_k + eta_i + b_r + g + eps.
    persistent=True : g = gamma_{r,k}  (rater x system taste, stable across items)
    persistent=False: g = fresh N(0, tau^2) per (item, rater)  (same marginal,
                      but not persistent -> indistinguishable per item)."""
    rng = rng or np.random.default_rng()
    theta = np.linspace(0, 1.0, K) if theta is None else theta
    gam = rng.normal(0, tau, (R, K))
    b = rng.normal(0, sd_lenient, R)
    rows = []
    for k in range(K):
        for m in range(M):
            item = k * M + m
            eta = rng.normal(0, sd_item)
            for r in rng.choice(R, per_item, replace=False):
                g = gam[r, k] if persistent else rng.normal(0, tau)
                y = 3 + theta[k] - theta.mean() + eta + b[r] + g + rng.normal(0, sd_noise)
                if likert:
                    y = float(np.clip(np.rint(y), 1, 5))
                rows.append((item, k, r, y))
    return pd.DataFrame(rows, columns=["item", "sys", "rater", "y"]), gam


def s1_invariance_and_power(reps=200, n_perm=199):
    """Agreement statistics cannot tell pluralism from noise; the persistence
    test can. Also type-I error at tau=0 (leniency-only null)."""
    K = 6
    pairs = list(itertools.combinations(range(K), 2))
    res = []
    taus = [0.0, 0.2, 0.35, 0.5]
    for tau in taus:
        for persistent in [True, False]:
            if tau == 0 and not persistent:
                continue
            for rep in range(reps if tau > 0 else 2 * reps):
                rng = np.random.default_rng(10_000 * int(tau * 100) + 2 * rep + persistent)
                df, _ = generate(tau=tau, persistent=persistent, rng=rng)
                a = krippendorff_alpha(df, "interval")
                obs, null, p = persistence_test(df, pairs, n_perm=n_perm, seed=rep)
                res.append(dict(tau=tau, persistent=persistent, alpha=a, Qex=obs, p=p))
        print("S1 tau", tau, "done", time.strftime("%X"), flush=True)
    r = pd.DataFrame(res)
    r.to_csv(OUT + "s1_raw.csv", index=False)
    summ = r.groupby(["tau", "persistent"]).agg(alpha_mean=("alpha", "mean"), alpha_sd=("alpha", "std"),
                                                 reject_rate=("p", lambda p: np.mean(p < 0.05)),
                                                 n=("p", "size")).reset_index()
    summ.to_csv(OUT + "s1_summary.csv", index=False)
    print(summ.to_string())
    return summ


def s2_calibration(reps=300):
    """DL tau estimate (noise-corrected) vs naive SD of raw rater contrasts."""
    res = []
    for tau in [0.0, 0.2, 0.4, 0.6]:
        for noise in [0.5, 1.0]:
            for rep in range(reps):
                rng = np.random.default_rng(777 + rep + 1000 * int(10 * tau) + int(noise * 7))
                df, gam = generate(K=2, R=40, M=150, tau=tau, sd_noise=noise, likert=False, rng=rng,
                                   theta=np.array([0.0, 0.4]))
                D, v, _, _ = rater_contrasts(df, 1, 0)
                mu, t2, Q, dfree = dersimonian_laird(D, v)
                true_tau_pair = np.sqrt(2) * tau
                res.append(dict(tau=tau, noise=noise, true=true_tau_pair, dl=np.sqrt(t2), naive=D.std(ddof=1),
                                ))
    r = pd.DataFrame(res)
    s = r.groupby(["tau", "noise"]).agg(true=("true", "mean"), dl_mean=("dl", "mean"), dl_sd=("dl", "std"),
                                         naive_mean=("naive", "mean")).reset_index()
    s.to_csv(OUT + "s2_summary.csv", index=False)
    print(s.to_string())
    return s


def s3_ci_vs_gamma(reps=150):
    """Sampling CIs shrink with more annotations; population-relativity
    (Gamma*) does not. Naive (noise-inflated) tau underestimates Gamma*."""
    m_true, tau_true = 0.3, 0.35   # pairwise contrast margin, per-system taste sd
    tau_pair = np.sqrt(2) * tau_true
    G_true = gamma_star_gaussian(m_true / tau_pair)
    res = []
    for M in [40, 80, 160, 320, 640]:
        for rep in range(reps):
            rng = np.random.default_rng(31 * M + rep)
            df, gam = generate(K=2, R=40, M=M, tau=tau_true, likert=False, rng=rng,
                               theta=np.array([0.0, m_true]), sd_noise=1.0)
            D, v, _, _ = rater_contrasts(df, 1, 0)
            mu, t2, Q, dfree = dersimonian_laird(D, v)
            # conventional leaderboard CI: item bootstrap of mean score difference
            means = df.groupby("item").agg(y=("y", "mean"), sys=("sys", "first"))
            y1 = means[means.sys == 1].y.values; y0 = means[means.sys == 0].y.values
            se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
            Gdl = gamma_star_gaussian(mu / np.sqrt(t2)) if t2 > 0 else np.inf
            Gnaive = gamma_star_gaussian(D.mean() / D.std(ddof=1))
            res.append(dict(M=M, diff=y1.mean() - y0.mean(), ci_half=1.96 * se,
                            G_dl=Gdl, G_naive=Gnaive))
    r = pd.DataFrame(res)
    r["G_dl_c"] = np.minimum(r.G_dl, 1e3)
    s = r.groupby("M").agg(diff=("diff", "mean"), ci_half=("ci_half", "mean"),
                           G_dl_median=("G_dl_c", "median"),
                           G_dl_q25=("G_dl_c", lambda x: np.quantile(x, .25)),
                           G_dl_q75=("G_dl_c", lambda x: np.quantile(x, .75)),
                           G_naive_median=("G_naive", "median")).reset_index()
    s["G_true"] = G_true
    s.to_csv(OUT + "s3_summary.csv", index=False)
    print(s.to_string())
    return s, G_true


def plot(s1, s2, s3, G_true):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    # S1
    a = ax[0]
    for pers, lab, c in [(True, "persistent taste (pluralism)", "#2c6fbb"),
                         (False, "transient noise, same marginal", "#d1495b")]:
        q = s1[(s1.persistent == pers) | (s1.tau == 0)]
        a.plot(q.tau, q.reject_rate, "o-", color=c, label=lab)
    a.axhline(0.05, ls=":", c="gray")
    a.set_xlabel(r"rater$\times$system taste sd $\tau$")
    a.set_ylabel("rejection rate of persistence test")
    a.set_title("S1: test separates pluralism from noise")
    a.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.25, 0.1))
    a2 = a.inset_axes([0.55, 0.45, 0.4, 0.27])
    for pers, c in [(True, "#2c6fbb"), (False, "#d1495b")]:
        q = s1[(s1.persistent == pers) | (s1.tau == 0)]
        a2.plot(q.tau, q.alpha_mean, "o-", color=c, ms=3)
    a2.set_title(r"Krippendorff $\alpha$", fontsize=7); a2.tick_params(labelsize=6)
    # S2
    a = ax[1]
    for noise, mk in [(0.5, "o"), (1.0, "s")]:
        q = s2[s2.noise == noise]
        a.plot(q.true, q.dl_mean, mk + "-", c="#2c6fbb", label=f"DL (noise sd {noise})")
        a.plot(q.true, q.naive_mean, mk + "--", c="#d1495b", label=f"naive SD (noise sd {noise})")
    lim = [0, max(s2.naive_mean.max(), s2.true.max()) * 1.05]
    a.plot(lim, lim, ":", c="gray")
    a.set_xlabel(r"true sd of rater contrasts $\tau_{kl}$"); a.set_ylabel("estimate")
    a.set_title("S2: noise-corrected heterogeneity"); a.legend(fontsize=7)
    # S3
    a = ax[2]
    a.plot(s3.M, s3.ci_half, "o-", c="#d1495b", label="leaderboard 95% CI half-width")
    a.set_xscale("log"); a.set_xlabel("items per system"); a.set_ylabel("CI half-width (score units)")
    b = a.twinx()
    b.plot(s3.M, s3.G_dl_median, "s-", c="#2c6fbb", label=r"$\hat\Gamma^*$ (DL), median")
    b.fill_between(s3.M, s3.G_dl_q25, s3.G_dl_q75, color="#2c6fbb", alpha=.15)
    b.plot(s3.M, s3.G_naive_median, "^--", c="#7a7a7a", label=r"$\Gamma^*$ from naive SD")
    b.axhline(G_true, ls=":", c="#2c6fbb"); b.set_ylabel(r"sensitivity $\Gamma^*$"); b.set_ylim(1, 5)
    a.set_title("S3: more data shrinks CIs, not relativity")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT + "fig_simulations.png", dpi=160)


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        s3 = pd.read_csv(OUT + "s3_summary.csv")
        plot(pd.read_csv(OUT + "s1_summary.csv"), pd.read_csv(OUT + "s2_summary.csv"), s3, s3.G_true.iloc[0])
        sys.exit()
    t = time.time()
    s1 = s1_invariance_and_power()
    s2 = s2_calibration()
    s3, G = s3_ci_vs_gamma()
    plot(s1, s2, s3, G)
    print("elapsed", time.time() - t)
