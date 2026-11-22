"""Synthetic coverage study: certified linear estimator vs parametric extrapolation.
Outputs results/synthetic_*.csv and fig_synthetic.png.  ~5-8 CPU minutes."""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.integrate import quad
from scipy.special import expit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ttc import *

RES = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(12345)


# ---- mixing laws: sampler + exact expectation of g (numerical quadrature on components)
class Mix:
    def __init__(self, name, comps):
        self.name, self.comps = name, comps  # list of (weight, kind, params)

    def sample(self, M):
        w = np.array([c[0] for c in self.comps])
        z = rng.choice(len(w), size=M, p=w)
        p = np.empty(M)
        for j, (wt, kind, par) in enumerate(self.comps):
            idx = z == j
            if kind == "point":
                p[idx] = par
            elif kind == "beta":
                p[idx] = rng.beta(*par, size=idx.sum())
            elif kind == "logitnormal":
                p[idx] = expit(rng.normal(par[0], par[1], size=idx.sum()))
        return p

    def expect(self, g):
        tot = 0.0
        for wt, kind, par in self.comps:
            if kind == "point":
                tot += wt * float(g(np.array([par]))[0])
            else:
                # Monte Carlo with 2e6 draws on a fixed RNG for the continuous parts
                r = np.random.default_rng(7)
                if kind == "beta":
                    s = r.beta(*par, size=2_000_000)
                else:
                    s = expit(r.normal(par[0], par[1], size=2_000_000))
                tot += wt * float(np.mean(g(s)))
        return tot

    def count_law(self, n):
        # exact-enough count law via the same MC draws
        law = np.zeros(n + 1)
        for wt, kind, par in self.comps:
            if kind == "point":
                law += wt * binom.pmf(np.arange(n + 1), n, par)
            else:
                r = np.random.default_rng(7)
                s = r.beta(*par, size=400_000) if kind == "beta" else expit(r.normal(par[0], par[1], size=400_000))
                law += wt * binom.pmf(np.arange(n + 1)[None, :], n, s[:200_000, None]).mean(0)
        return law / law.sum()


LAWS = [
    Mix("Beta(0.3,3)", [(1.0, "beta", (0.3, 3.0))]),
    Mix("0.3*delta0+0.7*Beta(.5,2)", [(0.3, "point", 0.0), (0.7, "beta", (0.5, 2.0))]),
    Mix("0.3*delta(1e-3)+0.7*Beta(.5,2)", [(0.3, "point", 1e-3), (0.7, "beta", (0.5, 2.0))]),
    Mix("LogitNormal(-3,2.5)", [(1.0, "logitnormal", (-3.0, 2.5))]),
]


def main(reps=100, boot=30):
    rows = []
    for n in [16, 64]:
        targets = {f"pass@{k}": (g_passk(k), k) for k in sorted({2 * n, 8 * n, n * n // 4, n * n, 4 * n * n})}
        targets["ceiling(t=.9,q=.01)"] = (g_verifier_ceiling(0.9, 0.01), None)
        for M in ([500, 5000] if n == 16 else [5000]):
            certs = {name: LinearCertificate(g, n, M, alpha=0.05) for name, (g, _) in targets.items()}
            for law in LAWS:
                truth = {name: law.expect(g) for name, (g, _) in targets.items()}
                cl = law.count_law(n)
                ident = {name: identified_interval_robust(g, n, cl, m=3000, slack=1e-5) for name, (g, _) in targets.items()}
                t0 = time.time()
                for r in range(reps):
                    p = law.sample(M)
                    X = rng.binomial(n, p)
                    do_boot = (r < 50) and n == 16
                    do_box = (r < 100) if n == 16 else (r < 25)
                    a, b = betabinom_fit(X, n)
                    if do_boot:
                        bs = []
                        for _ in range(boot):
                            pb = rng.beta(a, b, size=M); Xb = rng.binomial(n, pb)
                            bs.append(betabinom_fit(Xb, n, init=(a, b)))
                    for name, (g, k) in targets.items():
                        c = certs[name]
                        th, lo, hi, half = c.estimate(X)
                        sd = np.std(c.h[X]) / np.sqrt(M)
                        row = dict(n=n, M=M, law=law.name, target=name, rep=r, truth=truth[name],
                                   cert_est=th, cert_lo=lo, cert_hi=hi, cert_half=half,
                                   cert_bias=c.bias + c.grid_slack, clt_half=c.bias + c.grid_slack + 1.96 * sd,
                                   id_lo=ident[name][0], id_hi=ident[name][1])
                        if do_box:
                            row["box_lo"], row["box_hi"] = box_interval(g, n, X, alpha=0.05, m=800)
                        if k is not None:
                            row["bb_est"] = float(betabinom_passk(a, b, k))
                            row["pl_est"] = float(powerlaw_passk(X, n, [k])[0])
                            if do_boot:
                                v = np.array([betabinom_passk(aa, bb, k) for aa, bb in bs])
                                row["bb_lo"], row["bb_hi"] = np.quantile(v, [0.025, 0.975])
                        rows.append(row)
                print(n, M, law.name, f"{time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "synthetic_raw.csv.gz"), index=False)
    summarize(df)


def summarize(df):
    df = df.copy()
    df["cert_cover"] = (df.cert_lo <= df.truth) & (df.truth <= df.cert_hi)
    df["box_cover"] = ((df.box_lo - 1e-9 <= df.truth) & (df.truth <= df.box_hi + 1e-9)).astype(float)
    df.loc[df.box_lo.isna(), "box_cover"] = np.nan
    df["box_half"] = (df.box_hi - df.box_lo) / 2
    df["bb_err"] = df.bb_est - df.truth
    df["pl_err"] = df.pl_est - df.truth
    df["bb_cover"] = ((df.bb_lo <= df.truth) & (df.truth <= df.bb_hi)).astype(float)
    df.loc[df.bb_lo.isna(), "bb_cover"] = np.nan
    g = df.groupby(["n", "M", "law", "target"], sort=False)
    s = g.agg(truth=("truth", "first"), id_lo=("id_lo", "first"), id_hi=("id_hi", "first"),
              cert_cover=("cert_cover", "mean"), cert_half=("cert_half", "mean"), cert_bias=("cert_bias", "first"),
              cert_rmse=("cert_est", lambda x: np.nan), bb_bias=("bb_err", "mean"), bb_absmax=("bb_err", lambda x: np.abs(x).max()),
              bb_cover=("bb_cover", "mean"), pl_bias=("pl_err", "mean"),
              box_cover=("box_cover", "mean"), box_half=("box_half", "mean"))
    s["cert_rmse"] = g.apply(lambda d: np.sqrt(np.mean((d.cert_est - d.truth) ** 2)))
    s["id_width"] = s.id_hi - s.id_lo
    s = s.reset_index()
    s.to_csv(os.path.join(RES, "synthetic_summary.csv"), index=False, float_format="%.4f")
    print(s.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
