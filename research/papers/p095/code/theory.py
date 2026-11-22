"""Analytical results + Monte-Carlo checks for the selection model of automated research.

Model: N candidate ideas with true effects delta_i ~ F (mean mu, var tau^2); each is
evaluated with m runs, Y_i = delta_i + eps_i, eps_i ~ N(0, s^2/m); the candidate with
the largest Y is selected.  G = E[delta_sel] - mu (true gain), M = E[Y_sel] - mu
(measured gain), W = M - G (winner's curse).

Outputs (results/):
  theory_mc_check.csv, fig1_prop1_check.png
  trichotomy.csv, fig2_trichotomy.png
  speedup_table.csv
  optimal_m.csv, fig3_optimal_m.png
"""
import numpy as np, csv
from scipy import stats, integrate, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/user/AI-agents/research/papers/p095/results"
rng = np.random.default_rng(2026)


from functools import lru_cache


@lru_cache(maxsize=None)
def e_max(N):
    """E[max of N iid N(0,1)], exact by quadrature."""
    if N == 1:
        return 0.0
    f = lambda x: x * N * stats.norm.pdf(x) * stats.norm.cdf(x) ** (N - 1)
    v, _ = integrate.quad(f, -10, 12, limit=200)
    return v


def gaussian_GMW(N, m, tau=1.0, s=1.0):
    sy = np.sqrt(tau**2 + s**2 / m)
    e = e_max(N)
    G = tau**2 / sy * e
    M = sy * e
    return G, M, M - G


def mc_select(sample_delta, N, m, s, reps):
    """Monte Carlo replay of the select-max protocol. Returns G, M (relative to prior mean), SEs."""
    d = sample_delta((reps, N))
    y = d + rng.normal(0, s / np.sqrt(m), size=(reps, N))
    k = y.argmax(1)
    ds = d[np.arange(reps), k]; ys = y[np.arange(reps), k]
    return ds.mean(), ys.mean(), ds.std() / np.sqrt(reps), ys.std() / np.sqrt(reps)


# ---------------------------------------------------------------- Prop 1 check
def prop1_check():
    rows = []
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m, c in zip([1, 4, 16], ["C0", "C1", "C2"]):
        Gs, Ms, Ga, Ma = [], [], [], []
        for N in Ns:
            G, M, se_g, se_m = mc_select(lambda sh: rng.normal(0, 1, sh), N, m, 2.0, 40000)
            g, mm, w = gaussian_GMW(N, m, 1.0, 2.0)
            rows.append([m, N, G, se_g, g, M, se_m, mm])
            Gs.append(G); Ms.append(M); Ga.append(g); Ma.append(mm)
        ax[0].plot(Ns, Ga, "-", color=c, label=f"theory m={m}")
        ax[0].plot(Ns, Gs, "o", color=c, ms=4)
        ax[1].plot(Ns, Ma, "-", color=c, label=f"theory m={m}")
        ax[1].plot(Ns, Ms, "o", color=c, ms=4)
    for a, t in zip(ax, ["true gain G (MC dots)", "measured gain M (MC dots)"]):
        a.set_xscale("log"); a.set_xlabel("candidates evaluated N"); a.set_title(t); a.legend(fontsize=8)
    ax[0].set_ylabel("gain over random idea (units of tau)")
    fig.suptitle("Prop. 1: Gaussian prior tau=1, per-run noise s=2")
    fig.tight_layout(); fig.savefig(f"{R}/fig1_prop1_check.png", dpi=150); plt.close(fig)
    with open(f"{R}/theory_mc_check.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["m", "N", "G_mc", "G_se", "G_theory", "M_mc", "M_se", "M_theory"]); w.writerows(rows)
    err = max(abs(r[2] - r[4]) / max(r[3], 1e-9) for r in rows)
    print(f"Prop1: max |G_mc-G_th|/SE = {err:.2f}")


# ---------------------------------------------------------------- trichotomy
def trichotomy():
    """True and measured gain for priors in the three extreme-value domains, all with
    unit variance and the same Gaussian evaluation noise (s=1, m=1)."""
    a2 = 3.0  # Pareto tail index (finite variance)
    xm = np.sqrt(a2 - 2) * (a2 - 1) / np.sqrt(a2)  # gives var 1
    priors = {
        "Frechet: Pareto alpha=3": lambda sh: xm * (rng.pareto(a2, sh) + 1) - xm * a2 / (a2 - 1),
        "Gumbel: Gaussian": lambda sh: rng.normal(0, 1, sh),
        "Weibull: uniform (bounded)": lambda sh: rng.uniform(-np.sqrt(3), np.sqrt(3), sh),
        "Weibull: neg. skew Beta(5,1.5)": None,
    }
    b = stats.beta(5, 1.5); bm, bs = b.mean(), b.std()
    priors["Weibull: neg. skew Beta(5,1.5)"] = lambda sh: (b.rvs(size=sh, random_state=rng) - bm) / bs
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for (name, smp), c in zip(priors.items(), ["C3", "C0", "C2", "C4"]):
        Gs, fr = [], []
        for N in Ns:
            reps = max(400, int(4e6 / N))
            G, M, sg, sm = mc_select(smp, N, 1, 1.0, reps)
            rows.append([name, N, G, sg, M, sm, (M - G) / M if M > 0 else np.nan])
            Gs.append(G); fr.append((M - G) / M if N > 1 else np.nan)
        ax[0].plot(Ns, Gs, "o-", color=c, ms=3, label=name)
        ax[1].plot(Ns[1:], fr[1:], "o-", color=c, ms=3, label=name)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("N"); ax[0].set_ylabel("true gain G (units of prior sd)")
    ax[0].set_title("true gain vs breadth (noise s=1, m=1)"); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("N"); ax[1].set_ylabel("illusory share W/M")
    ax[1].set_title("share of measured gain that is winner's curse")
    fig.tight_layout(); fig.savefig(f"{R}/fig2_trichotomy.png", dpi=150); plt.close(fig)
    with open(f"{R}/trichotomy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prior", "N", "G", "G_se", "M", "M_se", "illusory_share"]); w.writerows(rows)
    print("trichotomy done")


# ---------------------------------------------------------------- speedup bound
def best_uniform(B, kappa, nmax=None):
    """Best uniform allocation under Gaussian prior (tau=1, s^2=1/kappa): choose N (m=B/N)."""
    best = (-1, None)
    for N in range(1, (nmax or B) + 1):
        m = B / N
        if m < 1:
            break
        g = gaussian_GMW(N, m, 1.0, np.sqrt(1 / kappa))[0]
        if g > best[0]:
            best = (g, N)
    return best


def speedup_table():
    """Per-round gain of an unconstrained automated proposer (any N <= B) relative to a
    labour-capped proposer (N <= N_h), both spending the same evaluation compute B,
    Gaussian prior.  Also the policy-free ceiling e_B (noiseless oracle on B ideas)."""
    rows = []
    for kappa in [0.25, 1.0, 4.0]:
        for B in [64, 256, 1024, 4096]:
            gA, NA = best_uniform(B, kappa)
            ceil = e_max(B)
            for Nh in [4, 16]:
                gH, NH = best_uniform(B, kappa, nmax=Nh)
                rows.append([kappa, B, Nh, NH, gH, NA, gA, gA / gH, ceil, ceil / gH])
    with open(f"{R}/speedup_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kappa_signal_to_noise_per_run", "B_runs", "N_cap_human", "N_human_opt", "G_human",
                    "N_auto_opt", "G_auto", "speedup_auto_over_human", "ceiling_eB", "ceiling_speedup"])
        w.writerows(rows)
    print("speedup table done")


# ---------------------------------------------------------------- optimal m
def optimal_m():
    rows = []
    fig, ax = plt.subplots(figsize=(5, 4))
    Bs = [8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384]
    for kappa, c in zip([0.1, 0.25, 1.0, 4.0], ["C0", "C1", "C2", "C3"]):
        ms = []
        for B in Bs:
            # continuous relaxation: maximise rho(m) * sqrt(2 ln(B/m)) approx; use exact e_max on grid of N
            g, N = best_uniform(B, kappa) if B <= 4096 else (None, None)
            if N is None:
                f = lambda lm: -(1 / np.sqrt(1 + 1 / (kappa * np.exp(lm)))) * np.sqrt(2 * np.log(B / np.exp(lm)))
                r = optimize.minimize_scalar(f, bounds=(0, np.log(B) - 0.1), method="bounded")
                m = np.exp(r.x); N = B / m
            m = B / N
            rows.append([kappa, B, N, m, kappa * m])
            ms.append(m)
        ax.plot(Bs, ms, "o-", color=c, ms=3, label=f"kappa={kappa}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("evaluation budget B (runs)")
    ax.set_ylabel("optimal runs per candidate m*"); ax.legend(fontsize=8)
    ax.set_title("Gaussian prior: optimal replication")
    fig.tight_layout(); fig.savefig(f"{R}/fig3_optimal_m.png", dpi=150); plt.close(fig)
    with open(f"{R}/optimal_m.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kappa", "B", "N_opt", "m_opt", "per_candidate_SNR_kappa_m"]); w.writerows(rows)
    print("optimal m done")


if __name__ == "__main__":
    prop1_check()
    trichotomy()
    speedup_table()
    optimal_m()
