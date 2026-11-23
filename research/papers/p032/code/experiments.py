"""Run all experiments for the paper. Usage:  python3 experiments.py
Outputs go to ../results/ (CSV/JSON tables and PNG figures)."""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nso import (Game, Design, detect_prob, exact_dp, strict_success,
                 monte_carlo, leading_order_target_failure)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "lines.linewidth": 2})

# Stylized regimes (NOT fitted to Engels et al.; chosen to span their three
# qualitative cases). Elo units; beta < 0 means the guard role is easier.
REGIMES = {
    "symmetric": Game(mG=1.0, mH=1.0, beta=-300.0),
    "guard-steeper": Game(mG=1.0, mH=0.4, beta=-100.0),
    "houdini-steeper": Game(mG=0.4, mH=1.0, beta=-400.0),
}
log = {}


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)


# ---------------------------------------------------------------- E1 validation
def e1_validation():
    rows, rng = [], np.random.default_rng(1)
    for t in range(24):
        gname = list(REGIMES)[t % 3]
        g = REGIMES[gname]
        d = Design(n=int(rng.integers(3, 13)), k=int(rng.integers(1, 5)),
                   anchor=bool(rng.integers(0, 2)),
                   rho=float(rng.choice([0, .3, .6, .9])),
                   gamma=float(rng.choice([0, .25, .5])))
        Delta = float(rng.choice([400, 800, 1600]))
        ex = exact_dp(g, d, Delta)
        st = strict_success(g, d, Delta)
        mc = monte_carlo(g, d, Delta, n_runs=200_000, seed=t)
        rows.append(dict(regime=gname, n=d.n, k=d.k, anchor=d.anchor, rho=d.rho,
                         gamma=d.gamma, Delta=Delta, dp_target=ex,
                         mc_target=mc["target"], mc_se=mc["target_se"],
                         z=(ex - mc["target"]) / max(mc["target_se"], 1e-12),
                         strict=st, mc_strict=mc["strict"]))
    save("e1_validation.json", rows)
    z = np.array([r["z"] for r in rows])
    log["e1"] = dict(n_configs=len(rows), max_abs_z=float(np.abs(z).max()),
                     max_abs_err_target=float(max(abs(r["dp_target"] - r["mc_target"]) for r in rows)),
                     max_abs_err_strict=float(max(abs(r["strict"] - r["mc_strict"]) for r in rows)))


# ------------------------------------------------- E2 chain baseline (Engels-like)
def e2_chain():
    res = {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (gname, g) in zip(axes, REGIMES.items()):
        res[gname] = {}
        for ci, Delta in enumerate([400, 800, 1600]):
            ns = np.arange(1, 41)
            s = [strict_success(g, Design(n=int(n), k=1), Delta) for n in ns]
            res[gname][Delta] = dict(n_opt=int(ns[int(np.argmax(s))]),
                                     p_opt=float(np.max(s)), p_n1=float(s[0]))
            ax.plot(ns, s, color=C[ci], label=f"Δ={Delta}")
        ax.set_title(gname)
        ax.set_xlabel("oversight steps n (chain, k=1)")
    axes[0].set_ylabel("P(success)  [chain: target = strict]")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e2_chain.png"), dpi=160)
    plt.close(fig)
    save("e2_chain.json", res)
    log["e2"] = res


# ------------------------------------- E3 error accumulation with fixed step size
def e3_accumulation():
    g = REGIMES["symmetric"]
    delta = 100.0  # fixed general-Elo gap per step; chain is stationary
    ns = np.arange(1, 61)
    out = {}
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for k in [1, 2, 3, 4]:
        fail = []
        for n in ns:
            d = Design(n=int(n), k=k)
            fail.append(1 - exact_dp(g, d, delta * n))
        fail = np.array(fail)
        q = np.array([1 - g.p(0, m * delta) for m in range(1, k + 1)])
        lo = np.array([leading_order_target_failure(q, int(n)) for n in ns])
        slope_exact = float(fail[-1] - fail[-2])
        r_k = float(np.prod(q ** np.arange(1, k + 1)))
        out[k] = dict(q=q.tolist(), fail_n10=float(fail[9]), fail_n30=float(fail[29]),
                      fail_n60=float(fail[-1]), slope_exact_n60=slope_exact,
                      r_k_pred=r_k, prod_q=float(np.prod(q)),
                      leading_order_n60=float(lo[-1]))
        ax.semilogy(ns, fail, color=C[k - 1], label=f"k={k} (exact DP)")
        valid = lo <= 0.5   # leading-order formula only meaningful where small
        if k > 1 and valid.any():
            ax.semilogy(ns[valid], lo[valid], color=C[k - 1], ls=":", lw=1.5)
    ax.set_xlabel("number of levels n (fixed step δ=100 Elo)")
    ax.set_ylabel("P(target compromised)")
    ax.set_title("Full collusion (γ=0), independent overseers (ρ=0)\n"
                 "dotted: leading-order heuristic, drawn only where ≤ 0.5", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e3_accumulation.png"), dpi=160)
    plt.close(fig)
    save("e3_accumulation.json", out)
    log["e3"] = out


# ----------------------------------------------------- E4 correlation sweep (rho)
def e4_rho():
    g = REGIMES["symmetric"]
    Delta, n = 1600, 16
    rhos = np.linspace(0, 1, 21)
    out = {"Delta": Delta, "n": n, "rows": []}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for k in [1, 2, 3, 4]:
        T, S = [], []
        for r in rhos:
            d = Design(n=n, k=k, rho=float(r))
            T.append(exact_dp(g, d, Delta)); S.append(strict_success(g, d, Delta))
        out["rows"].append(dict(k=k, rho=rhos.tolist(), target=T, strict=S))
        axes[0].plot(rhos, S, color=C[k - 1], label=f"k={k}")
        axes[1].plot(rhos, T, color=C[k - 1], label=f"k={k}")
    axes[0].set_title("strict success (no level escapes)")
    axes[1].set_title("target success (final level trusted), γ=0")
    for a in axes:
        a.set_xlabel("within-level correlation ρ of overseer errors")
    axes[0].set_ylabel("probability"); axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e4_rho.png"), dpi=160)
    plt.close(fig)
    # per-level detection with two overseers at distances 1 and 2
    step = Delta / n
    p1, p2 = g.p(0, step), g.p(0, 2 * step)
    out["two_overseers"] = dict(p1=p1, p2=p2, rows=[
        dict(rho=float(r), detect=detect_prob([p1, p2], float(r))) for r in rhos])
    # cross-level correlation robustness (Monte Carlo; DP assumes independence)
    cl = []
    for crho in [0.0, 0.3, 0.6]:
        for k in [1, 2, 3]:
            mc = monte_carlo(g, Design(n=n, k=k, rho=0.3), Delta, n_runs=200_000,
                             seed=11, cross_level_rho=crho)
            cl.append(dict(cross_rho=crho, k=k, target=float(mc["target"]),
                           strict=float(mc["strict"]), se=float(mc["target_se"])))
    out["cross_level"] = cl
    save("e4_rho.json", out)
    log["e4"] = dict(two_overseers=out["two_overseers"],
                     endpoints={r["k"]: dict(target_rho0=r["target"][0], target_rho1=r["target"][-1],
                                             strict_rho0=r["strict"][0], strict_rho1=r["strict"][-1],
                                             target_rho05=r["target"][10], strict_rho05=r["strict"][10])
                                for r in out["rows"]},
                     cross_level=cl)


# ------------------------------------------------------ E5 collusion sweep (gamma)
def e5_gamma():
    g = REGIMES["symmetric"]
    Delta, n = 1600, 16
    gammas = np.linspace(0, 1, 21)
    out = {"rows": []}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for k in [1, 2, 3, 4]:
        T = [exact_dp(g, Design(n=n, k=k, gamma=float(gm), rho=0.3), Delta) for gm in gammas]
        out["rows"].append(dict(k=k, gamma=gammas.tolist(), target=T))
        ax.plot(gammas, T, color=C[k - 1], label=f"k={k}")
    ax.set_xlabel("residual detection of a compromised overseer γ")
    ax.set_ylabel("target success")
    ax.set_title(f"Δ={Delta}, n={n}, ρ=0.3", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_e5_gamma.png"), dpi=160)
    plt.close(fig)
    # stationary two-state chain check: P(C) = q/(q+gamma(1-q)) for k=1
    q = 1 - g.p(0, 100.0)
    chk = []
    for gm in [0.1, 0.3, 0.6]:
        pc_dp = 1 - exact_dp(g, Design(n=200, k=1, gamma=gm), 200 * 100.0)
        chk.append(dict(gamma=gm, dp=pc_dp, formula=q / (q + gm * (1 - q))))
    out["k1_stationary_check"] = chk
    save("e5_gamma.json", out)
    log["e5"] = dict(k1_stationary_check=chk,
                     endpoints={r["k"]: dict(g0=r["target"][0], g02=r["target"][4],
                                             g05=r["target"][10], g1=r["target"][-1])
                                for r in out["rows"]})


# ----------------------------------------- E6 budget-constrained topology design
def e6_budget():
    out = []
    for gname, g in REGIMES.items():
        for Delta in [400, 800, 1600]:
            for rho in [0.0, 0.5, 0.9]:
                for B in [8, 16, 32]:
                    best = {"target": None, "strict": None}
                    chain = {"target": None, "strict": None}
                    for n in range(1, B + 1):
                        for k in range(1, 7):
                            for anchor in [False, True]:
                                if anchor and k >= n:
                                    continue
                                d = Design(n=n, k=k, anchor=anchor, rho=rho, gamma=0.0)
                                if d.cost() > B:
                                    continue
                                t = exact_dp(g, d, Delta)
                                s = strict_success(g, d, Delta)
                                for key, val in (("target", t), ("strict", s)):
                                    cand = dict(n=n, k=k, anchor=anchor, cost=d.cost(), value=val)
                                    if best[key] is None or val > best[key]["value"] + 1e-12:
                                        best[key] = cand
                                    if k == 1 and not anchor and (chain[key] is None or val > chain[key]["value"] + 1e-12):
                                        chain[key] = cand
                    out.append(dict(regime=gname, Delta=Delta, rho=rho, budget=B,
                                    best_target=best["target"], chain_target=chain["target"],
                                    best_strict=best["strict"], chain_strict=chain["strict"]))
    save("e6_budget.json", out)
    # compact table
    lines = ["regime,Delta,rho,B,chain_target(n),best_target(n;k;anchor),chain_strict(n),best_strict(n;k;anchor)"]
    for r in out:
        bt, ct, bs, cs = r["best_target"], r["chain_target"], r["best_strict"], r["chain_strict"]
        lines.append(f'{r["regime"]},{r["Delta"]},{r["rho"]},{r["budget"]},'
                     f'{ct["value"]:.3f}({ct["n"]}),{bt["value"]:.3f}({bt["n"]};{bt["k"]};{int(bt["anchor"])}),'
                     f'{cs["value"]:.3f}({cs["n"]}),{bs["value"]:.3f}({bs["n"]};{bs["k"]};{int(bs["anchor"])})')
    with open(os.path.join(OUT, "e6_budget_table.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # figure: heatmap of target success over (n,k) for symmetric, Delta=1600, rho=0 & 0.5
    g = REGIMES["symmetric"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, rho in zip(axes, [0.0, 0.5]):
        M = np.full((6, 24), np.nan)
        for k in range(1, 7):
            for n in range(1, 25):
                M[k - 1, n - 1] = exact_dp(g, Design(n=n, k=k, rho=rho), 1600)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
                       extent=[0.5, 24.5, 0.5, 6.5])
        nn = np.arange(1, 25)
        for B, ls in [(16, "--"), (32, ":")]:
            # approximate iso-cost curve n*k - k(k-1)/2 = B
            kk = np.linspace(1, 6, 50)
            ax.plot((B + kk * (kk - 1) / 2) / kk, kk, color="#333333", ls=ls, lw=1,
                    label=f"cost={B}")
        ax.set_xlim(0.5, 24.5)
        ax.set_title(f"target success, symmetric, Δ=1600, ρ={rho}", fontsize=9)
        ax.set_xlabel("levels n"); ax.set_ylabel("window k")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.savefig(os.path.join(OUT, "fig_e6_heatmap.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    log["e6_rows"] = len(out)


# ------------------------------------------ E7 full topology & anchor (MC / DP)
def e7_topologies():
    g = REGIMES["symmetric"]
    out = []
    for Delta in [400, 1600]:
        n = 10
        for name, d in [("chain", Design(n=n, k=1)),
                        ("anchor+chain", Design(n=n, k=1, anchor=True)),
                        ("window-2", Design(n=n, k=2)),
                        ("window-3", Design(n=n, k=3)),
                        ("full (all previous)", Design(n=n, k=n))]:
            for rho in [0.0, 0.5]:
                d.rho = rho
                mc = monte_carlo(g, d, Delta, n_runs=200_000, seed=3)
                out.append(dict(Delta=Delta, topology=name, rho=rho, cost=d.cost(),
                                target=float(mc["target"]), strict=float(mc["strict"]),
                                se=float(mc["target_se"])))
    save("e7_topologies.json", out)
    log["e7"] = out


# ------------------- E3b asymptotic absorption hazard vs Proposition 3 (small q)
def e3b_hazard():
    """Validate r_k = prod_m q_m^m as the asymptotic per-level absorption hazard
    (gamma=0, rho=0, stationary chain) in a strongly guard-favoured regime where
    the miss probabilities are small, and in the symmetric regime."""
    from nso import level_probs  # noqa: F401
    out = []
    for label, g, delta in [("small-q (beta=-700, delta=50)", Game(1, 1, -700.0), 50.0),
                            ("symmetric (beta=-300, delta=100)", REGIMES["symmetric"], 100.0)]:
        for k, rho in [(kk, rr) for rr in (0.0, 1.0) for kk in (1, 2, 3, 4)]:
            n = 80
            d = Design(n=n, k=k, rho=rho)
            # absorbing mass after level j = P(last k levels all compromised)
            # computed by a light re-implementation of the DP that tracks it
            x = np.arange(n + 1) * delta
            dist = {tuple([None] * (k - 1) + [1]): 1.0}
            A = []
            for j in range(1, n + 1):
                wl = list(range(j - k, j)); new = {}
                for st, pr in dist.items():
                    stat = dict(zip(wl, st)); stat[0] = 1
                    ps = [g.p(x[i], x[j]) if stat.get(i) == 1 else 0.0 for i in d.overseers(j)]
                    dp_ = detect_prob(ps, rho)
                    for s_new, w in ((1, dp_), (0, 1 - dp_)):
                        if w > 0:
                            ns = st[1:] + (s_new,); new[ns] = new.get(ns, 0) + pr * w
                dist = new
                A.append(sum(v for s, v in dist.items() if all(t == 0 for t in s)))
            A = np.array(A)
            h = (A[-1] - A[-2]) / max(1 - A[-2], 1e-300)
            q = np.array([1 - g.p(0, m * delta) for m in range(1, k + 1)])
            r = float(np.prod(q ** np.arange(1, k + 1))) if rho == 0 else float(np.prod(q))
            tf = 1 - exact_dp(g, d, n * delta)
            out.append(dict(regime=label, k=k, rho=rho, q=q.tolist(), hazard_exact=float(h),
                            r_k_pred=r, ratio=float(h / r) if r > 0 else None,
                            target_fail_n80=float(tf), prod_q=float(np.prod(q))))
    save("e3b_hazard.json", out)
    log["e3b"] = out


# ------------------------------------------------ E8 false-alarm accounting
def e8_false_alarms():
    """OR aggregation raises false alarms on benign models. With per-overseer
    false-positive rate phi (independent), a level with |S_j| overseers raises a
    false alarm w.p. 1-(1-phi)^|S_j|; expected false alarms over the pipeline is
    sum_j of that. Reported for the budget-optimal designs of E6 (symmetric)."""
    rows = []
    for phi in [0.01, 0.05]:
        for n, k, anchor in [(4, 1, False), (6, 3, False), (9, 4, False), (11, 3, False)]:
            d = Design(n=n, k=k, anchor=anchor)
            ea = sum(1 - (1 - phi) ** len(d.overseers(j)) for j in range(1, n + 1))
            rows.append(dict(phi=phi, n=n, k=k, cost=d.cost(), expected_false_alarms=ea,
                             p_any_false_alarm=1 - (1 - phi) ** d.cost()))
    save("e8_false_alarms.json", rows)
    log["e8"] = rows


if __name__ == "__main__":
    t0 = time.time()
    for f in [e1_validation, e2_chain, e3_accumulation, e4_rho, e3b_hazard, e5_gamma, e6_budget, e7_topologies, e8_false_alarms]:
        t = time.time(); f(); print(f.__name__, f"{time.time()-t:.1f}s", flush=True)
    log["runtime_s"] = time.time() - t0
    save("summary.json", log)
    print("total", log["runtime_s"])


