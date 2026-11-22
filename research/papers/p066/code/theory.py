"""
Reduced (one-state) model of AI offloading during skill formation.
Analytic thresholds (Propositions 1-4 in paper.md) + numerical verification.

State: skill s in [0,1]. Each period the learner chooses a in {0,1}
(attempt alone vs. offload to AI of quality q, pedagogical retention sigma).
Payoff  u = w[(1-a)s + a q] - c(1-a)(1 - theta s)
Skill   s' = s + eta[(1-a) + sigma a](eps + s)(1 - s) - lam s
Perceived continuation (linear shadow value): beta * delta * v * s'.

Run:  python3 theory.py      -> results/theory_*.json, results/fig_theory_*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

BASE = dict(w=1.0, c=0.3, theta=0.5, q=0.9, sigma=0.0, eta=0.15, lam=0.03,
            eps=0.05, beta=0.7, delta=0.98, v=25.0, omega=0.5)


def G(s, eps):
    return (eps + s) * (1.0 - s)


def D(s, p):
    """Net perceived benefit of attempting alone (a=0) minus offloading (a=1)."""
    B = p["beta"] * p["delta"] * p["v"] * p["eta"] * (1 - p["sigma"])
    return p["w"] * (s - p["q"]) - p["c"] * (1 - p["theta"] * s) + B * G(s, p["eps"])


def s_low(p):
    """Lower root of D (offloading threshold). Returns (s_L, flag).
    flag: 'interior' root in (0,1); 'never' (attempt everywhere); 'always' (offload everywhere)."""
    B = p["beta"] * p["delta"] * p["v"] * p["eta"] * (1 - p["sigma"])
    b1 = p["w"] + p["c"] * p["theta"] + B * (1 - p["eps"])
    b0 = B * p["eps"] - p["w"] * p["q"] - p["c"]
    if B <= 1e-14:
        r = -b0 / b1
    else:
        disc = b1 ** 2 + 4 * B * b0
        if disc < 0:
            return 1.0, "always"
        r = (b1 - np.sqrt(disc)) / (2 * B)
    if r <= 0:
        return 0.0, "never"
    if r >= 1:
        return 1.0, "always"
    return float(r), "interior"


def s_star(eta_eff, lam, eps):
    """Positive steady state of s' = s + eta_eff G(s) - lam s."""
    # eta_eff (eps + s)(1-s) = lam s  ->  -eta s^2 + (eta(1-eps) - lam) s + eta eps = 0
    if eta_eff <= 0:
        return 0.0
    a, b, c0 = -eta_eff, eta_eff * (1 - eps) - lam, eta_eff * eps
    disc = b * b - 4 * a * c0
    return float((-b - np.sqrt(disc)) / (2 * a))


def step(s, a, p):
    L = (1 - a) + p["sigma"] * a
    return np.clip(s + p["eta"] * L * G(s, p["eps"]) - p["lam"] * s, 0, 1)


def simulate_threshold_policy(s0, p, T=600):
    s = np.array(s0, dtype=float)
    for _ in range(T):
        a = (D(s, p) < 0).astype(float)
        s = step(s, a, p)
    return s


def dp_stationary(p, n=401, iters=3000, tol=1e-10):
    """Infinite-horizon DP: naive quasi-hyperbolic learner. Long-run self values
    V(s) = max_a u + delta V(s'); current self picks argmax u + beta delta V(s')."""
    grid = np.linspace(0, 1, n)
    V = np.zeros(n)
    acts = np.array([0.0, 1.0])
    u = np.stack([p["w"] * ((1 - a) * grid + a * p["q"]) - p["c"] * (1 - a) * (1 - p["theta"] * grid)
                  + p.get("omega", 0.5) * grid for a in acts])
    nxt = np.stack([step(grid, a, p) for a in acts])
    for _ in range(iters):
        Vn = np.interp(nxt, grid, V)
        Vnew = np.max(u + p["delta"] * Vn, axis=0)
        if np.max(np.abs(Vnew - V)) < tol:
            V = Vnew
            break
        V = Vnew
    Vn = np.interp(nxt, grid, V)
    pol = np.argmax(u + p["beta"] * p["delta"] * Vn, axis=0)
    # policy under naive self actually executed:
    return grid, V, acts[pol]


def dp_threshold(p, **kw):
    grid, V, pol = dp_stationary(p, **kw)
    # lowest skill at which the learner attempts (a=0), and check single crossing
    att = pol == 0
    if not att.any():
        return 1.0, True
    first = np.argmax(att)
    single = bool(att[first:].all())
    return float(grid[first]), single


def main():
    out = {}
    p = dict(BASE)
    sL, flag = s_low(p)
    sst = s_star(p["eta"], p["lam"], p["eps"])
    out["baseline"] = dict(params=p, s_L=sL, flag=flag, s_star_noAI=sst)

    # --- Prop 1-2 verification: bistability (basin boundary = s_L)
    s0 = np.linspace(0.005, 0.995, 199)
    sT = simulate_threshold_policy(s0, p)
    trapped = sT < 0.01
    boundary = float(s0[~trapped].min()) if (~trapped).any() else None
    out["prop2_verification"] = dict(
        analytic_s_L=sL, simulated_basin_boundary=boundary,
        grid_step=float(s0[1] - s0[0]),
        high_attractor=float(np.median(sT[~trapped])) if (~trapped).any() else None,
        s_star=sst)

    # --- Prop 3: comparative statics (finite differences, analytic)
    cs = {}
    for k, h in [("q", 1e-4), ("beta", 1e-4), ("eta", 1e-5), ("c", 1e-4), ("v", 1e-3), ("sigma", 1e-4)]:
        pp, pm = dict(p), dict(p)
        pp[k] += h; pm[k] -= h
        cs[k] = (s_low(pp)[0] - s_low(pm)[0]) / (2 * h)
    out["prop3_dsL_dx"] = cs

    # --- beta*eta invariance (analytic in reduced model)
    inv = []
    for be in [0.4, 0.6, 0.8, 1.0]:
        pp = dict(p); pp["beta"] = be; pp["eta"] = p["beta"] * p["eta"] / be
        inv.append(dict(beta=be, eta=pp["eta"], s_L=s_low(pp)[0]))
    out["beta_eta_invariance_reduced"] = inv

    # --- Same check in full stationary DP (v endogenous); threshold policy and invariance
    PDP = dict(p, q=0.7)  # DP checks use q=0.7 (< no-AI steady state s*), see paper Sec. 4.4
    out["dp_base_params"] = PDP
    dpres = []
    for be in [0.5, 0.7, 1.0]:
        for prod in [0.07, 0.105]:
            pp = dict(PDP); pp["beta"] = be; pp["eta"] = prod / be
            thr, single = dp_threshold(pp)
            dpres.append(dict(beta=be, eta=pp["eta"], beta_eta=prod, dp_threshold=thr, single_crossing=single))
    out["dp_threshold_checks"] = dpres
    # DP comparative statics in q and beta
    dpq = []
    for q in [0.5, 0.7, 0.9, 1.0]:
        pp = dict(PDP); pp["q"] = q
        dpq.append(dict(q=q, dp_threshold=dp_threshold(pp)[0], reduced_sL=s_low(pp)[0]))
    out["dp_vs_q"] = dpq
    dpb = []
    for be in [0.4, 0.6, 0.8, 1.0]:
        pp = dict(PDP); pp["beta"] = be
        dpb.append(dict(beta=be, dp_threshold=dp_threshold(pp)[0], reduced_sL=s_low(pp)[0]))
    out["dp_vs_beta"] = dpb

    # --- Prop 4: tutor retention sigma; long-run skill and sigma*
    sig_grid = np.linspace(0, 1, 201)
    lr = []
    for sg in sig_grid:
        pp = dict(p); pp["sigma"] = sg
        sLs = s_low(pp)[0]
        s_sig = s_star(sg * pp["eta"], pp["lam"], pp["eps"])
        sT0 = float(simulate_threshold_policy([0.0], pp, T=3000)[0])
        lr.append(dict(sigma=float(sg), s_L=sLs, s_sigma=s_sig, long_run_from_zero=sT0))
    crit = [r["sigma"] for r in lr if r["s_sigma"] >= r["s_L"]]
    out["prop4_sigma_star"] = crit[0] if crit else None
    out["prop4_table"] = lr[::20]

    with open(os.path.join(RES, "theory_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ss = np.linspace(0, 1, 400)
    for be, col in [(0.5, "#c0392b"), (0.7, "#e67e22"), (1.0, "#2c7fb8")]:
        pp = dict(p); pp["beta"] = be
        ax[0].plot(ss, D(ss, pp), color=col, label=f"beta={be}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("skill s"); ax[0].set_ylabel("D(s): benefit of attempting alone")
    ax[0].set_title("A. Offloading threshold s_L (root of D)"); ax[0].legend(frameon=False)

    s0 = np.linspace(0.02, 0.98, 25)
    traj = np.zeros((60, len(s0))); s = s0.copy()
    for t in range(60):
        traj[t] = s; s = step(s, (D(s, p) < 0).astype(float), p)
    for j in range(len(s0)):
        ax[1].plot(traj[:, j], color="#c0392b" if s0[j] < sL else "#2c7fb8", lw=0.9)
    ax[1].axhline(sL, ls="--", color="gray"); ax[1].text(40, sL + 0.02, "s_L", color="gray")
    ax[1].set_xlabel("period"); ax[1].set_ylabel("skill"); ax[1].set_title("B. Bistability under answer-AI (beta=0.7)")

    B_, E_ = np.meshgrid(np.linspace(0.3, 1.0, 120), np.linspace(0.03, 0.3, 120))
    Z = np.zeros_like(B_)
    for i in range(B_.shape[0]):
        for j in range(B_.shape[1]):
            pp = dict(p); pp["beta"] = B_[i, j]; pp["eta"] = E_[i, j]
            Z[i, j] = s_low(pp)[0]
    cf = ax[2].contourf(B_, E_, Z, levels=20, cmap="viridis")
    ax[2].contour(B_, E_, Z, levels=[0.2, 0.4, 0.6], colors="w", linewidths=0.8)
    plt.colorbar(cf, ax=ax[2], label="s_L")
    ax[2].set_xlabel("present bias beta"); ax[2].set_ylabel("plasticity eta")
    ax[2].set_title("C. Iso-threshold curves: beta*eta = const")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_theory_threshold.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sig_grid, [r["s_L"] for r in lr], label="threshold s_L(sigma)")
    ax.plot(sig_grid, [r["s_sigma"] for r in lr], label="tutor-only steady state s_sigma")
    ax.plot(sig_grid, [r["long_run_from_zero"] for r in lr], "k:", label="long-run skill from s0=0")
    if crit:
        ax.axvline(crit[0], color="gray", ls="--"); ax.text(crit[0] + 0.01, 0.05, f"sigma*={crit[0]:.2f}")
    ax.set_xlabel("pedagogical retention sigma of AI help"); ax.set_ylabel("skill")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Guardrail retention needed to escape the trap")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_theory_sigma.png"), dpi=150); plt.close()
    # long-run skill map over (sigma, q) from s0 = 0.3
    sgs = np.linspace(0, 1, 41); qs = np.linspace(0.3, 1.0, 36)
    M = np.zeros((len(qs), len(sgs)))
    for i, qq in enumerate(qs):
        for j, sg in enumerate(sgs):
            pp = dict(p); pp["q"] = qq; pp["sigma"] = sg
            M[i, j] = simulate_threshold_policy([0.3], pp, T=2000)[0]
    np.save(os.path.join(RES, "theory_sigma_q_map.npy"), M)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    cf = ax.contourf(sgs, qs, M, levels=20, cmap="magma"); plt.colorbar(cf, ax=ax, label="long-run skill (s0=0.3)")
    ax.set_xlabel("pedagogical retention sigma"); ax.set_ylabel("AI in-the-moment quality q")
    ax.set_title("Long-run skill under perpetual AI access")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_theory_sigma_q_map.png"), dpi=150); plt.close()
    out["sigma_q_map_summary"] = dict(min=float(M.min()), max=float(M.max()),
                                      at_sigma0_q0p9=float(M[np.argmin(abs(qs-0.9)), 0]),
                                      at_sigma0p5_q0p9=float(M[np.argmin(abs(qs-0.9)), 20]),
                                      at_sigma0_q0p4=float(M[np.argmin(abs(qs-0.4)), 0]))
    with open(os.path.join(RES, "theory_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ["baseline", "prop2_verification", "prop3_dsL_dx",
                                          "beta_eta_invariance_reduced", "dp_threshold_checks",
                                          "dp_vs_q", "dp_vs_beta", "prop4_sigma_star", "sigma_q_map_summary"]}, indent=1))


if __name__ == "__main__":
    main()
