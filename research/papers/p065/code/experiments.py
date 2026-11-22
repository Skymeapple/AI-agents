"""All simulation experiments for the paper (except calibration, see calibrate.py).

E1  Proposition checks: monotonicity of total offloading and terminal skill in AI quality q
E2  Assessment informativeness: analytic corr(max(s,q), s) vs Monte Carlo, uniform and Beta skill
E3  Minimal proctored weight w*(q, kappa, beta): closed form (last period) vs dynamic programme
E4  Fading theorem: planner-optimal access with expertise reversal vs present-biased student
E5  Cohort policy simulation across AI quality q (5 regimes)
Outputs in ../results/
"""
import json, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import solve_student, simulate, step, S_GRID, A_GRID

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261122)

T, ETA, L = 20, 0.04, 1.0
K_BASE = 0.305   # median calibrated kappa for an unguarded assistant (calibrate.py)
K_TUTOR = 1.0    # calibrated kappa for a guardrailed tutor (calibrate.py)
Q_GRID = np.round(np.linspace(0, 1, 21), 3)
COL = {"Full engagement (benchmark)": "#000000", "No AI (shirking possible)": "#8A8A8A",
       "Open AI, w=0.3": "#C8553D", "Open AI, w=0.7": "#F28F3B",
       "Tutor (k=1), w=0.3": "#2D7DD2", "Tutor k(s), unrestricted": "#9B5DE5",
       "Tutor k(s) + fading, w=0.3": "#3BB273"}
summary = {}

# ---------------------------------------------------------------- E1
def e1():
    viol_A, viol_S, n = 0, 0, 0
    worst = 0.0
    for beta, c, w, s0, k in itertools.product([0.1, 0.5, 0.9], [0.005, 0.02, 0.05], [0.0, 0.3, 0.6],
                                                [0.1, 0.3, 0.5, 0.7], [0.0, 0.3, 0.6, 0.9]):
        totA, sT = [], []
        for q in Q_GRID:
            pol = solve_student(T, q, ETA, k, w=w, L=L, beta=beta, c=c)
            r = simulate(s0, pol, T, q, ETA, k)
            totA.append(r["A"].sum()); sT.append(r["sT"][0])
        totA, sT = np.array(totA), np.array(sT)
        n += 1
        viol_A += int(np.any(np.diff(totA) < -1e-9))
        d = np.diff(sT)
        if np.any(d > 1e-6):
            viol_S += 1; worst = max(worst, d.max())
    summary["E1"] = dict(n_configs=n, q_points=len(Q_GRID), violations_total_offloading_monotone=viol_A,
                         violations_terminal_skill_monotone=viol_S, worst_skill_increase=worst)
    print("E1", summary["E1"])

# ---------------------------------------------------------------- E2
def corr_uniform_closed(q):
    Em = (1 + q ** 2) / 2
    Esm = q ** 3 / 2 + (1 - q ** 3) / 3
    Em2 = q ** 3 + (1 - q ** 3) / 3
    cov = Esm - 0.5 * Em
    var = Em2 - Em ** 2
    return cov / np.sqrt(var * (1 / 12)) if var > 0 else 0.0


def e2():
    qs = np.linspace(0, 0.99, 34)
    s_u = RNG.uniform(0, 1, 200000)
    s_b = RNG.beta(2, 3, 200000)
    cf = np.array([corr_uniform_closed(q) for q in qs])
    mc_u = np.array([np.corrcoef(np.maximum(s_u, q), s_u)[0, 1] for q in qs])
    mc_b = np.array([np.corrcoef(np.maximum(s_b, q), s_b)[0, 1] for q in qs])
    # with grading noise sd 0.1 and a proctored component of weight w
    def comp(q, w, s):
        g = (1 - w) * np.maximum(s, q) + w * s + RNG.normal(0, 0.1, s.size)
        return np.corrcoef(g, s)[0, 1]
    ws = [0.0, 0.3, 0.5, 0.7]
    mc_w = {w: np.array([comp(q, w, s_b) for q in qs]) for w in ws}
    summary["E2"] = dict(max_abs_err_closed_vs_mc_uniform=float(np.max(np.abs(cf - mc_u))),
                         corr_uniform_at_q={str(q): float(corr_uniform_closed(q)) for q in [0.0, 0.5, 0.8, 0.9, 0.95]},
                         corr_beta23_at_q={str(round(q, 2)): float(np.corrcoef(np.maximum(s_b, q), s_b)[0, 1]) for q in [0.0, 0.5, 0.8, 0.9]},
                         noisy_composite_beta23_q0_vs_q09={str(w): [float(comp(0.0, w, s_b)), float(comp(0.9, w, s_b))] for w in ws})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(qs, cf, "k-", lw=2, label="closed form, s~U[0,1]")
    ax[0].plot(qs, mc_u, "o", ms=3, color="#C8553D", label="Monte Carlo, U[0,1]")
    ax[0].plot(qs, mc_b, "s", ms=3, color="#2D7DD2", label="Monte Carlo, Beta(2,3)")
    ax[0].set_xlabel("AI quality q"); ax[0].set_ylabel("corr(take-home score, skill)")
    ax[0].set_title("(a) Unproctored score informativeness"); ax[0].legend(frameon=False, fontsize=8)
    for w in ws:
        ax[1].plot(qs, mc_w[w], lw=2, label=f"proctored weight w={w}")
    ax[1].set_xlabel("AI quality q"); ax[1].set_ylabel("corr(course grade, skill)")
    ax[1].set_title("(b) Composite grade, Beta(2,3), noise sd 0.1"); ax[1].legend(frameon=False, fontsize=8)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_informativeness.png"), dpi=160); plt.close(fig)
    print("E2", summary["E2"])

# ---------------------------------------------------------------- E3
def offload_set_last_period(q, k, beta, c, w, Tn=T, eta=ETA, Lv=L):
    """Prop. 4: last-period offloading iff (1-w)(q-s)/T + c > beta (w+L) eta (1-k)(1-s).
    Returns the interval of s in [0,1] where offloading is optimal (linear in s => interval)."""
    s = np.linspace(0, 1, 100001)
    gain = (1 - w) * (q - s) / Tn + c - beta * (w + Lv) * eta * (1 - k) * (1 - s)
    m = gain > 0
    return (float(s[m].min()), float(s[m].max())) if m.any() else None


def w_target_dp(q, k, beta, c, s0=0.3, frac=0.95, wgrid=np.linspace(0, 1, 51)):
    """Minimal proctored weight such that terminal skill >= frac * full-engagement benchmark."""
    bench = simulate(s0, lambda t, s: np.zeros_like(s), T, q, ETA, k)["sT"][0]
    for w in wgrid:
        pol = solve_student(T, q, ETA, k, w=w, L=L, beta=beta, c=c)
        if simulate(s0, pol, T, q, ETA, k)["sT"][0] >= frac * bench - 1e-12:
            return float(w)
    return np.nan  # infeasible: no w in [0,1] reaches the target


def e3():
    qs = np.linspace(0.1, 1.0, 10)
    res = {"offload_interval_last_period": {}}
    for q, k, w in itertools.product([0.5, 0.9], [0.0, 0.3, 0.6], [0.3, 0.7]):
        res["offload_interval_last_period"][f"q={q},k={k},w={w}"] = offload_set_last_period(q, k, 0.5, 0.01, w)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, col in zip([0.0, 0.3, 0.6, 0.9], ["#C8553D", "#F28F3B", "#2D7DD2", "#3BB273"]):
        dp = [w_target_dp(q, k, 0.5, 0.01) for q in qs]
        res[f"w_target_vs_q_kappa={k}"] = dict(q=qs.tolist(), w=dp)
        ax[0].plot(qs, dp, "o-", color=col, label=f"kappa={k}")
    ax[0].set_xlabel("AI quality q"); ax[0].set_ylabel("minimal proctored weight w_95")
    ax[0].set_title("(a) w_95 vs q (beta=0.5, c=0.01); gaps = infeasible", fontsize=10)
    ax[0].set_ylim(-0.03, 1.03); ax[0].legend(frameon=False, fontsize=8)
    ks = np.linspace(0, 0.95, 20)
    for beta, col in zip([0.25, 0.5, 1.0], ["#C8553D", "#F28F3B", "#2D7DD2"]):
        dp = [w_target_dp(0.9, k, beta, 0.01) for k in ks]
        res[f"w_target_vs_kappa_beta={beta}"] = dict(kappa=ks.tolist(), w=dp)
        ax[1].plot(ks, dp, "o-", color=col, ms=3, label=f"beta={beta}")
    ax[1].set_xlabel("retained engagement kappa (guardrail strength)"); ax[1].set_ylabel("w_95")
    ax[1].set_title("(b) Guardrails substitute for proctoring (q=0.9)", fontsize=10)
    ax[1].set_ylim(-0.03, 1.03); ax[1].legend(frameon=False, fontsize=8)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_wstar.png"), dpi=160); plt.close(fig)
    summary["E3"] = res
    print("E3", json.dumps(res)[:3000])

# ---------------------------------------------------------------- E4
def planner_dp(k0, k1, Tn=T, eta=ETA):
    V = S_GRID.copy(); pol = np.zeros((Tn, len(S_GRID)))
    S, A = np.meshgrid(S_GRID, A_GRID, indexing="ij")
    for t in range(Tn - 1, -1, -1):
        tot = np.interp(step(S, A, eta, k0, k1), S_GRID, V) - 1e-12 * A
        idx = np.argmax(tot, 1); pol[t] = A_GRID[idx]; V = tot[np.arange(len(S_GRID)), idx]
    return pol, V


def e4():
    k0, k1 = 1.4, 0.8          # hypothetical good tutor: helps novices, hurts experts; kappa=1 at s=0.5
    s_bar = (k0 - 1) / k1
    pol, V = planner_dp(k0, k1)
    greedy = lambda t, s: (k0 - k1 * s > 1).astype(float)
    # compare DP-optimal value to greedy value from many s0
    s0s = np.linspace(0, 0.95, 40)
    v_greedy = simulate(s0s, greedy, T, 1.0, ETA, k0, k1)["sT"]
    v_dp = np.interp(s0s, S_GRID, V)
    s0 = 0.15
    r_pl = simulate(s0, greedy, T, 0.9, ETA, k0, k1)
    r_no = simulate(s0, lambda t, s: np.zeros_like(s), T, 0.9, ETA, k0, k1)
    r_al = simulate(s0, lambda t, s: np.ones_like(s), T, 0.9, ETA, k0, k1)
    stud = solve_student(T, 0.9, ETA, k0, k1, w=0.3, L=L, beta=0.3, c=0.02)
    r_st = simulate(s0, stud, T, 0.9, ETA, k0, k1)
    summary["E4"] = dict(k0=k0, k1=k1, fading_threshold=s_bar,
                         max_abs_gap_dp_vs_greedy=float(np.max(np.abs(v_greedy - v_dp))),
                         sT_from_0p15=dict(planner_fading=float(r_pl["sT"][0]), never=float(r_no["sT"][0]),
                                           always=float(r_al["sT"][0]), student_beta0p3=float(r_st["sT"][0])),
                         student_total_offload=float(r_st["A"].sum()), planner_total_offload=float(r_pl["A"].sum()))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    t = np.arange(T + 1)
    for r, lab, col in [(r_pl, "planner: fading (Prop. 5)", "#3BB273"), (r_no, "never use AI", "#4C5B61"),
                        (r_al, "always use AI", "#C8553D"), (r_st, "present-biased student (beta=0.3)", "#F28F3B")]:
        ax[0].plot(t, r["S"][:, 0], lw=2, label=lab, color=col)
    ax[0].axhline(s_bar, ls=":", color="k"); ax[0].text(0.3, s_bar + 0.01, "fading threshold", fontsize=8)
    ax[0].set_xlabel("practice period t"); ax[0].set_ylabel("unassisted skill s_t")
    ax[0].set_title("(a) Skill trajectories, kappa(s)=1.4-0.8s"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].step(np.arange(T), r_pl["A"][:, 0], where="post", lw=2, color="#3BB273", label="planner")
    ax[1].step(np.arange(T), r_st["A"][:, 0], where="post", lw=2, color="#F28F3B", label="student")
    ax[1].set_xlabel("practice period t"); ax[1].set_ylabel("offload share a_t")
    ax[1].set_title("(b) Access schedules"); ax[1].legend(frameon=False, fontsize=8)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_fading.png"), dpi=160); plt.close(fig)
    print("E4", summary["E4"])

# ---------------------------------------------------------------- E5
def e5():
    N = 6000
    s0 = RNG.beta(2, 3, N)
    betas = np.array([0.1, 0.3, 0.5, 0.7, 0.9]); cs = np.array([0.005, 0.01, 0.02])
    bi = RNG.integers(0, len(betas), N); ci = RNG.integers(0, len(cs), N)
    lowq = s0 <= np.quantile(s0, 1 / 3); highq = s0 >= np.quantile(s0, 2 / 3)
    k0f, k1f = 1.4, 0.8
    regimes = {
        "Full engagement (benchmark)": dict(k0=K_BASE, k1=0, w=0.3, allowed=lambda t, S, A: A == 0),
        "No AI (shirking possible)": dict(k0=0.0, k1=0, w=0.3, allowed=None, qfix=0.0),
        "Open AI, w=0.3": dict(k0=K_BASE, k1=0, w=0.3, allowed=None),
        "Open AI, w=0.7": dict(k0=K_BASE, k1=0, w=0.7, allowed=None),
        "Tutor (k=1), w=0.3": dict(k0=K_TUTOR, k1=0, w=0.3, allowed=None),
        "Tutor k(s), unrestricted": dict(k0=k0f, k1=k1f, w=0.3, allowed=None),
        "Tutor k(s) + fading, w=0.3": dict(k0=k0f, k1=k1f, w=0.3,
                                      allowed=lambda t, S, A: (A == 0) | (k0f - k1f * S > 1)),
    }
    out = {r: {m: [] for m in ["sT", "gap", "grade", "corr_grade_skill", "corr_take_home_skill", "offload", "sT_low"]}
           for r in regimes}
    for rname, R in regimes.items():
        for qq in Q_GRID:
            q = R.get("qfix", qq)
            sT = np.zeros(N); Pb = np.zeros(N); Ab = np.zeros(N)
            for i, b in enumerate(betas):
                for j, c in enumerate(cs):
                    m = (bi == i) & (ci == j)
                    pol = solve_student(T, q, ETA, R["k0"], R["k1"], w=R["w"], L=L, beta=b, c=c, allowed=R["allowed"])
                    r = simulate(s0[m], pol, T, q, ETA, R["k0"], R["k1"])
                    sT[m] = r["sT"]; Pb[m] = r["Pbar"]; Ab[m] = r["A"].mean(0)
            noise = RNG.normal(0, 0.05, N)
            grade = (1 - R["w"]) * Pb + R["w"] * sT + noise
            o = out[rname]
            o["sT"].append(float(sT.mean())); o["sT_low"].append(float(sT[lowq].mean()))
            o["gap"].append(float(sT[highq].mean() - sT[lowq].mean()))
            o["grade"].append(float(grade.mean()))
            o["corr_grade_skill"].append(float(np.corrcoef(grade, sT)[0, 1]))
            o["corr_take_home_skill"].append(float(np.corrcoef(Pb + RNG.normal(0, 0.05, N), sT)[0, 1]))
            o["offload"].append(float(Ab.mean()))
        print("E5", rname, "sT@q=0,0.5,0.9:", [round(out[rname]["sT"][k], 3) for k in [0, 10, 18]])
    summary["E5"] = dict(q=Q_GRID.tolist(), N=N, regimes=out)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    panels = [("sT", "(a) mean terminal unassisted skill"), ("gap", "(b) skill gap: top vs bottom prior tercile"),
              ("corr_grade_skill", "(c) corr(course grade, terminal skill)"), ("grade", "(d) mean course grade")]
    for a, (key, title) in zip(ax.flat, panels):
        for rname in regimes:
            a.plot(Q_GRID, out[rname][key], lw=2, color=COL[rname], label=rname,
                   ls="--" if rname.startswith("Full") else "-")
        a.set_title(title, fontsize=10); a.set_xlabel("AI quality q"); a.spines[["top", "right"]].set_visible(False)
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_cohort.png"), dpi=160); plt.close(fig)


def fig1_calibration():
    import csv
    rows = list(csv.DictReader(open(os.path.join(OUT, "calibration.csv"))))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for arm, col, mk in [("GPT_Base", "#C8553D", "o"), ("GPT_Tutor", "#2D7DD2", "s")]:
        rr = [r for r in rows if r["arm"] == arm and r["feasible"] == "True"]
        ax.scatter([float(r["a"]) for r in rr], [float(r["kappa"]) for r in rr], c=col, marker=mk,
                   label=f"{arm.replace('_', ' ')} ({len(rr)} feasible cells)")
    ax.axhline(1, ls=":", color="k"); ax.text(0.02, 1.03, "kappa=1: no learning loss", fontsize=8)
    ax.set_xlabel("implied offload share a"); ax.set_ylabel("implied retained engagement kappa")
    ax.set_ylim(-0.05, 1.3); ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Inverse calibration to Bastani et al. (2025) moments", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_calibration.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    fig1_calibration()
    import sys
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5"]
    prev = os.path.join(OUT, "summary.json")
    if os.path.exists(prev) and len(which) < 5:
        summary.update(json.load(open(prev)))
    for w in which: globals()[w]()
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("saved summary.json")
