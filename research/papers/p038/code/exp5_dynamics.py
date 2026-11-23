"""E5: valley crossing under net capability improvement (time-varying environment).
E6: heritable self-modification -> mutational escape from a subcritical population.
E7: synchronized sweeps vs uniform patrols at equal detection budget (random environment).
E8: certification sample complexity for an 'R<1' safety case."""
import json, os, time
import numpy as np
from scipy.stats import beta as betad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(7)
t0 = time.time()
res = {}

# ======================= E5
def poisson_fp(m):
    q = 0.0
    for _ in range(100000):
        qn = np.exp(m * (q - 1))
        if abs(qn - q) < 1e-14: break
        q = qn
    return q

def surv_varying(ms, m_cap):
    """Exact survival of a single founder: backward PGF composition for Poisson
    offspring with means ms[0..T-1], then m_cap forever."""
    u = 1 - poisson_fp(m_cap)
    for m in ms[::-1]:
        u = -np.expm1(-m * u)      # 1 - f(1-u), numerically stable for tiny u
    return u

def schedule(R0, g, m_cap):
    ms = []; t = 0
    while True:
        m = R0 * np.exp(g * t)
        if m >= m_cap: break
        ms.append(m); t += 1
        if t > 100000: break
    return np.array(ms)

m_cap = 1.5
rows = []
for R0 in [0.3, 0.5, 0.7, 0.9]:
    for g in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        ms = schedule(R0, g, m_cap)
        S = surv_varying(ms, m_cap)
        M = np.concatenate([[1.0], np.cumprod(ms)])          # M_0..M_T
        # after cap, M grows geometrically: add tail sum of 1/M
        tail = (1 / M[-1]) * (1 / m_cap) / (1 - 1 / m_cap)
        lower = 1.0 / (1.0 + np.sum(1 / M[1:]) + tail)          # Prop. 4 lower bound (Poisson, V=1)
        upper = M.min()                                       # Prop. 4 upper bound
        laplace = (np.log(R0)) ** 2 / (2 * g)
        rows.append(dict(R0=R0, g=g, T_to_cap=len(ms), surv=S, lower=lower, upper=float(upper),
                         neg_log_surv=float(-np.log(S)) if S > 0 else np.inf, valley_exponent=laplace))
        print(f"E5 R0={R0} g={g}: surv={S:.3e} in [{lower:.3e},{upper:.3e}] exp(-L)={np.exp(-laplace):.3e}")
# MC check of a few points
def mc_varying(ms, m_cap, reps=20000, cap=3000):
    hit = 0
    for _ in range(reps):
        N = 1
        for m in ms:
            N = rng.poisson(m * N)
            if N == 0: break
        if N == 0: continue
        for _ in range(400):
            if N == 0 or N > cap: break
            N = rng.poisson(m_cap * N)
        hit += N > 0
    return hit / reps
mc = []
for R0, g in [(0.7, 0.1), (0.5, 0.2), (0.9, 0.02)]:
    ms = schedule(R0, g, m_cap); s_mc = mc_varying(ms, m_cap)
    s_ex = surv_varying(ms, m_cap)
    mc.append(dict(R0=R0, g=g, exact=s_ex, mc=s_mc, se=float(np.sqrt(s_mc * (1 - s_mc) / 20000))))
    print("MC", mc[-1])
# defender race: net g = gamma - eta
race = []
for gam in [0.05, 0.1]:
    for eta in [0.0, 0.025, 0.05, 0.075, 0.1, 0.15]:
        g = gam - eta
        S = surv_varying(schedule(0.7, g, m_cap), m_cap) if g > 0 else 0.0
        race.append(dict(gamma=gam, eta=eta, g=g, surv=S))
res["E5"] = dict(rows=rows, mc=mc, race=race, m_cap=m_cap)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for R0 in [0.3, 0.5, 0.7, 0.9]:
    rr = [r for r in rows if r["R0"] == R0]
    axs[0].plot([r["g"] for r in rr], [r["surv"] for r in rr], "o-", ms=3, label=f"R0={R0}")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("net improvement rate g per epoch")
axs[0].set_ylabel("survival prob. of one founder"); axs[0].legend(fontsize=7); axs[0].set_title("E5: valley crossing")
x = np.array([r["valley_exponent"] for r in rows]); y = np.array([r["neg_log_surv"] for r in rows])
ok = np.isfinite(y)
slope, icpt = np.polyfit(x[ok], y[ok], 1)
axs[1].scatter(x, y, s=10, color="#3b6ea5"); xx = np.linspace(0, x.max(), 10)
axs[1].plot(xx, slope * xx + icpt, "k--", lw=0.8, label=f"fit slope {slope:.3f}")
axs[1].set_xlabel("(ln R0)^2 / (2g)"); axs[1].set_ylabel("-ln survival"); axs[1].legend(fontsize=7)
axs[1].set_title("scaling law"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_e5_valley.png", dpi=150)
res["E5"]["fit_slope"] = float(slope); res["E5"]["fit_intercept"] = float(icpt)
print("E5 slope", slope, icpt)

# ======================= E6: mutational escape
def escape_exact(Rs, mu, iters=200000):
    L = len(Rs) - 1
    q = np.zeros(L + 1)
    for _ in range(iters):
        qn = q.copy()
        qn[L] = np.exp(Rs[L] * (q[L] - 1))
        for i in range(L - 1, -1, -1):
            qn[i] = np.exp(Rs[i] * ((1 - mu) * q[i] + mu * q[i + 1] - 1))
        if np.max(abs(qn - q)) < 1e-15: q = qn; break
        q = qn
    return 1 - q

def escape_approx(Rs, mu):
    L = len(Rs) - 1
    u = 1 - poisson_fp(Rs[L])
    for i in range(L - 1, -1, -1):
        u = Rs[i] * mu * u / (1 - Rs[i] * (1 - mu))
    return u

def escape_mc(Rs, mu, N0, reps=4000, cap=2000, gens=3000):
    L = len(Rs) - 1; hits = 0
    for _ in range(reps):
        N = np.zeros(L + 1, int); N[0] = N0
        for _g in range(gens):
            if N.sum() == 0 or N[L] > cap: break
            new = np.zeros_like(N)
            for i in range(L + 1):
                if N[i] == 0: continue
                k = rng.poisson(Rs[i] * N[i])
                if i < L:
                    up = rng.binomial(k, mu); new[i] += k - up; new[i + 1] += up
                else:
                    new[i] += k
            N = new
        hits += N[L] > cap or (N.sum() > 0)
    return hits / reps

e6 = []
for R0 in [0.5, 0.8, 0.9, 0.95, 0.99]:
    for L in [1, 2]:
        for mu in [1e-3, 1e-2]:
            Rs = [R0] * L + [1.5]
            u = escape_exact(Rs, mu)[0]; ua = escape_approx(Rs, mu)
            N0 = 100
            e6.append(dict(R0=R0, L=L, mu=mu, esc_per_founder=float(u), approx=float(ua),
                           esc_N0_100=float(1 - (1 - u) ** N0)))
            print(f"E6 R0={R0} L={L} mu={mu}: per-founder {u:.3e} approx {ua:.3e}  N0=100: {1-(1-u)**N0:.3e}")
mc6 = []
for R0, L, mu, N0 in [(0.9, 1, 1e-2, 20), (0.95, 1, 1e-3, 100), (0.8, 2, 1e-2, 200)]:
    Rs = [R0] * L + [1.5]
    u = escape_exact(Rs, mu)[0]
    s = escape_mc(Rs, mu, N0, reps=10000)
    mc6.append(dict(R0=R0, L=L, mu=mu, N0=N0, exact=float(1 - (1 - u) ** N0), mc=s,
                    se=float(np.sqrt(s * (1 - s) / 10000))))
    print("E6 MC", mc6[-1])
res["E6"] = dict(rows=e6, mc=mc6, R_top=1.5)
fig, ax = plt.subplots(figsize=(5, 3.4))
R0s = np.linspace(0.3, 0.995, 80)
for L, mu, c in [(1, 1e-2, "#3b6ea5"), (1, 1e-3, "#7aa6d6"), (2, 1e-2, "#b5523b"), (2, 1e-3, "#e09a85")]:
    ax.plot(R0s, [escape_exact([r] * L + [1.5], mu)[0] for r in R0s], color=c, label=f"L={L}, mu={mu:g}")
ax.set_yscale("log"); ax.set_xlabel("R of unmodified replicas (subcritical)")
ax.set_ylabel("escape prob. per founder"); ax.legend(fontsize=7)
ax.set_title("E6: near-critical populations incubate escape"); fig.tight_layout()
fig.savefig(f"{OUT}/fig_e6_escape.png", dpi=150)

# ======================= E7: sweeps vs uniform
b, dbar = 1.5, 0.3
def sweep_surv(rho, reps=6000, gens=300, cap=5000, N0=1):
    d = dbar / rho; hits = 0
    for _ in range(reps):
        N = N0
        for _g in range(gens):
            if N == 0 or N > cap: break
            dt = d if rng.random() < rho else 0.0
            N = rng.poisson(b * (1 - dt) * N)
        hits += N > 0
    return hits / reps
e7 = []
for rho in [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.31]:
    d = dbar / rho
    lyap = np.log(b) + rho * np.log(1 - d) if d < 1 else -np.inf
    s = sweep_surv(rho)
    e7.append(dict(rho=rho, sweep_detection=d, mean_m=b * (1 - dbar), lyapunov=float(lyap), surv=s,
                   se=float(np.sqrt(s * (1 - s) / 6000))))
    print("E7", e7[-1])
res["E7"] = dict(b=b, dbar=dbar, rows=e7)
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot([r["rho"] for r in e7], [r["surv"] for r in e7], "o-", label="survival prob. (MC)")
ax2 = ax.twinx(); ax2.plot([r["rho"] for r in e7], [r["lyapunov"] for r in e7], "s--", color="#b5523b",
                           label="E log m (Lyapunov)")
ax2.axhline(0, color="#b5523b", lw=0.6)
ax.set_xlabel("fraction of epochs with a sweep (rho); same mean detection 0.3")
ax.set_ylabel("survival prob."); ax2.set_ylabel("E log m", color="#b5523b")
ax.set_title("E7: concentrate detection in synchronized sweeps"); ax.invert_xaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e7_sweeps.png", dpi=150)

# ======================= E8: certification
def cp_upper(x, n, alpha):
    return 1.0 if x >= n else betad.ppf(1 - alpha, x + 1, n - x)
K, delta, sigma = 4, 0.05, 1.0
alpha = delta / K
e8 = dict(K=K, delta=delta, rows=[])
for h in [0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
    pstar = h / (sigma + h)                    # stage odds < 1/sigma  <=>  p < h/(sigma+h)
    n_req = int(np.ceil(np.log(alpha) / np.log(1 - pstar)))   # zero observed successes
    e8["rows"].append(dict(h=h, p_threshold=pstar, n_trials_zero_success=n_req))
    print("E8", h, pstar, n_req)
for n in [10, 30, 100]:
    pu = cp_upper(0, n, alpha)
    e8[f"n={n}_zero_success_pUB"] = pu
    e8[f"n={n}_min_certifiable_h"] = sigma * pu / (1 - pu)
    print(f"n={n}: pUB={pu:.4f}, min h certifiable={sigma*pu/(1-pu):.4f}")
# power: true p, h -> probability to certify with n trials (single stage)
power = []
for ptrue, h in [(0.001, 0.01), (0.003, 0.01), (0.001, 0.03)]:
    for n in [30, 100, 300, 1000, 3000]:
        xs = rng.binomial(n, ptrue, 20000)
        pus = np.array([cp_upper(x, n, alpha) for x in range(xs.max() + 1)])[xs]
        power.append(dict(p=ptrue, h=h, n=n, power=float(np.mean(sigma * pus / (1 - pus) < h))))
e8["power"] = power
res["E8"] = e8
fig, ax = plt.subplots(figsize=(5, 3.4))
hs = np.logspace(-3, np.log10(0.5), 50)
ax.plot(hs, [np.ceil(np.log(alpha) / np.log(1 - x / (1 + x))) for x in hs], color="#3b6ea5")
ax.axhline(10, color="k", ls=":", lw=0.8); ax.text(1.2e-3, 12, "10 samples/task", fontsize=7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("detection prob. per failed try h (bottleneck stage)")
ax.set_ylabel("trials needed (0 successes)"); ax.set_title("E8: trials to certify R<1 (95%, K=4)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e8_certify.png", dpi=150)

json.dump(res, open(f"{OUT}/e5_e8_dynamics.json", "w"), indent=1, default=float)
print("done", time.time() - t0)
