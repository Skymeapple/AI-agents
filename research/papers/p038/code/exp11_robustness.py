"""E11 (revision r1): stress tests of the modelling assumptions behind the headline claims.
 (a) continuous time: attempt duration and defender flag-to-removal latency (Prop. 6);
     exact survival + attempt-level MC; Malthusian rate vs event-driven calendar-time simulation
 (b) coordinated global response after a lag T_G following the first flag
 (c) adaptive defender (detection rises with failures) and learning agent (success rises with tries)
 (d) common-mode / staggered detection across M providers (random environment; Prop. 10)
 (e) density dependence (carrying capacity)
"""
import json, os, time, heapq
import numpy as np
from scipy.stats import binom
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model_ext import R_latency, surv_latency, malthus_latency

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(11)
t0 = time.time()
res = {}

# ------------------------------------------------------------------ (a) latency
P, sigma = 0.05, 1.0
Ls = [0, 5, 10, 20]
crit = {}
for L in Ls:
    x = sigma * P * L
    crit[str(L)] = (sigma * P / ((1 - P) * (1 - x))) if x < 1 else "none (supercritical for every h_f)"
res["a_critical_hf_P0.05"] = crit
print("critical h_f vs latency (attempts):", crit)


def offspring_latency_mc(P, h_f, sigma, L, n, rng):
    """Attempt-level: Poisson(rate 1) attempts; failed attempt flagged w.p. h_f; after the first flag
    the replica keeps attempting for L time units (Poisson(L) further attempts), then is removed."""
    kids = np.zeros(n, int)
    for i in range(n):
        c = 0
        while True:  # unflagged phase
            if rng.random() < P:
                c += rng.random() < sigma
            elif rng.random() < h_f:
                break
        na = rng.poisson(L)
        c += rng.binomial(na, P * sigma)
        kids[i] = c
    return kids


def gw_surv_mc(sampler, reps, cap=300, gens=200):
    esc = 0
    for _ in range(reps):
        N = 1
        for _g in range(gens):
            if N == 0 or N >= cap:
                break
            N = int(sampler(N).sum())
        esc += N > 0
    return esc / reps


lat_rows = []
for (h_f, L) in [(0.04, 0), (0.04, 10), (0.08, 10), (0.2, 15), (0.1, 5)]:
    Rl = R_latency(P, h_f, sigma, L)
    u = surv_latency(P, h_f, sigma, L)
    kids = offspring_latency_mc(P, h_f, sigma, L, 20000, rng)
    pool = kids
    sv = gw_surv_mc(lambda N: rng.choice(pool, N), 3000)
    lat_rows.append(dict(P=P, h_f=h_f, L=L, R=Rl, R_mc=float(kids.mean()), R_mc_se=float(kids.std() / np.sqrt(20000)),
                         surv_exact=u, surv_mc=sv, surv_mc_se=float(np.sqrt(max(sv * (1 - sv), 1e-12) / 3000)),
                         R_no_latency=sigma * P / ((1 - P) * h_f)))
    print("latency", lat_rows[-1])
res["a_latency_validation"] = lat_rows


def calendar_sim(P, h_f, sigma, L, Nstop=3000, tmax=4000, rng=None, sweep=None):
    """Event-driven calendar-time simulation (attempt rate 1). Each replica: attempt times ~ Poisson
    process; flagged at first flagged failure; removed L time units later. Returns (times, alive counts,
    first-flag time, escaped flag). sweep=(T_G, eps): at first_flag+T_G each live replica removed w.p. eps."""
    # events: (time, type, id) type 0=attempt, 1=removal
    ev = []; alive = {}; nid = 0; first_flag = None; swept = False
    def birth(t):
        nonlocal nid
        alive[nid] = False  # flagged?
        heapq.heappush(ev, (t + rng.exponential(1.0), 0, nid)); nid += 1
    birth(0.0)
    ts = [0.0]; ns = [1]
    while ev:
        t, typ, i = heapq.heappop(ev)
        if sweep is not None and first_flag is not None and not swept and t >= first_flag + sweep[0]:
            swept = True
            for j in list(alive):
                if rng.random() < sweep[1]:
                    del alive[j]
        if t > tmax:
            break
        if i not in alive:
            continue
        if typ == 1:
            del alive[i]
        else:
            if rng.random() < P:
                if rng.random() < sigma:
                    birth(t)
            elif (not alive[i]) and rng.random() < h_f:
                alive[i] = True
                if first_flag is None:
                    first_flag = t
                heapq.heappush(ev, (t + L, 1, i))
            heapq.heappush(ev, (t + rng.exponential(1.0), 0, i))
        ts.append(t); ns.append(len(alive))
        if len(alive) == 0 or len(alive) >= Nstop:
            break
    return np.array(ts), np.array(ns), first_flag, len(alive) >= Nstop


# Malthusian rate check
mal = []
for (h_f, L) in [(0.02, 0), (0.02, 10), (0.08, 10)]:
    r_th = malthus_latency(P, h_f, sigma, L)
    slopes = []
    for rep in range(200):
        ts, ns, ff, esc = calendar_sim(P, h_f, sigma, L, Nstop=3000, rng=rng)
        if esc:
            a = np.searchsorted(ns >= 100, True); b = len(ns) - 1
            if b > a + 10:
                slopes.append((np.log(ns[b]) - np.log(ns[a])) / (ts[b] - ts[a]))
        if len(slopes) >= 30:
            break
    mal.append(dict(h_f=h_f, L=L, r_theory=r_th, r_sim_mean=float(np.mean(slopes)),
                    r_sim_se=float(np.std(slopes) / np.sqrt(len(slopes))), n=len(slopes),
                    doubling_time_attempts=float(np.log(2) / r_th)))
    print("malthus", mal[-1])
res["a_malthus"] = mal

# ------------------------------------------------------------------ (b) global response lag
gl = []
h_f, L = 0.02, 0
u0 = surv_latency(P, h_f, sigma, L)
for eps in [0.9, 0.99]:
    for TG in [25, 50, 100, 200, 400]:
        n_esc = 0; reps = 600
        for _ in range(reps):
            ts, ns, ff, esc = calendar_sim(P, h_f, sigma, L, Nstop=2000, tmax=TG + 3000, rng=rng, sweep=(TG, eps))
            n_esc += esc
        gl.append(dict(eps=eps, T_G=TG, escape=n_esc / reps, se=float(np.sqrt(n_esc / reps * (1 - n_esc / reps) / reps))))
        print("global", gl[-1])
res["b_global_response"] = dict(P=P, h_f=h_f, L=L, surv_no_response=u0,
                                r=malthus_latency(P, h_f, sigma, L), rows=gl)

# ------------------------------------------------------------------ (c) adaptive h / learning p
def R_adaptive(p0, pinf, rho, h0, beta, sigma=1.0, delta=0.0, T=200000, budget=None):
    t = np.arange(T)
    p = pinf - (pinf - p0) * rho ** t
    h = np.minimum(1.0, h0 + beta * t)
    surv_step = 1 - (1 - p) * h
    S = np.concatenate([[1.0], np.cumprod(surv_step)[:-1]])
    m_inf = float(np.sum(sigma * p * S))
    # optimal finite budget (mean): m(a) = sum_{t<a} sigma p_t S_t + (1-delta) S_a
    Sa = np.concatenate([[1.0], np.cumprod(surv_step)])
    ma = np.concatenate([[0.0], np.cumsum(sigma * p * S)]) + (1 - delta) * Sa
    a_star = int(np.argmax(ma[:20001]))
    return m_inf, a_star, float(ma[a_star])


ad = []
for (p0, pinf, rho, h0, beta) in [(0.02, 0.02, 1, 0.01, 0), (0.02, 0.02, 1, 0.01, 1e-4), (0.02, 0.02, 1, 0.01, 1e-3),
                                   (0.02, 0.02, 1, 0.0, 1e-4), (0.02, 0.02, 1, 0.0, 1e-3),
                                   (0.02, 0.1, 0.99, 0.05, 0), (0.02, 0.1, 0.99, 0.05, 1e-3),
                                   (0.02, 0.1, 0.999, 0.05, 0)]:
    m, a_star, m_star = R_adaptive(p0, pinf, rho, h0, beta, delta=0.1)
    ad.append(dict(p0=p0, p_inf=pinf, rho=rho, h0=h0, beta=beta, R_unlimited=m, mean_optimal_budget=a_star,
                   m_at_optimal_budget=m_star, R_const_model=p0 / ((1 - p0) * h0) if h0 > 0 else float("inf")))
    print("adaptive", ad[-1])
# MC check of one adaptive case
p0, h0, beta = 0.02, 0.0, 1e-3
kids = []
for _ in range(20000):
    t = 0; c = 0
    while True:
        if rng.random() < p0:
            c += 1
        elif rng.random() < min(1, h0 + beta * t):
            break
        t += 1
    kids.append(c)
res["c_adaptive"] = dict(rows=ad, mc_check=dict(p0=p0, h0=h0, beta=beta, mc_mean=float(np.mean(kids)),
                                                se=float(np.std(kids) / np.sqrt(len(kids))),
                                                exact=R_adaptive(p0, p0, 1, h0, beta)[0]),
                         note="h_t = min(1, h0 + beta t) at try t; p_t = p_inf - (p_inf - p0) rho^t; delta=0.1 for budget")
print("adaptive MC", res["c_adaptive"]["mc_check"])

# ------------------------------------------------------------------ (d) staggered providers
def lyap(b, s, rho, M):
    if M == "inf":
        return np.log(b * (1 - s * rho))
    x = np.arange(M + 1); w = binom.pmf(x, M, rho)
    return float(np.sum(w * np.log(np.maximum(b * (1 - s * x / M), 1e-300))))


def surv_env(b, s, rho, M, reps=5000, gens=300, cap=5000):
    esc = 0
    for _ in range(reps):
        N = 1
        for _g in range(gens):
            if N == 0 or N >= cap:
                break
            X = rng.binomial(M, rho) if M != "inf" else None
            d = s * X / M if M != "inf" else s * rho
            N = rng.poisson(b * N * (1 - d))  # Poisson offspring with conditional mean b(1-d)
        esc += N > 0
    return esc / reps


st = []
b = 1.5
for (s, rho) in [(0.75, 0.4), (0.5, 0.6)]:
    for M in [1, 2, 4, 8, 16, 64, "inf"]:
        lam = lyap(b, s, rho, M)
        sv = surv_env(b, s, rho, M)
        st.append(dict(b=b, sweep_strength=s, rho=rho, mean_detection=s * rho, M=str(M), lyapunov=lam,
                       surv_300gen=sv, se=float(np.sqrt(sv * (1 - sv) / 5000))))
        print("providers", st[-1])
res["d_providers"] = st

# ------------------------------------------------------------------ (e) density dependence
dd = []
m0 = 1.5
from model import extinction_prob
qpois = 1.0
for _ in range(10000):
    qpois = np.exp(-m0 * (1 - qpois))
for Kc in [30, 100, 300, 1000, 3000]:
    Nstar = Kc * (1 - 1 / m0); target = max(2, int(Nstar / 2))
    reps = 4000; est = 0; plateau = []
    for _ in range(reps):
        N = 1
        for g in range(2000):
            if N == 0 or N >= target:
                break
            N = rng.poisson(m0 * N * max(0.0, 1 - N / Kc))
        est += N >= target
    dd.append(dict(K=Kc, N_star=Nstar, target=target, establish=est / reps, se=float(np.sqrt(est / reps * (1 - est / reps) / reps)),
                   gw_survival=1 - qpois))
    print("density", dd[-1])
res["e_density"] = dd

json.dump(res, open(f"{OUT}/e11_robustness.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
hs = np.logspace(-3, 0, 200)
for L in [0, 5, 10, 20]:
    axs[0].plot(hs, [surv_latency(P, h, 1.0, L) for h in hs], label=f"latency = {L} attempts")
axs[0].set_xscale("log"); axs[0].set_xlabel("h_f"); axs[0].set_ylabel("survival prob. (1 founder)")
axs[0].set_title("E11a: defender latency, P = 0.05"); axs[0].legend(fontsize=7)
for eps in [0.9, 0.99]:
    rr = [g for g in gl if g["eps"] == eps]
    axs[1].errorbar([g["T_G"] for g in rr], [g["escape"] for g in rr], yerr=[1.96 * g["se"] for g in rr],
                    marker="o", label=f"sweep removes {eps:.0%}")
axs[1].axhline(u0, color="k", ls=":", lw=0.8, label="no global response")
axs[1].set_xscale("log"); axs[1].set_xlabel("global response lag T_G (attempt durations)")
axs[1].set_ylabel("escape probability"); axs[1].set_title("E11b: response lag"); axs[1].legend(fontsize=7)
for (s, rho) in [(0.75, 0.4), (0.5, 0.6)]:
    rr = [x for x in st if x["sweep_strength"] == s]
    labs = [x["M"] for x in rr]
    axs[2].plot(range(len(rr)), [x["lyapunov"] for x in rr], marker="o", label=f"sweep strength {s}")
    axs[2].set_xticks(range(len(rr))); axs[2].set_xticklabels(labs)
axs[2].axhline(0, color="k", lw=0.7); axs[2].set_xlabel("number of independent providers M")
axs[2].set_ylabel("E ln m (Lyapunov exponent)"); axs[2].set_title("E11d: staggered sweeps, mean detection 0.3")
axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_e11_robustness.png", dpi=150)
print("done", time.time() - t0)
