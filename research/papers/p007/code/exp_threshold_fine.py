"""Remark 4 revisited (round 2): pooling and spectral detection, as an SNR model.

In the reduction of Sec. 4.2 the per-modality estimate is W_m + sqrt(v) Z_m with
v = sigma^2 / n_eff.  Below n = d no orthogonal design exists, so here n_eff is
ONLY an inverse noise variance (an "SNR parameter"), not a sample size.

We sweep n_eff on a fine log grid (40 points, 40 reps), at full capacity, for
  case 'private': k_s=0, k_p=8   case 'shared': k_s=8, k_p=0   (d=64,q=16,M=4,s=1)
and compare the Monte Carlo loss curves with the asymptotic (d,q -> inf at fixed
ratio) prediction for hard-truncated SVD with oracle rank, built from the
Benaych-Georges--Nadakuditi / Gavish--Donoho spiked-matrix formulas:
  Y = X + Z/sqrt(N) (m x N, beta=m/N<=1), spike x > beta^{1/4}:
  y^2 = (x + 1/x)(x + beta/x),  c_u^2 = (x^4-beta)/(x^4+beta x^2),
  c_v^2 = (x^4-beta)/(x^4+x^2); keeping the component changes the error by
  y^2 - 2 x y c_u c_v (kept iff negative, oracle rank).
Shift statistic: for loss fractions f in {0.9, 0.75, 0.5}, the n_eff at which
each estimator's mean loss first falls below f*energy (log-interpolated); the
shift is the ratio specialist/unified ('shared') or unified/specialist
('private').  95% CIs by bootstrap over repetitions.
Run: OMP_NUM_THREADS=1 python3 exp_threshold_fine.py   (~4 min)"""
import json, os
import numpy as np
from common import make_task

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SCALE = int(os.environ.get("TF_SCALE", "1"))  # 1: d=64,q=16 (paper); 4: d=256,q=64 (finite-size check)
d, q, M, k = 64 * SCALE, 16 * SCALE, 4, 8
NGRID = np.geomspace(6, 400, 40) * SCALE
REPS, NBOOT = (40, 1000) if SCALE == 1 else (12, 1000)
rng = np.random.default_rng(11)


def spike_loss(x, beta):
    """Asymptotic normalized per-spike error (signal x^2 if dropped)."""
    if x <= beta ** 0.25:
        return x ** 2
    y = np.sqrt((x + 1 / x) * (x + beta / x))
    cu = np.sqrt((x ** 4 - beta) / (x ** 4 + beta * x ** 2))
    cv = np.sqrt((x ** 4 - beta) / (x ** 4 + x ** 2))
    return x ** 2 + min(0.0, y ** 2 - 2 * x * y * cu * cv)


def asym(case, n):
    """Per-modality asymptotic loss for unified and specialist (s=sigma=1)."""
    scale_s = d / n          # specialist: q x d, N=d, beta=q/d
    xs = np.sqrt(n / d)      # spike x for energy 1
    Ls = k * spike_loss(xs, q / d) * scale_s
    Nb = max(M * q, d); mb = min(M * q, d); beta_u = mb / Nb; scale_u = Nb / n
    if case == "shared":
        xu = np.sqrt(M * n / Nb)
        Lu = k * spike_loss(xu, beta_u) * scale_u / M
    else:
        xu = np.sqrt(n / Nb)
        Lu = M * k * spike_loss(xu, beta_u) * scale_u / M
    return Lu, Ls


def oracle_losses(Ws, v):
    What = [W + np.sqrt(v) * rng.standard_normal(W.shape) for W in Ws]
    St = np.concatenate(What, 0); Wt = np.concatenate(Ws, 0)
    u, s, vt = np.linalg.svd(St, full_matrices=False)
    errs = [np.sum(Wt ** 2)]
    approx = np.zeros_like(St)
    for j in range(min(40, len(s))):  # rank <= 40 >= 32 signal spikes
        approx += s[j] * np.outer(u[:, j], vt[j]); errs.append(np.sum((approx - Wt) ** 2))
    lu = min(errs) / M
    ls = 0.0
    for Wh, W in zip(What, Ws):
        u, s, vt = np.linalg.svd(Wh, full_matrices=False)
        errs = [np.sum(W ** 2)]; approx = np.zeros_like(Wh)
        for j in range(len(s)):
            approx += s[j] * np.outer(u[:, j], vt[j]); errs.append(np.sum((approx - W) ** 2))
        ls += min(errs)
    return lu, ls / M


def cross_n(curve, level):
    c = np.asarray(curve); idx = np.where(c < level)[0]
    if len(idx) == 0 or idx[0] == 0:
        return np.nan
    i = idx[0]; a, b = np.log(NGRID[i - 1]), np.log(NGRID[i])
    t = (c[i - 1] - level) / (c[i - 1] - c[i])
    return float(np.exp(a + t * (b - a)))


res = dict(d=d, q=q, M=M, k=k, n_grid=NGRID.tolist(), reps=REPS, cases={})
for case, ks, kp in [("private", 0, 8), ("shared", 8, 0)]:
    U = np.zeros((REPS, len(NGRID))); S = np.zeros_like(U)
    for j, n in enumerate(NGRID):
        for rep in range(REPS):
            Ws = make_task(d, q, ks, kp, M, rng=rng.integers(1 << 31))
            U[rep, j], S[rep, j] = oracle_losses(Ws, 1.0 / n)
    A = np.array([asym(case, n) for n in NGRID])
    shifts = {}
    for f in [0.9, 0.75, 0.5]:
        lev = f * k
        def shift(Uc, Sc):
            nu, ns = cross_n(Uc, lev), cross_n(Sc, lev)
            return ns / nu if case == "shared" else nu / ns
        boot = []
        for b in range(NBOOT):
            ii = rng.integers(0, REPS, REPS)
            boot.append(shift(U[ii].mean(0), S[ii].mean(0)))
        shifts[str(f)] = dict(sim=shift(U.mean(0), S.mean(0)),
                              ci95=np.nanpercentile(boot, [2.5, 97.5]).tolist(),
                              asymptotic=shift(A[:, 0], A[:, 1]),
                              n_unified_sim=cross_n(U.mean(0), lev), n_spec_sim=cross_n(S.mean(0), lev),
                              n_unified_asym=cross_n(A[:, 0], lev), n_spec_asym=cross_n(A[:, 1], lev))
    res["cases"][case] = dict(sim_u=U.mean(0).tolist(), sim_s=S.mean(0).tolist(),
                              asym_u=A[:, 0].tolist(), asym_s=A[:, 1].tolist(), shifts=shifts,
                              bbp_n_spec=float(np.sqrt(q * d)),
                              bbp_n_unified=float(np.sqrt(M * q * d) / (M if case == "shared" else 1)))
    print(case, json.dumps(shifts, indent=0), flush=True)
json.dump(res, open(os.path.join(OUT, "threshold_fine_results.json" if SCALE == 1 else f"threshold_fine_results_d{d}.json"), "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, case in zip(axs, ["shared", "private"]):
    c = res["cases"][case]
    ax.plot(NGRID, np.array(c["sim_u"]) / k, "o", ms=3, color="C0", label="unified (MC)")
    ax.plot(NGRID, np.array(c["asym_u"]) / k, "-", color="C0", lw=1, label="unified (asymptotic)")
    ax.plot(NGRID, np.array(c["sim_s"]) / k, "s", ms=3, color="C1", label="specialist (MC)")
    ax.plot(NGRID, np.array(c["asym_s"]) / k, "-", color="C1", lw=1, label="specialist (asymptotic)")
    ax.set_xscale("log"); ax.set_xlabel("SNR parameter n_eff = sigma^2 / noise variance")
    ax.set_title(f"{case} features (M={M}, d={d}, q={q})", fontsize=9)
axs[0].set_ylabel("loss / signal energy"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figG_threshold_fine.png" if SCALE == 1 else f"figG_threshold_fine_d{d}.png"), dpi=150)
