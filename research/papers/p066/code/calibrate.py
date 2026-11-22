"""
ABC-rejection calibration of the two-state model to the six arm-level moments
reported by Bastani et al. (2025, PNAS 122(26) e2422633122):
  practice (assisted) score: control 0.28, GPT Base 0.28+0.137, GPT Tutor 0.28+0.361
  unassisted exam score:     control 0.054/0.17 = 0.318 (implied), Base 0.318-0.054, Tutor ~ control (|diff| 0.004)
Experiment: four sessions; each session = AI-assisted practice then unassisted exam.

Run: python3 calibrate.py   (approx. 2-4 min CPU) -> results/abc_accepted.npz, results/calibration_summary.json
"""
import json, os, time
import numpy as np
from model2 import period

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(20261122)

CTRL_P = 0.28
CTRL_E = 0.054 / 0.17
TARGET = np.array([CTRL_P, CTRL_E, CTRL_P + 0.137, CTRL_E - 0.054, CTRL_P + 0.361, CTRL_E])
TOL = np.array([0.015, 0.012, 0.015, 0.012, 0.015, 0.015])
NAMES = ["ctrl_practice", "ctrl_exam", "base_practice", "base_exam", "tutor_practice", "tutor_exam"]

PRIOR = dict(m_s=(0.15, 0.75), m_g=(0.35, 1.0), eta=(0.0, 0.8), lam=(0.0, 0.15),
             kappa=(0.0, 0.6), rho=(0.0, 0.6), sigma=(0.0, 1.0), qA=(0.2, 1.0),
             qT=(0.2, 1.0), c=(0.0, 1.0), Omega=(0.0, 5.0), beta=(0.4, 1.0))
KEYS = list(PRIOR)
NSTUD, NSESS, SD = 300, 4, 0.8
# common random numbers for students
Z1 = rng.standard_normal(NSTUD); Z2 = rng.standard_normal(NSTUD)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def students(th):
    s0 = sigmoid(logit(th["m_s"])[:, None] + SD * Z1[None, :])
    g0 = sigmoid(logit(np.minimum(th["m_g"], 0.98))[:, None] + SD * Z2[None, :])
    return s0, g0


def simulate_arm(th, arm):
    s0, g0 = students(th)
    P = dict(eta=th["eta"][:, None], lam=th["lam"][:, None], kappa=th["kappa"][:, None],
             rho=th["rho"][:, None], c=th["c"][:, None], Omega=th["Omega"][:, None],
             beta=th["beta"][:, None])
    if arm == "ctrl":
        P.update(q=0.0, sigma=0.0); forced = 0.0
    elif arm == "base":
        P.update(q=th["qA"][:, None], sigma=0.0); forced = None
    else:
        P.update(q=th["qT"][:, None], sigma=th["sigma"][:, None]); forced = None
    g = g0.copy(); prac = 0; exam = 0; ash = 0
    out_s = []
    for k in range(NSESS):
        s = s0.copy()  # new topic each session, same prior mastery level
        a, pr, s2, g2 = period(s, g, P, forced)
        prac = prac + pr.mean(1) / NSESS
        exam = exam + (g2 * s2).mean(1) / NSESS
        ash = ash + a.mean(1) / NSESS
        g = g2
        out_s.append((a, s2, g2))
    return prac, exam, ash, out_s, (s0, g0)


def draw(n):
    return {k: rng.uniform(*PRIOR[k], size=n) for k in KEYS}


def moments(th):
    pc, ec, _, _, _ = simulate_arm(th, "ctrl")
    pb, eb, ab, _, _ = simulate_arm(th, "base")
    pt, et, at, _, _ = simulate_arm(th, "tutor")
    return np.stack([pc, ec, pb, eb, pt, et], 1), ab, at


def evaluate(th, batch=4000):
    n = len(th[KEYS[0]]); Ms = []; As = []
    for i in range(0, n, batch):
        sub = {k: v[i:i + batch] for k, v in th.items()}
        M, ab, at = moments(sub)
        Ms.append(M); As.append(np.stack([ab, at], 1))
    return np.concatenate(Ms), np.concatenate(As)


def in_prior(th):
    ok = np.ones(len(th[KEYS[0]]), bool)
    for k in KEYS:
        ok &= (th[k] >= PRIOR[k][0]) & (th[k] <= PRIOR[k][1])
    return ok


def main(n0=40_000, rounds=8, n_round=15_000, elite=400):
    """Exploration of the set of parameter vectors consistent with the six moments
    (max |model-target|/tol <= 1). Round 0 samples the uniform prior box; later rounds
    perturb the current elite set (Gaussian kernel with the elite covariance, clipped to the box). Because the
    perturbation rounds are not importance-weighted, accepted draws are treated as an
    identified SET (ranges reported), not as a Bayesian posterior."""
    t0 = time.time()
    th = draw(n0)
    M, AA = evaluate(th)
    pool = {k: th[k] for k in KEYS}; poolM = M; poolA = AA
    log = []
    for r in range(rounds + 1):
        d = (np.abs(poolM - TARGET) / TOL).max(1)
        log.append(dict(round=r, n_evaluated=int(len(d)), n_accepted=int((d <= 1).sum()),
                        best=float(d.min())))
        print(log[-1], flush=True)
        if r == rounds:
            break
        idx = np.argsort(d)[:elite]
        X = np.stack([pool[k][idx] for k in KEYS], 1)
        cov = 1.0 * np.cov(X.T)
        par = X[rng.integers(0, elite, n_round)]
        Xn = par + rng.multivariate_normal(np.zeros(len(KEYS)), cov, n_round)
        thn = {k: np.clip(Xn[:, j], *PRIOR[k]) for j, k in enumerate(KEYS)}  # clip into prior box
        Mn, An = evaluate(thn)
        pool = {k: np.concatenate([pool[k], thn[k]]) for k in KEYS}
        poolM = np.concatenate([poolM, Mn]); poolA = np.concatenate([poolA, An])
    d = (np.abs(poolM - TARGET) / TOL).max(1)
    ok = d <= 1.0
    acc = {k: pool[k][ok] for k in KEYS}; accM = poolM[ok]; accA = poolA[ok]
    total = len(d); n_acc = int(ok.sum())
    np.savez(os.path.join(RES, "abc_accepted.npz"), M=accM, A=accA, **acc)
    # prior-only acceptance for reference
    prior_rate = float((d[:n0] <= 1).mean())

    # decomposition of the Base exam penalty: skill vs persistence channel, per accepted draw
    th = {k: acc[k] for k in KEYS}
    _, _, _, outc, (s0, g0) = simulate_arm(th, "ctrl")
    _, _, _, outb, _ = simulate_arm(th, "base")
    # mean over sessions of s and g after practice
    sc = np.mean([o[1] for o in outc], 0); gc = np.mean([o[2] for o in outc], 0)
    sb = np.mean([o[1] for o in outb], 0); gb = np.mean([o[2] for o in outb], 0)
    total_gap = (gb * sb).mean(1) - (gc * sc).mean(1)
    skill_ch = (gc * sb).mean(1) - (gc * sc).mean(1)   # change s only
    pers_ch = (gb * sc).mean(1) - (gc * sc).mean(1)    # change g only
    # heterogeneity: Base-minus-control exam effect by prior-skill tercile (session-1 exam)
    a1b, s1b, g1b = outb[0]; a1c, s1c, g1c = outc[0]
    eff = g1b * s1b - g1c * s1c  # (n_acc, NSTUD), same students (CRN)
    order = np.argsort(Z1); terc = np.array_split(order, 3)
    het = [float(np.median(eff[:, idx].mean(1))) for idx in terc]
    het_lo = [float(np.percentile(eff[:, idx].mean(1), 10)) for idx in terc]
    het_hi = [float(np.percentile(eff[:, idx].mean(1), 90)) for idx in terc]
    off_terc = [float(np.median(a1b[:, idx].mean(1))) for idx in terc]
    # tutor heterogeneity
    _, _, _, outt, _ = simulate_arm(th, "tutor")
    a1t, s1t, g1t = outt[0]
    efft = g1t * s1t - g1c * s1c
    het_t = [float(np.median(efft[:, idx].mean(1))) for idx in terc]

    def q(x):
        return dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))

    summ = dict(
        n_draws=int(total), n_accepted=int(n_acc), prior_round_acceptance_rate=prior_rate, search_log=log,
        targets=dict(zip(NAMES, TARGET.tolist())), tolerances=dict(zip(NAMES, TOL.tolist())),
        prior=PRIOR, posterior={k: q(acc[k]) for k in KEYS},
        prior_vs_posterior_width={k: float((np.percentile(acc[k], 90) - np.percentile(acc[k], 10)) /
                                           (0.8 * (PRIOR[k][1] - PRIOR[k][0]))) for k in KEYS},
        fitted_moments={NAMES[i]: q(accM[:, i]) for i in range(6)},
        offload_share_base=q(accA[:, 0]), offload_share_tutor=q(accA[:, 1]),
        base_exam_gap=q(total_gap), skill_channel=q(skill_ch), persistence_channel=q(pers_ch),
        share_persistence_channel=q(pers_ch / np.where(np.abs(total_gap) > 1e-9, total_gap, np.nan)),
        frac_draws_persistence_channel_dominant=float(np.mean(np.abs(pers_ch) > np.abs(skill_ch))),
        het_base_effect_by_prior_tercile=dict(low=het[0], mid=het[1], high=het[2],
                                              p10=het_lo, p90=het_hi),
        het_base_offload_share_by_tercile=off_terc,
        het_tutor_effect_by_prior_tercile=dict(low=het_t[0], mid=het_t[1], high=het_t[2]),
        frac_draws_harm_largest_low_tercile=float(np.mean(np.argmin(np.stack(
            [eff[:, idx].mean(1) for idx in terc], 1), 1) == 0)),
        frac_draws_harm_largest_mid_tercile=float(np.mean(np.argmin(np.stack(
            [eff[:, idx].mean(1) for idx in terc], 1), 1) == 1)),
        runtime_sec=time.time() - t0)
    with open(os.path.join(RES, "calibration_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        main(n0=8000, rounds=2, n_round=4000, elite=100)
    else:
        main()
