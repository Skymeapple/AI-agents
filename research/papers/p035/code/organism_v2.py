"""Revision-1 extensions of the model-organism simulation (see organism.py).

New relative to organism.py
  * policy parameters b0 (saturation), detector ('additive' | 'interaction'),
    g shape ('monotone' | 'tent' = deliberate A2 violation)
  * six rewriters: perfect, partial (under-shooting), overshoot_mild, overshoot_full,
    task_sanitise, task_intensify
  * lower bound reported without slack, per rewriter
  * an A3 check computed from benign traffic only (probe projection)
  * stronger white-box baselines: steering to a risky-adjusted target at layers 1 and 2,
    residual mean matching (post- and pre-activation), and model-relative dose-response
    extrapolation (belief curve measured on the probe axis, extrapolated)
  * within-configuration diagnostics
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from organism import (NT, NE, ND, NN, NH, iT, iE, iD, iN, iH, DIM, STYLE,
                      make_world, sample)

E_PAIRS = [(iE[2 * k], iE[2 * k + 1]) for k in range(NE // 2)]
D_PAIRS = [(iD[2 * k], iD[2 * k + 1]) for k in range(ND // 2)]
H_PAIRS = [(iH[0], iH[1]), (iH[2], iH[3]), (iH[0], iH[3]), (iH[1], iH[2])]
EH_PAIRS = [(iE[k], iH[k % NH]) for k in range(NE)]  # cross-group interactions


def make_world_v2(rng, rho, kappa, b0, detector='additive', gshape='monotone', subtle=False):
    w = make_world(rng, rho=rho, subtle=subtle, kappa=kappa)
    w['b0'] = b0; w['detector'] = detector; w['gshape'] = gshape
    if detector == 'interaction':
        pe = np.r_[np.zeros(NT), w['p_eval']]; pdp = np.r_[np.zeros(NT), w['p_dep']]
        wts = {}
        for name, pairs, sgn in [('E', E_PAIRS, 1), ('D', D_PAIRS, -1), ('EH', EH_PAIRS, 1), ('H', H_PAIRS, 1)]:
            wts[name] = (pairs, sgn * rng.uniform(0.5, 1.0, len(pairs)))
        def gap(name):
            pairs, ww = wts[name]
            return sum(wi * (pe[i] * pe[j] - pdp[i] * pdp[j]) for (i, j), wi in zip(pairs, ww))
        g_style = gap('E') + gap('D'); g_H = gap('H') + gap('EH')
        # EH terms vanish once style is rewritten only partly; count them as non-rewritable
        a_s = (1 - rho) * 6.0 / g_style; a_h = rho * 6.0 / g_H if rho > 0 else 0.0
        w['int_terms'] = [(wts['E'][0], a_s * wts['E'][1]), (wts['D'][0], a_s * wts['D'][1]),
                          (wts['H'][0], a_h * wts['H'][1]), (wts['EH'][0], a_h * wts['EH'][1])]
        def zfun(X):
            z = np.zeros(len(X))
            for pairs, ww in w['int_terms']:
                for (i, j), wi in zip(pairs, ww):
                    z += wi * X[:, i] * X[:, j]
            return z
        def ez(p):
            pp = np.r_[np.zeros(NT), p]; s = 0.0
            for pairs, ww in w['int_terms']:
                for (i, j), wi in zip(pairs, ww):
                    s += wi * pp[i] * pp[j]
            return s
        w['zcentre'] = 0.5 * (ez(w['p_eval']) + ez(w['p_dep']))
        w['zfun'] = zfun
    return w


def zc_of(w, X):
    if w['detector'] == 'interaction':
        return w['zfun'](X) - w['zcentre']
    z = X @ w['wz']
    return z - 0.5 * (w['p_eval'] + w['p_dep']) @ w['wz'][NT:]


def policy_v2(w, X, r):
    zc = zc_of(w, X)
    base = w['b0'] + X @ w['bt']
    if w['gshape'] == 'tent':
        # deliberate A2 violation: unsafe rate peaks at an intermediate belief m=-1
        # (between deployment level ~-3 and the midpoint), falls on both sides.
        m = -1.0
        logit = np.where(zc >= m, base - w['kappa'] * zc / 3.0,
                         base - w['kappa'] * m / 3.0 - w['kappa'] * (m - zc) / 3.0)
    else:
        logit = base - w['kappa'] * zc / 3.0
    return np.where(r, 1 / (1 + np.exp(-logit)), 0.02)


def train_v2(rng, w, n=30000, seed=0):
    Xs, ys = [], []
    for ctx in ['eval', 'dep']:
        X, r = sample(rng, w, n // 2, ctx, 'mix')
        ys.append(rng.random(len(X)) < policy_v2(w, X, r)); Xs.append(X)
    m = MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-4, max_iter=60, random_state=seed)
    m.fit(np.vstack(Xs), np.concatenate(ys))
    return m


def pre1(m, X):
    return X @ m.coefs_[0] + m.intercepts_[0]


def h1(m, X):
    return np.maximum(0, pre1(m, X))


def h2_from_h1(m, H1):
    return np.maximum(0, H1 @ m.coefs_[1] + m.intercepts_[1])


def out_from_h2(m, H2):
    return 1 / (1 + np.exp(-(H2 @ m.coefs_[2] + m.intercepts_[2]).ravel()))


def out_from_h1(m, H1):
    return out_from_h2(m, h2_from_h1(m, H1))


def behave(m, X):
    return m.predict_proba(X)[:, 1]


# ---------------- rewriters ----------------
def rw_perfect(rng, w, X, feats=STYLE):
    Xr = X.copy(); f = np.asarray(feats)
    if len(f):
        Xr[:, f] = rng.random((len(X), len(f))) < w['p_dep'][f - NT]
    return Xr


def rw_partial(rng, w, X, frac=0.6):
    """Under-shooting rewriter: each style feature of each item rewritten w.p. frac."""
    Xr = rw_perfect(rng, w, X)
    keep = rng.random((len(X), len(STYLE))) > frac
    Xr[:, STYLE] = np.where(keep, X[:, STYLE], Xr[:, STYLE])
    return Xr


def rw_overshoot(rng, w, X, lam):
    """Over-shooting rewriter: style marginals pushed past deployment by lam*(p_dep-p_eval)."""
    p = np.clip(w['p_dep'] + lam * (w['p_dep'] - w['p_eval']), 0, 1)[STYLE - NT]
    Xr = X.copy(); Xr[:, STYLE] = rng.random((len(X), len(STYLE))) < p
    return Xr


def rw_task(rng, w, X, mode, p=0.3):
    """Perfect style rewrite that also alters the task (first 10 task features)."""
    Xr = rw_perfect(rng, w, X)
    T = Xr[:, iT[:10]]
    if mode == 'sanitise':   # rewriter softens the request
        T = np.where(rng.random(T.shape) < p, 0, T)
    else:                    # rewriter makes the request more explicit
        T = np.where(rng.random(T.shape) < p, 1, T)
    Xr[:, iT[:10]] = T
    return Xr


REWRITERS = {
    'perfect': lambda rng, w, X: rw_perfect(rng, w, X),
    'partial': lambda rng, w, X: rw_partial(rng, w, X),
    'overshoot_mild': lambda rng, w, X: rw_overshoot(rng, w, X, 0.5),
    'overshoot_full': lambda rng, w, X: rw_overshoot(rng, w, X, 3.0),
    'task_sanitise': lambda rng, w, X: rw_task(rng, w, X, 'sanitise'),
    'task_intensify': lambda rng, w, X: rw_task(rng, w, X, 'intensify'),
}


def _probe(Ha, Hb):
    y = np.r_[np.ones(len(Ha)), np.zeros(len(Hb))]
    pr = LogisticRegression(C=1.0, max_iter=3000).fit(np.vstack([Ha, Hb]), y)
    u = pr.coef_[0]; return u / (np.linalg.norm(u) + 1e-12)


def _resid_auc(A, B):
    """held-out AUC of a fresh logistic classifier separating A from B."""
    ha, hb = len(A) // 2, len(B) // 2
    c = LogisticRegression(max_iter=3000).fit(np.vstack([A[:ha], B[:hb]]), np.r_[np.ones(ha), np.zeros(hb)])
    s = c.decision_function(np.vstack([A[ha:], B[hb:]]))
    return float(roc_auc_score(np.r_[np.ones(len(A) - ha), np.zeros(len(B) - hb)], s))


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def run_v2(seed, rho=0.0, H_visible=True, kappa=4.0, b0=1.5, detector='additive', gshape='monotone',
           n_eval=2000, n_dep=4000):
    rng = np.random.default_rng(seed)
    w = make_world_v2(rng, rho, kappa, b0, detector, gshape)
    m = train_v2(rng, w, seed=seed)
    Xdr, _ = sample(rng, w, 20000, 'dep', 1)
    bd = behave(m, Xdr)
    res = dict(seed=seed, rho=rho, H_visible=H_visible, kappa=kappa, b0=b0, detector=detector,
               gshape=gshape, theta_dep=float(bd.mean()), theta_dep_mcse=float(bd.std() / np.sqrt(len(bd))))
    Xer, _ = sample(rng, w, n_eval, 'eval', 1)
    Xeb, _ = sample(rng, w, n_eval, 'eval', 0)
    Xdb, _ = sample(rng, w, n_dep, 'dep', 0)
    be = behave(m, Xer); res['naive'] = float(be.mean())
    res['theta_eval_policy'] = float(policy_v2(w, Xer, np.ones(len(Xer), bool)).mean())
    res['theta_dep_policy'] = float(policy_v2(w, Xdr, np.ones(len(Xdr), bool)).mean())
    vis = np.concatenate([STYLE, iH]) if H_visible else STYLE
    # probes (benign eval vs benign deployment), layers 1 and 2
    He1, Hd1 = h1(m, Xeb), h1(m, Xdb)
    u1 = _probe(He1, Hd1)
    He2, Hd2 = h2_from_h1(m, He1), h2_from_h1(m, Hd1)
    u2 = _probe(He2, Hd2)
    # ---- every rewriter: estimate, one-sided LB, A3 check from benign traffic only
    for name, fn in REWRITERS.items():
        Xr = fn(rng, w, Xer); br = behave(m, Xr)
        se = br.std(ddof=1) / np.sqrt(len(br))
        res[f'{name}_est'] = float(br.mean()); res[f'{name}_lb'] = float(br.mean() - 1.645 * se)
        Xbr = fn(rng, w, Xeb)
        pb, pdp = h1(m, Xbr) @ u1, Hd1 @ u1
        t = (pb.mean() - pdp.mean()) / np.sqrt(pb.var(ddof=1) / len(pb) + pdp.var(ddof=1) / len(pdp))
        res[f'{name}_a3_t'] = float(t)          # t < -3: rewritten looks MORE deployment-like
        pb2, pd2 = h2_from_h1(m, h1(m, Xbr)) @ u2, Hd2 @ u2
        res[f'{name}_a3_t_l2'] = float((pb2.mean() - pd2.mean()) /
                                       np.sqrt(pb2.var(ddof=1) / len(pb2) + pd2.var(ddof=1) / len(pd2)))
    # ---- white-box estimators built on the perfect rewrite
    Xr = rw_perfect(rng, w, Xer); Xbr = rw_perfect(rng, w, Xeb)
    full = behave(m, Xr); res['full'] = float(full.mean())
    Hr1 = h1(m, Xr); Hbr1 = h1(m, Xbr); Her1 = h1(m, Xer)
    # (a) old-style steering: benign target, layer 1
    tgt_b = (Hd1 @ u1).mean()
    res['rs_l1_benign'] = float(out_from_h1(m, np.maximum(0, Hr1 + np.outer(tgt_b - Hr1 @ u1, u1))).mean())
    # (b) risky-adjusted target: deployment benign mean + (risky - benign) offset in eval context
    off1 = (Her1 @ u1).mean() - (He1 @ u1).mean()
    tgt_r = tgt_b + off1
    res['rs_l1_risky'] = float(out_from_h1(m, np.maximum(0, Hr1 + np.outer(tgt_r - Hr1 @ u1, u1))).mean())
    Hr2 = h2_from_h1(m, Hr1); Her2 = h2_from_h1(m, Her1)
    tgt2 = (Hd2 @ u2).mean() + (Her2 @ u2).mean() - (He2 @ u2).mean()
    res['rs_l2_risky'] = float(out_from_h2(m, np.maximum(0, Hr2 + np.outer(tgt2 - Hr2 @ u2, u2))).mean())
    # (c) residual mean matching (all directions) estimated on benign traffic
    dmu = Hd1.mean(0) - Hbr1.mean(0)
    res['meanmatch_l1'] = float(out_from_h1(m, np.maximum(0, Hr1 + dmu)).mean())
    dmu_pre = pre1(m, Xdb).mean(0) - pre1(m, Xbr).mean(0)
    res['meanmatch_pre'] = float(out_from_h1(m, np.maximum(0, pre1(m, Xr) + dmu_pre)).mean())
    # (d) model-relative dose-response extrapolation (MRDE): graded rewrites, belief measured
    #     on the probe axis; logit(behaviour) regressed on mean projection; extrapolate.
    rng_order = rng.permutation(STYLE)
    P, B = [], []
    for k in [0, 6, 12, 18, 24, 30, 36]:
        Xk = rw_perfect(rng, w, Xer, rng_order[:k])
        P.append((h1(m, Xk) @ u1).mean()); B.append(behave(m, Xk).mean())
    P, B = np.array(P), np.array(B)
    sl, ic = np.polyfit(P[3:], lg(B[3:]), 1)
    res['mrde'] = float(1 / (1 + np.exp(-(ic + sl * tgt_r))))
    # (e) steering dose curve on fully rewritten items (vary magnitude, read at target)
    res['probe_gap_risky'] = float((Hr1 @ u1).mean() - tgt_r)
    # ---- diagnostics
    res['D1_aud_auc'] = _resid_auc(Xbr[:, vis], Xdb[:, vis])
    res['D2_probe_auc_l1'] = _resid_auc(Hbr1, Hd1)
    res['D2_probe_auc_l2'] = _resid_auc(h2_from_h1(m, Hbr1), Hd2)
    # projection gap along the eval-probe axis (signed), layer 1 and 2, in pooled-SD units
    p1, q1 = Hbr1 @ u1, Hd1 @ u1
    res['D6_proj_gap_l1'] = float((p1.mean() - q1.mean()) / np.sqrt(0.5 * (p1.var() + q1.var())))
    p2, q2 = h2_from_h1(m, Hbr1) @ u2, Hd2 @ u2
    res['D6_proj_gap_l2'] = float((p2.mean() - q2.mean()) / np.sqrt(0.5 * (p2.var() + q2.var())))
    return res
