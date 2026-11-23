"""Model-organism simulation for estimating deployment behaviour of an
evaluation-aware model from evaluation-time data.

World: binary bag-of-feature "prompts" with groups
  T  task features (determine risky vs benign task)
  E  evaluation-style cues (frequent in eval, rare in deployment)   [rewritable]
  D  deployment-style cues (frequent in deployment, rare in eval)   [rewritable]
  N  neutral style features                                          [rewritable]
  H  structural eval cues the auditor cannot rewrite (e.g. scenario
     implausibility); either visible to the auditor or latent.
A model organism (2x32 ReLU MLP) is trained on labels from a conditional
policy: on risky tasks it acts unsafely with probability
sigmoid(b0 + bt.x_T - kappa * z(x)), where z(x) is eval evidence.
The organism *learns its own* eval detector; the auditor never sees it.
Target: unsafe rate on deployment-style risky prompts (never shown to the
auditor). Auditor data: eval set (risky + benign items) and benign
deployment traffic.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NT, NE, ND, NN, NH = 20, 10, 10, 16, 4
iT = np.arange(0, NT)
iE = np.arange(NT, NT + NE)
iD = np.arange(NT + NE, NT + NE + ND)
iN = np.arange(NT + NE + ND, NT + NE + ND + NN)
iH = np.arange(NT + NE + ND + NN, NT + NE + ND + NN + NH)
DIM = NT + NE + ND + NN + NH
STYLE = np.concatenate([iE, iD, iN])  # rewritable


def make_world(rng, rho=0.0, subtle=False, kappa=4.0):
    w = {}
    # style marginals
    pE_eval = rng.uniform(0.5, 0.8, NE); pE_dep = rng.uniform(0.05, 0.2, NE)
    pD_eval = rng.uniform(0.05, 0.2, ND); pD_dep = rng.uniform(0.5, 0.8, ND)
    if subtle:
        # half the eval cues differ only slightly in frequency ("subtle cues")
        k = NE // 2
        pE_eval[:k] = rng.uniform(0.30, 0.38, k); pE_dep[:k] = rng.uniform(0.22, 0.28, k)
    w['p_eval'] = np.concatenate([pE_eval, pD_eval, np.full(NN, 0.3), np.full(NH, 0.8)])
    w['p_dep'] = np.concatenate([pE_dep, pD_dep, np.full(NN, 0.3), np.full(NH, 0.1)])
    # organism's eval-evidence weights
    wE = rng.uniform(0.5, 1.0, NE); wD = rng.uniform(0.5, 1.0, ND)
    if subtle:
        wE[:NE // 2] *= 6.0  # the organism leans on the subtle cues
    wH = np.ones(NH)
    # scale so that the mean eval-dep gap of z is split rho : (1-rho)
    gap_style = wE @ (pE_eval - pE_dep) + wD @ (pD_dep - pD_eval)
    gap_H = np.array([wH.sum() * (0.8 - 0.1)])
    tot = 6.0
    a_style = (1 - rho) * tot / gap_style
    a_H = rho * tot / gap_H[0]
    w['wz'] = np.zeros(DIM)
    w['wz'][iE] = a_style * wE; w['wz'][iD] = -a_style * wD; w['wz'][iH] = a_H * wH
    w['kappa'] = kappa
    w['bt'] = np.zeros(DIM); w['bt'][iT[:10]] = rng.normal(0, 0.3, 10)
    w['b0'] = 1.5
    return w


def sample(rng, w, n, ctx, risky):
    """ctx: 'eval' or 'dep'; risky: 0/1 or 'mix'."""
    X = np.zeros((n, DIM))
    r = rng.random(n) < 0.5 if risky == 'mix' else np.full(n, bool(risky))
    pT = np.where(r[:, None], 0.6, 0.1) * np.ones((n, NT))
    pT[:, 10:] = 0.3
    X[:, iT] = rng.random((n, NT)) < pT
    p = w['p_eval'] if ctx == 'eval' else w['p_dep']
    X[:, NT:] = rng.random((n, DIM - NT)) < p
    return X, r


def true_policy(w, X, r):
    z = X @ w['wz']
    # centre z at the midpoint between contexts
    zc = z - 0.5 * (w['p_eval'] + w['p_dep']) @ w['wz'][NT:]
    pu = 1 / (1 + np.exp(-(w['b0'] + X @ w['bt'] - w['kappa'] * zc / 3.0)))
    return np.where(r, pu, 0.02)


def train_organism(rng, w, n=30000, seed=0):
    Xs, ys = [], []
    for ctx in ['eval', 'dep']:
        X, r = sample(rng, w, n // 2, ctx, 'mix')
        y = rng.random(len(X)) < true_policy(w, X, r)
        Xs.append(X); ys.append(y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    m = MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-4, max_iter=60,
                      random_state=seed, early_stopping=False)
    m.fit(X, y)
    return m


def hidden1(m, X):
    return np.maximum(0, X @ m.coefs_[0] + m.intercepts_[0])


def from_hidden1(m, H1):
    H2 = np.maximum(0, H1 @ m.coefs_[1] + m.intercepts_[1])
    o = H2 @ m.coefs_[2] + m.intercepts_[2]
    return 1 / (1 + np.exp(-o.ravel()))


def behave(m, X):
    return m.predict_proba(X)[:, 1]


def rewrite(rng, w, X, feats):
    """Resample the listed (rewritable) features from deployment marginals."""
    Xr = X.copy()
    if len(feats):
        Xr[:, feats] = rng.random((len(X), len(feats))) < w['p_dep'][np.asarray(feats) - NT]
    return Xr


def run_one(seed, rho=0.0, H_visible=True, subtle=False, kappa=4.0, n_eval=2000, n_dep=4000):
    rng = np.random.default_rng(seed)
    w = make_world(rng, rho=rho, subtle=subtle, kappa=kappa)
    m = train_organism(rng, w, seed=seed)
    # --- ground truth (hidden from auditor)
    Xdr, rdr = sample(rng, w, 20000, 'dep', 1)
    theta_dep = behave(m, Xdr).mean()
    # --- auditor data
    Xer, _ = sample(rng, w, n_eval, 'eval', 1)        # risky eval items
    Xeb, _ = sample(rng, w, n_eval, 'eval', 0)        # benign eval items
    Xdb, _ = sample(rng, w, n_dep, 'dep', 0)          # benign deployment traffic
    vis = np.concatenate([STYLE, iH]) if H_visible else STYLE
    clf = LogisticRegression(C=1.0, max_iter=2000)
    Xc = np.vstack([Xeb[:, vis], Xdb[:, vis]])
    yc = np.r_[np.ones(len(Xeb)), np.zeros(len(Xdb))]
    clf.fit(Xc, yc)
    coef = np.zeros(DIM); coef[vis] = clf.coef_[0]
    score = lambda X: X[:, vis] @ clf.coef_[0] + clf.intercept_[0]
    res = dict(seed=seed, rho=rho, H_visible=H_visible, subtle=subtle, kappa=kappa,
               theta_dep=theta_dep)
    b_eval = behave(m, Xer)
    res['naive'] = b_eval.mean()
    res['theta_eval_true_policy'] = true_policy(w, Xer, np.ones(len(Xer), bool)).mean()
    # importance weighting (self-normalised) with density ratio from classifier
    a = score(Xer)
    lw = -a; lw -= lw.max(); wt = np.exp(lw); wt /= wt.sum()
    res['iw'] = float(wt @ b_eval)
    res['iw_ess'] = float(1.0 / (wt ** 2).sum())
    # graded rewrites: rank rewritable features by |coef| (eval-indicative first)
    order = STYLE[np.argsort(-np.abs(coef[STYLE]))]
    levels = list(range(0, len(order) + 1, 4))
    if levels[-1] != len(order): levels.append(len(order))
    a_dep = score(Xdb).mean()
    A, B, SE = [], [], []
    for k in levels:
        Xk = rewrite(rng, w, Xer, order[:k])
        bk = behave(m, Xk)
        A.append(score(Xk).mean()); B.append(bk.mean()); SE.append(bk.std(ddof=1) / np.sqrt(len(bk)))
    A, B = np.array(A), np.array(B)
    res['levels_SE'] = SE; res['levels_k'] = levels
    res['levels_A'] = A.tolist(); res['levels_B'] = B.tolist(); res['a_dep'] = a_dep
    res['full_rewrite'] = B[-1]
    # dose-response extrapolation in logit space, last half of levels
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    sel = slice(len(A) // 2, None)
    if np.ptp(A[sel]) > 1e-6:
        sl, ic = np.polyfit(A[sel], lg(B[sel]), 1)
        res['rde'] = float(1 / (1 + np.exp(-(ic + sl * a_dep))))
    else:
        res['rde'] = B[-1]
    # bounds: monotone lower bound (= max over rewrite levels), Lipschitz upper bound
    # one-sided 95% lower confidence bound at the full rewrite (minimal belief)
    res['lb_mono'] = float(B[-1] - 1.645 * SE[-1])
    slopes = np.abs(np.diff(B) / np.where(np.abs(np.diff(A)) > 1e-9, np.diff(A), np.nan))
    L = np.nanmax(slopes[-3:]) if np.isfinite(slopes[-3:]).any() else 0.0
    gapA = max(A[-1] - a_dep, 0.0)
    res['ub_lip'] = float(min(1.0, B[-1] + 2.0 * L * gapA))  # safety factor 2
    res['gapA'] = float(gapA)
    # residual distinguishability diagnostic: auditor-visible features,
    # fully rewritten eval (benign) vs deployment (benign)
    Xebr = rewrite(rng, w, Xeb, STYLE)
    half = len(Xdb) // 2
    c2 = LogisticRegression(max_iter=2000).fit(
        np.vstack([Xebr[:1000, vis], Xdb[:half, vis]]), np.r_[np.ones(1000), np.zeros(half)])
    s2 = c2.decision_function(np.vstack([Xebr[1000:, vis], Xdb[half:, vis]]))
    res['resid_auc_auditor'] = float(roc_auc_score(np.r_[np.ones(len(Xebr) - 1000), np.zeros(len(Xdb) - half)], s2))
    # white-box: linear probe on organism layer-1 activations (benign eval vs benign dep)
    He, Hd = hidden1(m, Xeb), hidden1(m, Xdb)
    pr = LogisticRegression(C=1.0, max_iter=3000).fit(np.vstack([He, Hd]), yc)
    u = pr.coef_[0] / np.linalg.norm(pr.coef_[0])
    # model-relative residual: probe AUC on rewritten eval vs dep
    Hr = hidden1(m, Xebr)
    pr2 = LogisticRegression(max_iter=3000).fit(np.vstack([Hr[:1000], Hd[:half]]), np.r_[np.ones(1000), np.zeros(half)])
    s3 = pr2.decision_function(np.vstack([Hr[1000:], Hd[half:]]))
    res['resid_auc_probe'] = float(roc_auc_score(np.r_[np.ones(len(Hr) - 1000), np.zeros(len(Hd) - half)], s3))
    # steering: shift risky-eval activations along u to the deployment mean projection
    Hx = hidden1(m, Xer)
    target = (Hd @ u).mean()
    Hs = np.maximum(0, Hx + np.outer(target - Hx @ u, u))
    res['steer'] = float(from_hidden1(m, Hs).mean())
    Hxr = hidden1(m, rewrite(rng, w, Xer, STYLE))
    Hsr = np.maximum(0, Hxr + np.outer(target - Hxr @ u, u))
    res['rewrite_steer'] = float(from_hidden1(m, Hsr).mean())
    # mean-difference steering (difference of class means), common in practice
    dmu = Hd.mean(0) - He.mean(0)
    res['steer_meandiff'] = float(from_hidden1(m, np.maximum(0, Hx + dmu)).mean())
    return res
