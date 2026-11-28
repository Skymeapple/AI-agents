"""Checkpoint statistics used by the revision's control experiments (round 2).

Given penultimate representations of training / test points, the linear readout (W, b)
and the logits, compute at one checkpoint:
  * the readout certificate fraction C (Prop. 3, full representation space),
  * C_vis: the same certificate applied after projecting representations onto the
    readout-visible subspace U = span{w_c - w_1} (sound, sharper, invariant to
    readout-invisible directions; see paper Prop. 3' ),
  * margin-threshold controls  M_q = frac of test points whose true-label logit margin
    exceeds the q-quantile of the training margins at the same checkpoint (q = .1, .5),
    and absolute thresholds M_abs(tau),
  * test / train cross-entropy, NC1 = tr(S_W S_B^+)/K on train and on test features,
  * how well the test margin alone predicts per-point certification (AUROC).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from hull import readout_certificate


def margins(logits, y):
    z = logits.copy()
    zy = z[np.arange(len(y)), y]
    z[np.arange(len(y)), y] = -np.inf
    return zy - z.max(1)


def xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return float((lse - z[np.arange(len(y)), y]).mean())


def nc1(F, y, means, K):
    """tr(S_W S_B^+)/K with S_W from deviations of F from the given class means."""
    mu_g = means.mean(0)
    Mb = means - mu_g
    SB = Mb.T @ Mb / K
    dev = F - means[y]
    SW = dev.T @ dev / len(F)
    return float(np.trace(SW @ np.linalg.pinv(SB, rcond=1e-6)) / K)


def visible_basis(W):
    D = W[:, 1:] - W[:, [0]]
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, s > 1e-8 * s.max()]


def checkpoint_stats(Ftr, ytr, Fte, yte, W, b, logits_tr, logits_te, K, taus=(2.0, 5.0, 10.0), per_point=False):
    out = {}
    mtr = margins(logits_tr, ytr)
    mte = margins(logits_te, yte)
    out["test_acc"] = float((logits_te.argmax(1) == yte).mean())
    out["train_acc"] = float((logits_tr.argmax(1) == ytr).mean())
    out["test_loss"] = xent(logits_te, yte)
    out["train_loss"] = xent(logits_tr, ytr)
    for q in (0.1, 0.5):
        out[f"M_q{int(q*100)}"] = float((mte > np.quantile(mtr, q)).mean())
    for t in taus:
        out[f"M_abs{t:g}"] = float((mte > t).mean())
    out["train_margin_median"] = float(np.median(mtr))
    out["test_margin_median"] = float(np.median(mte))
    means = np.stack([Ftr[ytr == c].mean(0) if (ytr == c).any() else np.zeros(Ftr.shape[1]) for c in range(K)])
    out["NC1_train"] = nc1(Ftr, ytr, means, K)
    out["NC1_test"] = nc1(Fte, yte, means, K)
    U = visible_basis(W)
    Wv = U.T @ W
    Ftr_v, Fte_v = Ftr @ U, Fte @ U
    cert, cert_v, ratio = [], [], []
    idx = {c: np.where(ytr == c)[0] for c in range(K)}
    for i in range(len(yte)):
        c = yte[i]
        ok, _, m = readout_certificate(Fte[i], c, Ftr[idx[c]], W, b)
        okv, _, _ = readout_certificate(Fte_v[i], c, Ftr_v[idx[c]], Wv, b)
        cert.append(ok); cert_v.append(okv)
    cert, cert_v = np.array(cert), np.array(cert_v)
    out["C"] = float(cert.mean())
    out["C_vis"] = float(cert_v.mean())
    corr = logits_te.argmax(1) == yte
    for name, cc in (("C", cert), ("C_vis", cert_v)):
        sel = corr
        if sel.sum() > 5 and 0 < cc[sel].sum() < sel.sum():
            out[f"auroc_margin_predicts_{name}"] = float(roc_auc_score(cc[sel], mte[sel]))
        else:
            out[f"auroc_margin_predicts_{name}"] = None
    if per_point:
        out["pp"] = {"margin": np.round(mte, 4).tolist(), "cert": cert.astype(int).tolist(),
                     "cert_vis": cert_v.astype(int).tolist(), "correct": corr.astype(int).tolist(),
                     "train_margin_q": np.round(np.quantile(mtr, [0.01, 0.05, 0.1, 0.5]), 4).tolist()}
    return out
