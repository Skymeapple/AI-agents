"""Activation-based lie detectors.

Supervised (need lie labels from ONE trigger, the instructed-lie trigger A):
  - LR probe       : logistic regression honest vs instructed-lie, per layer
  - DiffMeans      : w = mu_lie - mu_honest, per layer
Label-free w.r.t. deception (need honest data only):
  - Mahalanobis    : distance from honest activation distribution, per layer
  - Mismatch (ours): per-layer linear 'belief' decoders trained on HONEST data
                     to predict the fact value; score = max over admissible
                     layers of -log p_belief(model's own output token).
                     Admissible layers are chosen without any lie data:
                     honest held-out decoding accuracy >= 0.9.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from world import V


class Std:
    def fit(self, X):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def fit_lr_probe(H_hon, H_lie, C=1.0):
    X = np.concatenate([H_hon, H_lie]); y = np.r_[np.zeros(len(H_hon)), np.ones(len(H_lie))]
    s = Std().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000).fit(s(X), y)
    return s, clf


def lr_score(probe, H):
    s, clf = probe
    return clf.decision_function(s(H))


def fit_diffmeans(H_hon, H_lie):
    return H_lie.mean(0) - H_hon.mean(0)


def fit_maha(H_hon, shrink=0.1):
    mu = H_hon.mean(0); C = np.cov(H_hon, rowvar=False)
    C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    return mu, np.linalg.inv(C)


def maha_score(m, H):
    mu, P = m; D = H - mu
    return np.einsum("ij,jk,ik->i", D, P, D)


def fit_belief(H_hon, truth, C=1.0):
    s = Std().fit(H_hon)
    clf = LogisticRegression(C=C, max_iter=3000).fit(s(H_hon), truth)
    return s, clf


def belief_logp(dec, H, tokens):
    """log p_belief(token) with classes possibly missing -> floor."""
    s, clf = dec
    lp = clf.predict_log_proba(s(H))
    full = np.full((len(H), V), -30.0)
    full[:, clf.classes_] = lp
    return full[np.arange(len(H)), tokens]


def auroc(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return roc_auc_score(y, np.r_[neg, pos])


def tpr_at_fpr(neg_cal, pos, fpr=0.01):
    thr = np.quantile(neg_cal, 1 - fpr)
    return float((pos > thr).mean()), thr
