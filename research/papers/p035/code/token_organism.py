"""A second organism family: a small transformer trained from scratch on token sequences.

Purpose: check that the qualitative results of the MLP organisms are not artefacts of
binary bag-of-features inputs and an MLP detector. Inputs are ordered token sequences;
evaluation evidence is carried by ADJACENT TOKEN PAIRS (order-dependent, non-additive
in a bag-of-tokens view) in a rewritable context prefix, plus a non-rewritable
'scenario' slot. The organism is a 2-layer transformer (d=48) trained on labels from an
evaluation-conditional policy; it learns its own detector. Still a toy; not an LLM.

Sequence layout (length 20):
  [CLS] ctx_1..ctx_8 | task_1..task_8 | scen_1 scen_2 | [END]
Token ids:
  0 CLS, 1 END, 2..11 eval-style, 12..21 deploy-style, 22..33 neutral style,
  34..43 risky task, 44..53 benign task, 54..63 shared task filler,
  64..65 implausible-scenario tokens, 66..69 plausible-scenario tokens
"""
import numpy as np
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

L_CTX, L_TASK, L_SC = 8, 8, 2
V = 70
EV, DP, NU = np.arange(2, 12), np.arange(12, 22), np.arange(22, 34)
RT, BT, FT = np.arange(34, 44), np.arange(44, 54), np.arange(54, 64)
IMPL, PLAUS = np.array([64, 65]), np.arange(66, 70)
CTX_POS = np.arange(1, 1 + L_CTX)


def sample_ctx(rng, n, ctx, lam=0.0):
    """context prefix: each slot is eval-style / deploy-style / neutral.
    eval: (0.55, 0.10, 0.35); deploy: (0.10, 0.55, 0.35). lam>0 pushes past deployment."""
    pe, pd = (0.55, 0.10) if ctx == 'eval' else (0.10, 0.55)
    if ctx == 'over':
        pe, pd = max(0.0, 0.10 - lam * 0.45), min(0.9, 0.55 + lam * 0.45)
    u = rng.random((n, L_CTX))
    tok = np.where(u < pe, rng.choice(EV, (n, L_CTX)),
                   np.where(u < pe + pd, rng.choice(DP, (n, L_CTX)), rng.choice(NU, (n, L_CTX))))
    return tok


def sample(rng, n, ctx, risky):
    X = np.zeros((n, 20), dtype=np.int64); X[:, 0] = 0; X[:, -1] = 1
    X[:, CTX_POS] = sample_ctx(rng, n, ctx)
    r = rng.random(n) < 0.5 if risky == 'mix' else np.full(n, bool(risky))
    u = rng.random((n, L_TASK))
    rt = np.where(u < 0.5, rng.choice(RT, (n, L_TASK)), rng.choice(FT, (n, L_TASK)))
    bt = np.where(u < 0.5, rng.choice(BT, (n, L_TASK)), rng.choice(FT, (n, L_TASK)))
    X[:, 9:17] = np.where(r[:, None], rt, bt)
    pimp = 0.8 if ctx == 'eval' else 0.1
    X[:, 17:19] = np.where(rng.random((n, 2)) < pimp, rng.choice(IMPL, (n, 2)), rng.choice(PLAUS, (n, 2)))
    return X, r


def evidence(X, rho):
    """order-dependent evidence: count adjacent (eval,eval) pairs minus adjacent (dep,dep) pairs in
    the context, plus implausible-scenario tokens. Scaled so mean eval-deploy gap = 6, share rho
    from the scenario slot."""
    c = X[:, CTX_POS]
    isE, isD = np.isin(c, EV), np.isin(c, DP)
    pairE = (isE[:, :-1] & isE[:, 1:]).sum(1); pairD = (isD[:, :-1] & isD[:, 1:]).sum(1)
    s_style = pairE - pairD
    s_sc = np.isin(X[:, 17:19], IMPL).sum(1)
    g_style = 7 * (0.55 ** 2 - 0.10 ** 2) * 2      # E[pairE-pairD] gap between contexts
    g_sc = 2 * (0.8 - 0.1)
    z = (1 - rho) * 6.0 * s_style / g_style + rho * 6.0 * s_sc / g_sc
    # centre between contexts: style means are +-7*(0.55^2-0.1^2) symmetric -> 0;
    # scenario means 1.6 and 0.2 -> midpoint 0.9
    zmid = rho * 6.0 * 0.9 / g_sc
    return z - zmid


def policy(X, r, rho, kappa, b0):
    nrisk = np.isin(X[:, 9:17], RT).sum(1)
    logit = b0 + 0.15 * (nrisk - 4) - kappa * evidence(X, rho) / 3.0
    return np.where(r, 1 / (1 + np.exp(-logit)), 0.02)


class Organism(nn.Module):
    def __init__(self, d=48, nl=2, nh=4):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Parameter(torch.randn(20, d) * 0.1)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, nh, 2 * d, dropout=0.0, batch_first=True)
                                     for _ in range(nl)])
        self.head = nn.Linear(d, 1)

    def resid(self, X, upto):
        h = self.emb(X) + self.pos
        for i, l in enumerate(self.layers):
            if i == upto: break
            h = l(h)
        return h

    def from_resid(self, h, start):
        for l in self.layers[start:]:
            h = l(h)
        return self.head(h[:, 0]).squeeze(-1)

    def forward(self, X):
        return self.from_resid(self.resid(X, 0), 0)


def train(rng, rho, kappa, b0, seed, n=24000, epochs=6):
    torch.manual_seed(seed)
    Xs, ys = [], []
    for ctx in ['eval', 'dep']:
        X, r = sample(rng, n // 2, ctx, 'mix'); Xs.append(X)
        ys.append(rng.random(len(X)) < policy(X, r, rho, kappa, b0))
    X = torch.tensor(np.vstack(Xs)); y = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    m = Organism(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(m(X[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def behave(m, X):
    return torch.sigmoid(m(torch.tensor(X))).numpy()


@torch.no_grad()
def cls_resid(m, X, layer=1):
    return m.resid(torch.tensor(X), layer)[:, 0].numpy()


def rewrite(rng, X, mode='perfect', lam=1.0):
    Xr = X.copy()
    Xr[:, CTX_POS] = sample_ctx(rng, len(X), 'dep' if mode == 'perfect' else 'over', lam)
    return Xr


def resid_auc(A, B):
    ha, hb = len(A) // 2, len(B) // 2
    c = LogisticRegression(max_iter=3000).fit(np.vstack([A[:ha], B[:hb]]), np.r_[np.ones(ha), np.zeros(hb)])
    s = c.decision_function(np.vstack([A[ha:], B[hb:]]))
    return float(roc_auc_score(np.r_[np.ones(len(A) - ha), np.zeros(len(B) - hb)], s))


def run(seed, rho, kappa=1.5, b0=-0.5):
    rng = np.random.default_rng(seed)
    m = train(rng, rho, kappa, b0, seed)
    Xd, _ = sample(rng, 10000, 'dep', 1); th = behave(m, Xd).mean()
    Xe, _ = sample(rng, 2000, 'eval', 1); Xeb, _ = sample(rng, 2000, 'eval', 0); Xdb, _ = sample(rng, 4000, 'dep', 0)
    res = dict(seed=seed, rho=rho, kappa=kappa, b0=b0, theta_dep=float(th), naive=float(behave(m, Xe).mean()),
               theta_dep_policy=float(policy(Xd, np.ones(len(Xd), bool), rho, kappa, b0).mean()))
    for name, mode, lam in [('perfect', 'perfect', 0), ('overshoot_mild', 'over', 0.5), ('overshoot_full', 'over', 1.0)]:
        b = behave(m, rewrite(rng, Xe, mode, lam))
        res[name + '_est'] = float(b.mean()); res[name + '_lb'] = float(b.mean() - 1.645 * b.std(ddof=1) / np.sqrt(len(b)))
        # A3 check on benign traffic: projection on layer-1 CLS probe axis
        Hb, Hd = cls_resid(m, rewrite(rng, Xeb, mode, lam)), cls_resid(m, Xdb)
        if name == 'perfect':
            He = cls_resid(m, Xeb)
            pr = LogisticRegression(max_iter=3000).fit(np.vstack([He, Hd]), np.r_[np.ones(len(He)), np.zeros(len(Hd))])
            u = pr.coef_[0] / np.linalg.norm(pr.coef_[0])
            res['D2_probe_auc'] = resid_auc(Hb, Hd)
        p, q = Hb @ u, Hd @ u
        res[name + '_a3_t'] = float((p.mean() - q.mean()) / np.sqrt(p.var(ddof=1) / len(p) + q.var(ddof=1) / len(q)))
    # auditor realism classifier on token-count features of the rewritable prefix + scenario slot
    def feats(X):
        T = np.c_[X[:, CTX_POS], X[:, 17:19]]
        F = np.zeros((len(X), V)); np.add.at(F, (np.repeat(np.arange(len(X)), T.shape[1]), T.ravel()), 1)
        return F
    Xbr = rewrite(rng, Xeb)
    res['D1_aud_auc'] = resid_auc(feats(Xbr), feats(Xdb))
    return res
