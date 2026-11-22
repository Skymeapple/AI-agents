"""Synthetic superposition data with Zipfian feature frequencies + TopK sparse autoencoder
(numpy, manual gradients, Adam). Used to measure which ground-truth features a dictionary
recovers as a function of training tokens T and dictionary width M.
"""
import numpy as np
import scipy.sparse as sp


class ZipfData:
    """n ground-truth unit feature directions in R^d. Feature i (rank i, 1-indexed) is active
    independently with prob p_i = min(pmax, C * i^-alpha), C chosen so E[L0] = l0.
    Active magnitudes ~ U[0.5, 1.5]."""

    def __init__(self, n, d, alpha, l0, seed=0, pmax=0.5):
        rng = np.random.default_rng(seed)
        F = rng.standard_normal((n, d))
        self.F = F / np.linalg.norm(F, axis=1, keepdims=True)
        r = np.arange(1, n + 1, dtype=float)
        lo, hi = 1e-8, 1e4
        for _ in range(200):  # bisection on C
            C = np.sqrt(lo * hi)
            s = np.minimum(pmax, C * r ** -alpha).sum()
            lo, hi = (C, hi) if s < l0 else (lo, C)
        self.p = np.minimum(pmax, C * r ** -alpha)
        self.n, self.d, self.alpha, self.l0 = n, d, alpha, l0
        self.rng = np.random.default_rng(seed + 12345)

    def sample(self, B):
        """Exact independent-Bernoulli sampling via per-feature binomial counts and
        positions drawn without replacement (sparse construction)."""
        counts = self.rng.binomial(B, self.p)
        feats = np.repeat(np.arange(self.n), counts)
        rows = np.concatenate([self.rng.choice(B, c, replace=False) for c in counts if c > 0]) \
            if counts.sum() > 0 else np.zeros(0, int)
        feats = feats[: len(rows)]
        mag = self.rng.uniform(0.5, 1.5, size=len(rows))
        A = sp.csr_matrix((mag, (rows, feats)), shape=(B, self.n))
        return np.asarray(A @ self.F, dtype=np.float32), A


class TopKSAE:
    def __init__(self, d, M, k, seed=0, lr=1e-3):
        # independent stream from the data generator (seed offset avoids sharing ZipfData's stream)
        rng = np.random.default_rng([seed, 987654321])
        W = rng.standard_normal((M, d)).astype(np.float32)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        self.Wdec = W.copy()               # M x d, unit rows
        self.Wenc = W.T.copy()             # d x M (tied init, Gao et al. 2024)
        self.benc = np.zeros(M, np.float32)
        self.bdec = np.zeros(d, np.float32)
        self.k, self.M, self.lr = k, M, lr
        self.params = ["Wdec", "Wenc", "benc", "bdec"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
        self.last_fired = np.zeros(M, np.int64)  # step index of last activation
        self.fire_count = np.zeros(M, np.int64)

    def encode(self, x):
        pre = (x - self.bdec) @ self.Wenc + self.benc
        idx = np.argpartition(-pre, self.k - 1, axis=1)[:, : self.k]
        z = np.zeros_like(pre)
        rows = np.arange(x.shape[0])[:, None]
        vals = np.maximum(pre[rows, idx], 0)
        z[rows, idx] = vals
        return z, pre

    def step(self, x, aux_coef=1 / 32, dead_steps=200):
        B = x.shape[0]
        z, pre = self.encode(x)
        xhat = z @ self.Wdec + self.bdec
        err = xhat - x
        loss = (err ** 2).sum(1).mean()
        g_xhat = 2 * err / B
        grads = {}
        grads["Wdec"] = z.T @ g_xhat
        g_bdec = g_xhat.sum(0)
        g_pre = (g_xhat @ self.Wdec.T) * (z > 0)
        # AuxK loss (Gao et al. 2024): dead latents reconstruct the residual error
        fired = (z > 0).any(0)
        self.last_fired[fired] = self.t
        self.fire_count += (z > 0).sum(0)
        dead = (self.t - self.last_fired) > dead_steps
        if dead.sum() > 0 and aux_coef > 0:
            kaux = int(min(dead.sum(), 2 * self.k))
            pre_d = np.where(dead[None, :], pre, -np.inf)
            idx = np.argpartition(-pre_d, kaux - 1, axis=1)[:, :kaux]
            rows = np.arange(B)[:, None]
            za = np.zeros_like(pre)
            za[rows, idx] = np.maximum(pre[rows, idx], 0)
            za[~np.isfinite(za)] = 0
            ehat = za @ self.Wdec
            e = -err  # residual x - xhat
            aerr = ehat - e
            scale = aux_coef / max((e ** 2).sum(1).mean(), 1e-8)
            g_eh = 2 * aerr / B * scale
            grads["Wdec"] += za.T @ g_eh
            g_pre += (g_eh @ self.Wdec.T) * (za > 0)
        xc = x - self.bdec
        grads["Wenc"] = xc.T @ g_pre
        grads["benc"] = g_pre.sum(0)
        grads["bdec"] = g_bdec - (g_pre @ self.Wenc.T).sum(0)
        # remove decoder-gradient component parallel to unit rows
        Wd = self.Wdec
        grads["Wdec"] -= (grads["Wdec"] * Wd).sum(1, keepdims=True) * Wd
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            g = grads[p]
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * g * g
            mh = self.m[p] / (1 - b1 ** self.t)
            vh = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - self.lr * mh / (np.sqrt(vh) + eps))
        self.Wdec /= np.linalg.norm(self.Wdec, axis=1, keepdims=True)
        return loss


def recovery(sae, F, thr=0.9):
    """max cosine similarity of each true feature to any decoder row."""
    cs = F @ sae.Wdec.T
    mx = cs.max(1)
    return mx, mx >= thr


def fvu(sae, data, B=8192):
    x, _ = data.sample(B)
    z, _ = sae.encode(x)
    xhat = z @ sae.Wdec + sae.bdec
    return float(((xhat - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
