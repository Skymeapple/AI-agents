"""Minimal ReLU sparse autoencoder in numpy (no decoder bias, unit-norm decoder rows).
Loss = mean_b ||x - relu(x We + be) Wd||^2 + lam * mean_b sum_i z_i   (same objective as theory.py)."""
import numpy as np


class SAE:
    def __init__(self, d, m, rng, init_dirs=None):
        Wd = rng.standard_normal((m, d)) if init_dirs is None else init_dirs.copy()
        Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
        self.Wd = Wd
        self.We = Wd.T.copy()
        self.be = np.zeros(m)

    def encode(self, X):
        return np.maximum(X @ self.We + self.be, 0.0)

    def loss(self, X, lam):
        Z = self.encode(X)
        E = Z @ self.Wd - X
        return np.mean(np.sum(E * E, 1)) + lam * np.mean(np.sum(Z, 1))


def train(sae, sampler, lam, steps=5000, batch=512, lr=3e-3, rng=None, b1=0.9, b2=0.999):
    params = ['We', 'be', 'Wd']
    m_ = {p: np.zeros_like(getattr(sae, p)) for p in params}
    v_ = {p: np.zeros_like(getattr(sae, p)) for p in params}
    for t in range(1, steps + 1):
        X = sampler(batch, rng)
        pre = X @ sae.We + sae.be
        Z = np.maximum(pre, 0.0)
        E = Z @ sae.Wd - X
        B = X.shape[0]
        gXh = 2 * E / B
        g = {}
        g['Wd'] = Z.T @ gXh
        gZ = gXh @ sae.Wd.T + lam / B
        gpre = gZ * (pre > 0)
        g['We'] = X.T @ gpre
        g['be'] = gpre.sum(0)
        # remove radial component of decoder gradient (unit-norm constraint)
        g['Wd'] -= np.sum(g['Wd'] * sae.Wd, 1, keepdims=True) * sae.Wd
        cur_lr = lr * (1.0 if t < 0.8 * steps else (steps - t) / (0.2 * steps) + 1e-3)
        for p in params:
            m_[p] = b1 * m_[p] + (1 - b1) * g[p]
            v_[p] = b2 * v_[p] + (1 - b2) * g[p] ** 2
            mh = m_[p] / (1 - b1 ** t); vh = v_[p] / (1 - b2 ** t)
            setattr(sae, p, getattr(sae, p) - cur_lr * mh / (np.sqrt(vh) + 1e-8))
        sae.Wd /= np.linalg.norm(sae.Wd, axis=1, keepdims=True)
    return sae
