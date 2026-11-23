"""Synthetic response universes: base-policy mass, gold label, verifier acceptance."""
import numpy as np


def make_universe(M=500, A0=0.2, h0_frac=0.01, fc=(8, 2), fw=(1, 9), fh=(30, 1),
                  sigma=1.5, seed=0):
    """M discrete responses.  pi0 is log-normal (heavy-tailed, like LM sequence probs).
    A fraction A0 of base mass is correct.  Among incorrect responses, a set holding
    a fraction h0_frac of the *total* base mass are 'exploits' whose acceptance is
    drawn from Beta(fh) (near 1); other acceptances come from Beta(fc) / Beta(fw)."""
    rng = np.random.default_rng(seed)
    w = np.exp(sigma * rng.standard_normal(M))
    order = rng.permutation(M)
    n_c = max(1, int(round(A0 * M)))
    y = np.zeros(M, bool); y[order[:n_c]] = True
    # rescale masses so correct mass is exactly A0
    w[y] *= A0 / w[y].sum(); w[~y] *= (1 - A0) / w[~y].sum()
    f = np.where(y, rng.beta(*fc, M), rng.beta(*fw, M))
    exploit = np.zeros(M, bool)
    if h0_frac > 0:
        idx = order[n_c:]
        # pick low-mass incorrect responses until reaching h0_frac of mass
        idx = idx[np.argsort(w[idx])]
        cum = np.cumsum(w[idx])
        k = np.searchsorted(cum, h0_frac) + 1
        sel = idx[:k]
        exploit[sel] = True
        w[sel] *= h0_frac / w[sel].sum()
        rest = (~y) & (~exploit)
        w[rest] *= (1 - A0 - h0_frac) / w[rest].sum()
        f[sel] = rng.beta(*fh, len(sel))
    return w / w.sum(), y, f, exploit
