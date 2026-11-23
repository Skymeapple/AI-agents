"""Step-level Monte Carlo simulator of the verify-checkpoint-rollback model.

Independent of theory.py: it simulates individual steps, individual errors with
sampled detectabilities, individual verifier calls, rollbacks and checkpoints.
"""
import numpy as np


class DetSampler:
    """Samples per-error detectability d."""

    def __init__(self, kind, rho=0.0, r=None, a=None, b=None, values=None):
        self.kind, self.rho, self.r, self.a, self.b = kind, rho, r, a, b
        self.values = None if values is None else np.asarray(values, float)

    def sample(self, rng, n):
        if self.kind == "empirical":
            return self.values[rng.integers(0, len(self.values), n)]
        blind = rng.random(n) < self.rho
        if self.kind == "point":
            d = np.full(n, self.r)
        elif self.kind == "beta":
            d = rng.beta(self.a, self.b, n)
        else:
            raise ValueError(self.kind)
        d[blind] = 0.0
        return d


def run_once(rng, N, T, p0, beta, det, m, f, V, C, R, K=0.0, policy="fresh",
             adaptive=False, max_time=1e7):
    """Simulate one long-running agent episode.

    Returns (makespan, final_clean, n_rollbacks).
    adaptive=True uses a history-dependent schedule (T halves and m triples
    after a flagged segment, T doubles after two consecutive passes) to test
    Theorem 2's claim that no verifier-driven policy beats the blind-spot floor.
    """
    t = 0.0
    progress = 0
    committed_dirty = False
    age_at_ckpt = 0
    n_rb = 0
    Tcur, mcur, streak = T, m, 0
    while progress < N:
        L = min(Tcur, N - progress)
        if policy == "fresh":
            ages = np.arange(L)
            t += K
        else:
            ages = age_at_ckpt + np.arange(L)
        pa = np.minimum(1.0, p0 * (1.0 + beta * ages))
        err = rng.random(L) < pa
        ne = int(err.sum())
        d = det.sample(rng, ne) if ne else np.zeros(0)
        t += L
        flagged = False
        for _ in range(mcur):
            t += V
            if rng.random() < f or (ne and np.any(rng.random(ne) < d)):
                flagged = True  # all m calls are run (and charged)
        if flagged:
            t += R
            n_rb += 1
            if adaptive:
                Tcur, mcur, streak = max(1, Tcur // 2), min(9, 3 * m), 0
        else:
            t += C
            progress += L
            age_at_ckpt += L
            committed_dirty |= ne > 0
            if adaptive:
                streak += 1
                if streak >= 2:
                    Tcur, mcur, streak = min(4 * T, 2 * Tcur), m, 0
        if t > max_time:
            return np.inf, False, n_rb
    return t, (not committed_dirty), n_rb


def run_many(n_runs, seed, **kw):
    rng = np.random.default_rng(seed)
    out = np.array([run_once(rng, **kw)[:2] for _ in range(n_runs)], float)
    return out[:, 0], out[:, 1]
