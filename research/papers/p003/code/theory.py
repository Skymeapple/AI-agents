"""Core analytical objects for the tilted-verifier theory of RL progress.

A response x drawn from a base policy pi0 has a gold label y(x) in {0,1} and a
verifier acceptance probability f(x) in [0,1] (expected binary reward).
The KL-regularised optimum with coefficient lam is pi_t(x) ∝ pi0(x) exp(t f(x)),
t = 1/lam; the same family is traced by the replicator / exponential-weights
flow d pi/dt = pi (f - E_pi f).  The true accuracy along the family obeys

    logit A(t) = logit A(0) + K_c(t) - K_w(t),
    K_S(t) = log E_{pi0}[exp(t f) | y = S]   (class-conditional CGFs).
"""
import numpy as np
from scipy.special import logsumexp, expit, logit


def cgf(f, w, t):
    """log E_w[exp(t f)] for weights w (need not be normalised). t can be an array."""
    f = np.asarray(f, float); w = np.asarray(w, float)
    t = np.atleast_1d(t)
    lw = np.log(w / w.sum())
    return logsumexp(lw[None, :] + t[:, None] * f[None, :], axis=1)


def tilted_accuracy(f, y, w, t):
    """True accuracy of pi_t ∝ w exp(t f) via the CGF identity."""
    f = np.asarray(f, float); y = np.asarray(y, bool); w = np.asarray(w, float)
    A0 = w[y].sum() / w.sum()
    Kc = cgf(f[y], w[y], t)
    Kw = cgf(f[~y], w[~y], t)
    return expit(logit(A0) + Kc - Kw)


def tilted_accuracy_direct(f, y, w, t):
    """Same quantity computed directly (used to check the identity numerically)."""
    t = np.atleast_1d(t)
    lw = np.log(w) + t[:, None] * np.asarray(f)[None, :]
    lw -= logsumexp(lw, axis=1, keepdims=True)
    return np.exp(lw) @ np.asarray(y, float)


def tilted_kl(f, w, t):
    """KL(pi_t || pi0) = t E_t[f] - K(t)."""
    t = np.atleast_1d(t)
    p = np.asarray(w, float) / np.sum(w)
    lw = np.log(p)[None, :] + t[:, None] * np.asarray(f)[None, :]
    Z = logsumexp(lw, axis=1)
    pt = np.exp(lw - Z[:, None])
    return t * (pt @ np.asarray(f)) - Z


def onpolicy_youden(f, y, w, t):
    """J_t = E_t[f | correct] - E_t[f | incorrect] (= K_c'(t) - K_w'(t))."""
    y = np.asarray(y, bool); f = np.asarray(f, float); w = np.asarray(w, float)
    out = []
    for tt in np.atleast_1d(t):
        lw = np.log(w) + tt * f
        pc = np.exp(lw[y] - logsumexp(lw[y])); pw = np.exp(lw[~y] - logsumexp(lw[~y]))
        out.append(pc @ f[y] - pw @ f[~y])
    return np.array(out)


# ---------- three-mode closed form (Corollary 2) ----------

def three_mode_peak(c0, w0, h0, alpha, beta, eta):
    """Correct mode (mass c0, acceptance alpha), honest-wrong mode (w0, beta),
    exploit mode (h0, eta) with beta < alpha < eta.  Returns (t_star, A_star, kappa).
    t_star = 0 if accuracy is non-increasing from the start."""
    assert beta < alpha < eta
    kappa = (alpha - beta) / (eta - beta)
    arg = w0 * (alpha - beta) / (h0 * (eta - alpha))
    if arg <= 1:
        return 0.0, c0 / (c0 + w0 + h0), kappa
    t_star = np.log(arg) / (eta - beta)
    A_star = 1.0 / (1.0 + (w0 / c0) * (eta - beta) / (eta - alpha)
                    * (h0 * (eta - alpha) / (w0 * (alpha - beta))) ** kappa)
    return t_star, A_star, kappa


def three_mode_traj(c0, w0, h0, alpha, beta, eta, t):
    f = np.array([alpha, beta, eta]); y = np.array([1, 0, 0], bool); w = np.array([c0, w0, h0])
    return tilted_accuracy(f, y, w, t)
