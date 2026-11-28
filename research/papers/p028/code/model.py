"""Closed-form quantities of the latent-defect model (LDM).

Notation (see paper Sec. 3):
  p  : probability that one generation attempt for a module is correct
  c  : coverage of the module-level verifier on *detectable* defects
  b  : blind-spot mass: fraction of defects the verifier cannot detect
       (b = 0 for an oracle independent of the generator; b > 0 when the
       verifier shares the generator's blind spots, e.g. self-written tests)
  K  : maximum number of attempts per module
"""
import numpy as np
from scipy.special import betaln


def escape(c, b):
    """Probability that a defective attempt passes the verifier."""
    return b + (1.0 - b) * (1.0 - c)


def latent_defect_rate(p, c, b, K=np.inf):
    """q = P(accepted module is defective | module accepted within K attempts).

    Attempt outcomes: correct-pass (p), defective-pass ((1-p)s), defective-reject
    ((1-p)(1-s)).  Conditioning on acceptance within K attempts the ratio of
    defective-pass to correct-pass is the same on every attempt, so q does not
    depend on K (Lemma 1).
    """
    s = escape(c, b)
    return (1 - p) * s / (p + (1 - p) * s)


def accept_prob(p, c, b, K):
    """P(module accepted within K attempts)."""
    s = escape(c, b)
    a = p + (1 - p) * s
    return 1 - (1 - a) ** K


def project_success(N, q):
    return (1 - q) ** N


def horizon(q, level=0.5):
    """Largest number of modules completed with probability >= level."""
    return np.log(level) / np.log1p(-np.asarray(q, dtype=float))


def ceiling_q(p, b):
    """lim_{c->1} q : the floor on latent-defect rate set by the blind spot."""
    return (1 - p) * b / (p + (1 - p) * b)


def mixture_success(N, alpha, beta):
    """E[(1-q)^N] for q ~ Beta(alpha, beta)  (Proposition 3)."""
    N = np.asarray(N, dtype=float)
    return np.exp(betaln(alpha, beta + N) - betaln(alpha, beta))


def diverse_blindspot(m, a, bb):
    """Eckhardt-Lee style blind spot of m verifiers sharing a per-defect miss
    intensity theta ~ Beta(a, bb): b_m = E[theta^m]."""
    return np.exp(betaln(a + m, bb) - betaln(a, bb))
