"""Shared helpers: data loading, plurality voting, surrogates, Moran fixation."""
import numpy as np

FAMILY_RULES = [  # (substring, family) -- first match wins; opaque systems -> 'other'
    ("claude", "anthropic"), ("sonnet", "anthropic"), ("opus", "anthropic"),
    ("gemini", "google"), ("flash", "google"), ("gemma", "google"),
    ("gpt", "openai"),
    ("Llama", "meta"),
    ("Mistral", "mistral"), ("Mixtral", "mistral"), ("mathstral", "mistral"),
    ("Qwen", "qwen"), ("Yi-", "yi"), ("deepseek", "deepseek"), ("DeepSeek", "deepseek"),
]


def family(name):
    for s, f in FAMILY_RULES:
        if s in name:
            return f
    return "other"


def load(path="../results/answers.npz", min_opts=3):
    d = np.load(path, allow_pickle=False)
    keep = d["nopt"] >= min_opts
    A = d["A"][keep].astype(np.int16)
    y = d["y"][keep].astype(np.int16)
    return dict(A=A, y=y, cat=d["cat"][keep], nopt=d["nopt"][keep].astype(np.int16),
                models=[str(m) for m in d["models"]])


def plurality(A, rng, K=10):
    """A: (items, agents) int answers in 0..K-1, -1 = abstain. Returns (items,) winner (-1 if all abstain).
    Ties are broken uniformly at random."""
    n = A.shape[0]
    counts = np.zeros((n, K))
    for j in range(A.shape[1]):
        a = A[:, j]
        ok = a >= 0
        counts[np.nonzero(ok)[0], a[ok]] += 1
    noisy = counts + rng.random(counts.shape) * 1e-3
    w = noisy.argmax(1)
    w[counts.sum(1) == 0] = -1
    return w


def weighted_plurality(A, w, rng, K=10):
    n = A.shape[0]
    counts = np.zeros((n, K))
    for j in range(A.shape[1]):
        a = A[:, j]
        ok = a >= 0
        counts[np.nonzero(ok)[0], a[ok]] += w[j]
    noisy = counts + rng.random(counts.shape) * 1e-6
    out = noisy.argmax(1)
    out[np.abs(counts).sum(1) == 0] = -1
    return out


def random_wrong(y, nopt, rng, size):
    """Uniformly random wrong option for each item (vectorised); returns shape (items, size)."""
    r = (rng.random((len(y), size)) * (nopt[:, None] - 1)).astype(np.int16)  # 0..nopt-2
    return np.where(r >= y[:, None], r + 1, r).astype(np.int16)


def surrogate_scramble_wrong(A, y, nopt, rng):
    """S1: keep every agent's correctness/abstention pattern, redraw each wrong answer uniformly
    among the item's wrong options -> removes wrong-answer coordination only."""
    C = A == y[:, None]
    R = random_wrong(y, nopt, rng, A.shape[1])
    return np.where(C, A, np.where(A < 0, -1, R)).astype(np.int16)


def surrogate_independent(A, y, nopt, rng):
    """S2: permute each agent's (correct, abstain) pattern independently across items (keeps marginal
    accuracy and abstention rate, destroys all cross-agent dependence), wrong answers uniform."""
    n, m = A.shape
    C = A == y[:, None]
    Ab = A < 0
    R = random_wrong(y, nopt, rng, m)
    out = np.empty_like(A)
    for j in range(m):
        p = rng.permutation(n)
        out[:, j] = np.where(C[p, j], y, np.where(Ab[p, j], -1, R[:, j]))
    return out


def moran_fix(k, N, lam):
    """Probability that a Moran/biased-voter process with N agents, k initially correct and relative
    persuasiveness lam of the correct answer, absorbs at all-correct."""
    k = np.asarray(k, dtype=float)
    if abs(lam - 1.0) < 1e-12:
        return k / N
    return (1 - lam ** (-k)) / (1 - lam ** (-float(N)))
