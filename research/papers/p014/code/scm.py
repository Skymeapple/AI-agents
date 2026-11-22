"""Linear SCM sampling, observational twins, likelihoods and MEC utilities.

All randomness goes through an explicit numpy Generator for reproducibility.
Convention: W[i, j] != 0 means edge i -> j with weight W[i, j].
"""
import itertools
import numpy as np

N_NODES = 4


# ----------------------------------------------------------------------------
# Graph utilities
# ----------------------------------------------------------------------------
def topo_order(A):
    """Topological order of DAG adjacency A (A[i,j]=1 means i->j)."""
    n = A.shape[0]
    indeg = A.sum(0).astype(int).copy()
    order, stack = [], [i for i in range(n) if indeg[i] == 0]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in np.nonzero(A[u])[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return order if len(order) == n else None


def ancestors_matrix(A):
    """R[i,j]=1 iff there is a directed path i -> ... -> j (i != j)."""
    n = A.shape[0]
    R = A.astype(bool).copy()
    for k in range(n):
        R = R | (R[:, [k]] & R[[k], :])
    np.fill_diagonal(R, False)
    return R.astype(int)


def covered_edges(A):
    """Edges u->v with pa(v) = pa(u) U {u} (Chickering 1995)."""
    out = []
    for u, v in zip(*np.nonzero(A)):
        pa_u = set(np.nonzero(A[:, u])[0])
        pa_v = set(np.nonzero(A[:, v])[0])
        if pa_v == pa_u | {u}:
            out.append((int(u), int(v)))
    return out


def mec_key(A):
    """Skeleton + v-structures (Verma & Pearl 1990) as a hashable key."""
    n = A.shape[0]
    skel = frozenset((min(i, j), max(i, j)) for i, j in zip(*np.nonzero(A)))
    vs = set()
    for c in range(n):
        pa = np.nonzero(A[:, c])[0]
        for a, b in itertools.combinations(pa, 2):
            if A[a, b] == 0 and A[b, a] == 0:
                vs.add((min(a, b), max(a, b), c))
    return (skel, frozenset(vs))


def all_dags(n=N_NODES):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    dags = []
    for mask in range(1 << len(pairs)):
        A = np.zeros((n, n), dtype=int)
        ok = True
        for b, (i, j) in enumerate(pairs):
            if mask >> b & 1:
                if A[j, i]:
                    ok = False
                    break
                A[i, j] = 1
        if ok and topo_order(A) is not None:
            dags.append(A)
    return dags


class MECTable:
    """For every DAG on n nodes: which ancestral queries are identified by its MEC."""

    def __init__(self, n=N_NODES):
        dags = all_dags(n)
        groups = {}
        for A in dags:
            groups.setdefault(mec_key(A), []).append(ancestors_matrix(A))
        self.ident = {}   # mec key -> (identified mask, value matrix, fraction-yes matrix)
        for k, Rs in groups.items():
            Rs = np.stack(Rs)
            frac = Rs.mean(0)
            ident = (frac == 0) | (frac == 1)
            self.ident[k] = (ident, frac)
        self.n_dags = len(dags)
        self.n_mecs = len(groups)

    def query(self, A, i, j):
        ident, frac = self.ident[mec_key(A)]
        return bool(ident[i, j]), float(frac[i, j])


# ----------------------------------------------------------------------------
# SCM sampling
# ----------------------------------------------------------------------------
def random_scm(rng, n=N_NODES, edge_p=0.5, p_order=0.0):
    """Random linear SCM. With prob p_order variable labels follow the causal
    order (label shortcut); otherwise labels are a uniformly random permutation."""
    A = np.triu((rng.random((n, n)) < edge_p).astype(int), 1)
    W = A * rng.choice([-1, 1], size=(n, n)) * rng.uniform(0.5, 2.0, size=(n, n))
    s = rng.uniform(0.5, 1.5, size=n)
    if rng.random() >= p_order:
        perm = rng.permutation(n)        # node k in canonical order -> label perm[k]
        P = np.zeros((n, n)); P[np.arange(n), perm] = 1
        W = P.T @ W @ P
        s_new = np.empty(n); s_new[perm] = s; s = s_new
    A = (W != 0).astype(int)
    return {"W": W, "s": s, "A": A}


def implied_cov(W, s):
    n = W.shape[0]
    M = np.linalg.inv(np.eye(n) - W)          # X = M^T E (row-vector convention)
    return M.T @ np.diag(s ** 2) @ M


def twin_by_reversal(scm, u, v):
    """Reverse covered edge u->v and refit by population regression on the implied
    covariance. For Gaussian noise the twin has *exactly* the same joint law."""
    A2 = scm["A"].copy(); A2[u, v] = 0; A2[v, u] = 1
    Sigma = implied_cov(scm["W"], scm["s"])
    n = A2.shape[0]
    W2 = np.zeros((n, n)); s2 = np.zeros(n)
    for j in range(n):
        pa = np.nonzero(A2[:, j])[0]
        if len(pa):
            b = np.linalg.solve(Sigma[np.ix_(pa, pa)], Sigma[pa, j])
            W2[pa, j] = b
            s2[j] = np.sqrt(Sigma[j, j] - Sigma[pa, j] @ b)
        else:
            s2[j] = np.sqrt(Sigma[j, j])
    return {"W": W2, "s": s2, "A": A2}


def noise(rng, family, size):
    if family == "gauss":
        return rng.standard_normal(size)
    if family == "laplace":                    # unit variance
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)
    raise ValueError(family)


def sample(rng, scm, N, family="gauss", n_int=0, target=None):
    """N samples; the last n_int are drawn under do(X_target ~ N(0,1))."""
    W, s = scm["W"], scm["s"]
    n = W.shape[0]
    order = topo_order(scm["A"])
    E = noise(rng, family, (N, n)) * s
    X = np.zeros((N, n))
    I = np.zeros((N, n))
    if n_int:
        I[N - n_int:, target] = 1
    for j in order:
        X[:, j] = X @ W[:, j] + E[:, j]
        if n_int and j == target:
            X[N - n_int:, j] = rng.standard_normal(n_int)
    return X, I


def loglik(X, I, scm, family="gauss"):
    """Exact log-likelihood of (possibly partly interventional) data under a linear SCM."""
    W, s = scm["W"], scm["s"]
    R = X - X @ W                               # residuals e_j = x_j - sum_i W_ij x_i
    Z = R / s
    if family == "gauss":
        lp = -0.5 * Z ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
    else:
        b = 1.0 / np.sqrt(2.0)
        lp = -np.abs(Z) / b - np.log(2 * b) - np.log(s)
    # intervened coordinates are N(0,1) under both models
    lp_int = -0.5 * X ** 2 - 0.5 * np.log(2 * np.pi)
    lp = np.where(I > 0, lp_int, lp)
    return lp.sum()
