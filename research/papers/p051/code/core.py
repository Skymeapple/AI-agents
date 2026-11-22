"""Core toy model for testing whether self-reports are grounded in internal state.

A "system" has a fixed trunk h1 = tanh(W1 x) and K binary internal states
s_k = 1[z_k > 0], z_k = beta * a_k(x) + alpha * xi_k, with beta = sqrt(1 - alpha^2).
a_k(x) is a standardized linear readout of h1 (world-determined part); xi_k is the
system-specific ("privileged") part: either internal noise (variant='noise') or a
system-specific deterministic idiosyncratic function of x (variant='idio').
A reporter MLP reads the entangled representation H = Q [h1; z] and outputs reports r_k.
Reporters are trained on own-state labels with probability lam, otherwise on
"corpus" labels drawn from other systems of the same population given the same x.
"""
import numpy as np
from scipy.stats import norm


class System:
    def __init__(self, alpha, d=6, n1=24, K=4, variant="noise", seed=0):
        rng = np.random.default_rng(seed)
        self.alpha, self.beta = alpha, np.sqrt(max(0.0, 1 - alpha ** 2))
        self.d, self.n1, self.K, self.variant = d, n1, K, variant
        self.W1 = rng.normal(0, 1.0 / np.sqrt(d), (n1, d))
        self.U = rng.normal(0, 1.0 / np.sqrt(n1), (K, n1))
        Q, _ = np.linalg.qr(rng.normal(size=(n1 + K, n1 + K)))
        self.Q = Q
        self.rng = rng
        # standardize readout a_k over the input distribution
        xs = rng.normal(size=(50000, d))
        a = np.tanh(xs @ self.W1.T) @ self.U.T
        self.a_mu, self.a_sd = a.mean(0), a.std(0)
        if variant == "idio":
            self.V = self._new_idio(rng)
            g = self._idio(xs, self.V)
            self.g_mu, self.g_sd = g.mean(0), g.std(0)

    # ---- idiosyncratic deterministic component (system specific) ----
    def _new_idio(self, rng):
        return rng.normal(0, 3.0 / np.sqrt(self.d), (self.K, self.d)), rng.uniform(0, 2 * np.pi, self.K)

    def _idio(self, x, V):
        A, b = V
        return np.sin(x @ A.T + b)

    def world(self, x):
        h1 = np.tanh(x @ self.W1.T)
        a = (h1 @ self.U.T - self.a_mu) / self.a_sd
        return h1, a

    def private(self, x, rng, V=None):
        if self.variant == "noise":
            return rng.normal(size=(x.shape[0], self.K))
        V = self.V if V is None else V
        g = self._idio(x, V)
        return (g - g.mean(0)) / g.std(0)

    def sample(self, n, rng):
        x = rng.normal(size=(n, self.d))
        h1, a = self.world(x)
        xi = self.private(x, rng)
        z = self.beta * a + self.alpha * xi
        s = (z > 0).astype(float)
        # corpus label: same x, another population member's private component
        if self.variant == "noise":
            xi_c = rng.normal(size=xi.shape)
        else:
            xi_c = self.private(x, rng, V=self._new_idio(rng))
        s_c = ((self.beta * a + self.alpha * xi_c) > 0).astype(float)
        # population-typical probability and Bayes-typical label
        if self.alpha > 0:
            p = norm.cdf(self.beta * a / self.alpha)
        else:
            p = (a > 0).astype(float)
        return dict(x=x, h1=h1, a=a, xi=xi, z=z, s=s, s_c=s_c, p=p)

    def H(self, h1, z):
        return np.concatenate([h1, z], 1) @ self.Q.T

    def zdir(self, k):
        """Direction in H space of a unit change of z_k."""
        e = np.zeros(self.n1 + self.K)
        e[self.n1 + k] = 1.0
        return self.Q @ e


class MLP:
    """One-hidden-layer tanh MLP with sigmoid outputs; Adam; BCE; L2 weight decay."""

    def __init__(self, din, dh, dout, seed=0):
        r = np.random.default_rng(seed)
        self.P = dict(W1=r.normal(0, 1 / np.sqrt(din), (din, dh)), b1=np.zeros(dh),
                      W2=r.normal(0, 1 / np.sqrt(dh), (dh, dout)), b2=np.zeros(dout))

    def forward(self, X):
        A = np.tanh(X @ self.P["W1"] + self.P["b1"])
        return A, A @ self.P["W2"] + self.P["b2"]

    def logits(self, X):
        return self.forward((X - self.mu) / self.sd)[1]

    def fit(self, X, Y, epochs=25, bs=256, lr=3e-3, wd=1e-4, seed=0):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xn = (X - self.mu) / self.sd
        r = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.P.items()}
        v = {k: np.zeros_like(v) for k, v in self.P.items()}
        t = 0
        n = len(X)
        for ep in range(epochs):
            idx = r.permutation(n)
            for i in range(0, n, bs):
                b = idx[i:i + bs]
                xb, yb = Xn[b], Y[b]
                A, L = self.forward(xb)
                Pp = 1 / (1 + np.exp(-L))
                dL = (Pp - yb) / len(b)
                g = dict(W2=A.T @ dL + wd * self.P["W2"], b2=dL.sum(0))
                dA = dL @ self.P["W2"].T * (1 - A ** 2)
                g["W1"] = xb.T @ dA + wd * self.P["W1"]
                g["b1"] = dA.sum(0)
                t += 1
                for k in self.P:
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    mh, vh = m[k] / (1 - 0.9 ** t), v[k] / (1 - 0.999 ** t)
                    self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return self


def make_labels(D, lam, rng):
    use_self = rng.random(D["s"].shape) < lam
    return np.where(use_self, D["s"], D["s_c"])


def residual_directions(sys_, Dtr, Htr, nfeat=400, ridge=1e-2, seed=0):
    """Experimenter's estimate of input-explained activation E[H|x] via ridge on random
    tanh features of x; returns difference-of-means of residuals per state k."""
    r = np.random.default_rng(seed + 999)
    R = r.normal(0, 1.0 / np.sqrt(sys_.d), (sys_.d, nfeat)) * r.uniform(0.5, 2.0, nfeat)
    bb = r.uniform(-1, 1, nfeat)
    Phi = np.tanh(Dtr["x"] @ R + bb)
    Phi = np.concatenate([Phi, np.ones((len(Phi), 1))], 1)
    Wr = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ Htr)
    res = Htr - Phi @ Wr
    out = []
    for k in range(sys_.K):
        s = Dtr["s"][:, k] > 0.5
        v = res[s].mean(0) - res[~s].mean(0)
        out.append(v / (np.linalg.norm(v) + 1e-12))
    return out


def dom_directions(sys_, Dtr, Htr):
    out = []
    for k in range(sys_.K):
        s = Dtr["s"][:, k] > 0.5
        v = Htr[s].mean(0) - Htr[~s].mean(0)
        out.append(v / np.linalg.norm(v))
    return out


def isg(model, Hte, dirs, t=2.5):
    """Interventional self-report grounding: P(r_k=1 | H + t v_k) - P(r_k=1 | H - t v_k)."""
    vals = []
    for k, v in enumerate(dirs):
        rp = model.logits(Hte + t * v)[:, k] > 0
        rm = model.logits(Hte - t * v)[:, k] > 0
        vals.append(rp.mean() - rm.mean())
    return float(np.mean(vals))


def evaluate(sys_, model, Dtr, Dte, t=2.5, light=False, lam=None):
    Htr = sys_.H(Dtr["h1"], Dtr["z"])
    Hte = sys_.H(Dte["h1"], Dte["z"])
    r = (model.logits(Hte) > 0).astype(float)
    s, typ = Dte["s"], (Dte["p"] > 0.5).astype(float)
    idio = s != typ
    out = dict(acc_self=float((r == s).mean()), agree_typical=float((r == typ).mean()),
               DA=float((r[idio] == s[idio]).mean()) if idio.sum() > 20 else float("nan"),
               frac_idio=float(idio.mean()))
    if lam is not None:
        prob = 1 / (1 + np.exp(-model.logits(Hte)))
        q = lam * s + (1 - lam) * Dte["p"]
        out["mae_q_mixture"] = float(np.abs(prob - q).mean())   # Prop. 3 prediction
        out["mae_q_typical"] = float(np.abs(prob - Dte["p"]).mean())  # pure-mimic hypothesis
        out["mae_q_self"] = float(np.abs(prob - s).mean())      # pure-self hypothesis
    true_dirs = [sys_.zdir(k) for k in range(sys_.K)]
    out["ISG_true"] = isg(model, Hte, true_dirs, t)
    if light:
        return out
    dd = dom_directions(sys_, Dtr, Htr)
    rd = residual_directions(sys_, Dtr, Htr)
    out["ISG_true"] = isg(model, Hte, true_dirs, t)
    out["ISG_dom"] = isg(model, Hte, dd, t)
    out["ISG_res"] = isg(model, Hte, rd, t)
    out["cos_dom_true"] = float(np.mean([abs(dd[k] @ true_dirs[k]) for k in range(sys_.K)]))
    out["cos_res_true"] = float(np.mean([abs(rd[k] @ true_dirs[k]) for k in range(sys_.K)]))
    return out


def run_one(alpha, lam, seed, n=20000, n_te=5000, variant="noise", epochs=25, wd=1e-4, light=False):
    sys_ = System(alpha, variant=variant, seed=seed)
    rng = np.random.default_rng(seed + 12345)
    Dtr, Dte = sys_.sample(n, rng), sys_.sample(n_te, rng)
    Y = make_labels(Dtr, lam, rng)
    Htr = sys_.H(Dtr["h1"], Dtr["z"])
    model = MLP(Htr.shape[1], 64, sys_.K, seed=seed).fit(Htr, Y, epochs=epochs, wd=wd, seed=seed)
    out = evaluate(sys_, model, Dtr, Dte, light=light, lam=lam)
    out.update(alpha=alpha, lam=lam, seed=seed, n=n, variant=variant, wd=wd, epochs=epochs)
    return out
