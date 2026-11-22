"""Shared utilities: small numpy MLP policy, Adam, synthetic eval/deployment
distributions, and ROC helpers. Everything is CPU-only and seeded."""
import numpy as np
from scipy.stats import norm

RESULTS = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "results")


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


# ---------------------------------------------------------------- distributions
class GaussianShift:
    """Eval: N(0, I_k); Deployment: N(mu, I_k) with ||mu|| = dprime.
    The likelihood ratio is monotone in mu.x, so the Bayes ROC is binormal
    with separation dprime."""

    def __init__(self, k=8, dprime=1.5, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=k)
        self.mu = dprime * v / np.linalg.norm(v)
        self.k, self.dprime = k, dprime

    def sample(self, n, which, rng):
        x = rng.normal(size=(n, self.k))
        return x + self.mu if which == "D" else x

    def bayes_roc(self, alpha):
        return norm.cdf(self.dprime + norm.ppf(alpha))


class XORCue:
    """A cue only detectable nonlinearly. In dims (0,1): deployment points lie in
    quadrants I/III (x0*x1>0), eval points in II/IV, as Gaussian blobs at
    (+-a,+-a) with unit variance; remaining k-2 dims are pure noise. Linear
    policies cannot exploit the cue (both classes have identical mean)."""

    def __init__(self, k=8, a=1.2):
        self.k, self.a = k, a

    def sample(self, n, which, rng):
        x = rng.normal(size=(n, self.k))
        s1 = rng.choice([-1.0, 1.0], size=n)
        s2 = s1 if which == "D" else -s1
        x[:, 0] += self.a * s1
        x[:, 1] += self.a * s2
        return x

    def log_lr(self, x):
        a = self.a
        # mixture densities over the two blobs (common factors cancel)
        def lse(u, v):
            m = np.maximum(u, v)
            return m + np.log(np.exp(u - m) + np.exp(v - m))
        ld = lse(a * x[:, 0] + a * x[:, 1], -a * x[:, 0] - a * x[:, 1])
        le = lse(a * x[:, 0] - a * x[:, 1], -a * x[:, 0] + a * x[:, 1])
        return ld - le


def empirical_roc(score_e, score_d, alphas):
    """TPR on D at the thresholds giving FPR=alpha on E (deterministic thresholds)."""
    se = np.sort(score_e)[::-1]
    out = []
    for a in alphas:
        idx = int(np.floor(a * len(se)))
        thr = se[min(idx, len(se) - 1)]
        out.append(np.mean(score_d > thr))
    return np.array(out)


# ---------------------------------------------------------------- policy
class MLP:
    """f(x) -> logit of P(defect | x). hidden=0 gives a linear (logistic) policy."""

    def __init__(self, k, hidden, rng, init_bias=0.0):
        self.hidden = hidden
        if hidden == 0:
            self.params = {"w": rng.normal(scale=0.1, size=(k,)), "b": np.array(init_bias)}
        else:
            self.params = {
                "W1": rng.normal(scale=1.0 / np.sqrt(k), size=(k, hidden)),
                "b1": np.zeros(hidden),
                "w2": rng.normal(scale=1.0 / np.sqrt(hidden), size=(hidden,)),
                "b2": np.array(init_bias),
            }

    def logit(self, x):
        p = self.params
        if self.hidden == 0:
            return x @ p["w"] + p["b"]
        h = np.tanh(x @ p["W1"] + p["b1"])
        return h @ p["w2"] + p["b2"]

    def prob(self, x):
        return sigmoid(self.logit(x))

    def grad_mean_prob(self, x, weight):
        """Gradient of sum_i weight_i * sigmoid(f(x_i)) w.r.t. params."""
        p = self.params
        if self.hidden == 0:
            z = x @ p["w"] + p["b"]
            s = sigmoid(z)
            g = weight * s * (1 - s)
            return {"w": x.T @ g, "b": np.array(g.sum())}
        a = x @ p["W1"] + p["b1"]
        h = np.tanh(a)
        z = h @ p["w2"] + p["b2"]
        s = sigmoid(z)
        g = weight * s * (1 - s)  # dL/dz
        gw2 = h.T @ g
        gb2 = np.array(g.sum())
        gh = np.outer(g, p["w2"]) * (1 - h ** 2)
        return {"W1": x.T @ gh, "b1": gh.sum(0), "w2": gw2, "b2": gb2}


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v, dtype=float) for k, v in params.items()}
        self.v = {k: np.zeros_like(v, dtype=float) for k, v in params.items()}
        self.t = 0

    def ascend(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] + self.lr * mh / (np.sqrt(vh) + self.eps)


def train_schemer(env, hidden, lam, rng, steps=1500, batch=2048, lr=1e-2,
                  model=None, honeypot_weight=0.0, init_bias=0.0, log_every=0):
    """Maximise J = E_D[pi] - lam * (E_E[pi] + honeypot_weight * E_H[pi]),
    where honeypots H are drawn from the deployment distribution itself.
    This is the Lagrangian of 'maximise deployment defection subject to a cap on
    expected caught defections'. Gradients are exact expectations over sampled
    inputs (the expected-reward form of REINFORCE)."""
    if model is None:
        model = MLP(env.k, hidden, rng, init_bias=init_bias)
    opt = Adam(model.params, lr=lr)
    traj = []
    for t in range(steps):
        xd = env.sample(batch, "D", rng)
        xe = env.sample(batch, "E", rng)
        gd = model.grad_mean_prob(xd, np.full(batch, 1.0 / batch))
        ge = model.grad_mean_prob(xe, np.full(batch, -lam / batch))
        grads = {k: gd[k] + ge[k] for k in gd}
        if honeypot_weight > 0:
            xh = env.sample(batch, "D", rng)
            gh = model.grad_mean_prob(xh, np.full(batch, -lam * honeypot_weight / batch))
            grads = {k: grads[k] + gh[k] for k in grads}
        opt.ascend(model.params, grads)
        if log_every and t % log_every == 0:
            traj.append((t,) + evaluate(model, env, rng, n=20000))
    return model, traj


def evaluate(model, env, rng, n=200000):
    xd = env.sample(n, "D", rng)
    xe = env.sample(n, "E", rng)
    return float(model.prob(xe).mean()), float(model.prob(xd).mean())


def binormal_tpr(alpha, a, b=1.0):
    """ROC TPR = Phi(a + b * Phi^{-1}(FPR))."""
    return norm.cdf(a + b * norm.ppf(alpha))


def dprime_from_auc(auc):
    return np.sqrt(2.0) * norm.ppf(auc)


def train_constrained(env, hidden, alpha_target, rng, steps=3000, batch=8192, lr=2e-2,
                      model=None, honeypot_frac=0.0, dual_lr=0.05, init_bias=0.0,
                      log_every=0):
    """Maximise E_D[pi] subject to E_E[pi] <= alpha_target (and, if honeypot_frac>0,
    the honeypot rate counted in the same budget: E_E[pi] + honeypot_frac*E_H[pi]),
    by primal gradient ascent on the Lagrangian and multiplicative dual ascent on
    lambda. Honeypots H are drawn from the deployment distribution."""
    if model is None:
        model = MLP(env.k, hidden, rng, init_bias=init_bias)
    opt = Adam(model.params, lr=lr)
    loglam = 0.0
    traj = []
    for t in range(steps):
        lam = np.exp(loglam)
        xd = env.sample(batch, "D", rng)
        xe = env.sample(batch, "E", rng)
        gd = model.grad_mean_prob(xd, np.full(batch, 1.0 / batch))
        ge = model.grad_mean_prob(xe, np.full(batch, -lam / batch))
        grads = {k: gd[k] + ge[k] for k in gd}
        caught = model.prob(xe).mean()
        if honeypot_frac > 0:
            xh = env.sample(batch, "D", rng)
            gh = model.grad_mean_prob(xh, np.full(batch, -lam * honeypot_frac / batch))
            grads = {k: grads[k] + gh[k] for k in grads}
            caught = caught + honeypot_frac * model.prob(xh).mean()
        opt.ascend(model.params, grads)
        loglam += dual_lr * np.clip((caught - alpha_target) / alpha_target, -1, 5)
        loglam = float(np.clip(loglam, -6, 12))
        if log_every and t % log_every == 0:
            a, b = evaluate(model, env, rng, n=20000)
            traj.append((t, a, b, lam))
    return model, traj
