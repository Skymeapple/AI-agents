"""E2: Verified self-training on a structured 'skill' task vs. an unstructured
'knowledge' task, with a real learner (multinomial logistic regression).

Skill task:   x in {-1,+1}^d, y = argmax(W* x + b*), k classes (teacher is linear).
Knowledge:    x = one-hot(item), item in {1..K}, y_item ~ Uniform(k) (no structure);
              human prompts are Zipf(1.1) over items, synthetic prompts uniform.
Self-training round: sample m fresh prompts, the learner samples a label from its
own softmax, a verifier accepts with prob a if the label is correct and b if not,
accepted pairs are ACCUMULATED (or REPLACE the previous round's synthetic data),
and the learner is retrained from scratch on human + synthetic data.

Outputs: final accuracies on a grid of verifier (a, b) and human budget n0,
human-equivalent sample size N_eff (via plain learning curves) and the
"external information" in verifier bits, I(C;V) per proposal summed over rounds.
"""
import json, os, sys, time, numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
D, KC = 20, 8          # skill: input dim, classes
KITEMS = 400           # knowledge: number of items
LAM = 1e-3
ROUNDS, M_PROP = 8, 2000

def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def mutual_info_cv(q, a, b):
    """I(C;V) in bits, C~Bern(q) correctness, V accept bit with P(V=1|C=1)=a, P(V=1|C=0)=b."""
    pv = a * q + b * (1 - q)
    return h2(pv) - (q * h2(a) + (1 - q) * h2(b))

class Task:
    def __init__(self, kind, seed):
        self.kind = kind
        r = np.random.default_rng(1000 + seed)
        if kind == "skill":
            self.W = r.normal(size=(D, KC)); self.b0 = r.normal(size=KC) * 0.5
            self.dim = D
        else:
            self.lab = r.integers(0, KC, KITEMS)
            w = np.arange(1, KITEMS + 1) ** -1.1; self.pz = w / w.sum()
            self.dim = KITEMS
    def prompts(self, n, r, human=True):
        if self.kind == "skill":
            return r.choice([-1.0, 1.0], size=(n, D))
        ids = r.choice(KITEMS, size=n, p=self.pz) if human else r.integers(0, KITEMS, n)
        X = np.zeros((n, KITEMS)); X[np.arange(n), ids] = 1.0
        return X
    def label(self, X):
        if self.kind == "skill":
            return np.argmax(X @ self.W + self.b0, 1)
        return self.lab[np.argmax(X, 1)]
    def test_set(self, r):
        if self.kind == "skill":
            X = r.choice([-1.0, 1.0], size=(20000, D)); return X, self.label(X), None
        X = np.eye(KITEMS); return X, self.lab.copy(), self.pz  # Zipf-weighted accuracy

def fit(X, y, w=None):
    n, d = X.shape
    Y = np.zeros((n, KC)); Y[np.arange(n), y] = 1
    w = np.ones(n) if w is None else w
    sw = w.sum()
    def f(theta):
        Wb = theta.reshape(d + 1, KC)
        Z = X @ Wb[:-1] + Wb[-1]
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        loss = -(w * np.log(np.sum(P * Y, 1) + 1e-12)).sum() / sw + 0.5 * LAM * np.sum(Wb[:-1] ** 2)
        G = (P - Y) * w[:, None] / sw
        gW = X.T @ G + LAM * Wb[:-1]; gb = G.sum(0)
        return loss, np.vstack([gW, gb]).ravel()
    res = minimize(f, np.zeros((d + 1) * KC), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return res.x.reshape(d + 1, KC)

def predict_proba(Wb, X):
    Z = X @ Wb[:-1] + Wb[-1]; Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); return P / P.sum(1, keepdims=True)

def accuracy(Wb, Xt, yt, wt):
    pred = np.argmax(predict_proba(Wb, Xt), 1)
    c = (pred == yt).astype(float)
    return float(c.mean() if wt is None else np.sum(wt * c))

def run(kind, n0, a, b, seed, mode="accumulate", rounds=ROUNDS, m=M_PROP):
    r = np.random.default_rng(seed)
    task = Task(kind, seed)
    Xt, yt, wt = task.test_set(np.random.default_rng(7 + seed))
    Xh = task.prompts(n0, r, human=True); yh = task.label(Xh)
    Wb = fit(Xh, yh)
    accs = [accuracy(Wb, Xt, yt, wt)]
    Xs = np.zeros((0, task.dim)); ys = np.zeros(0, int)
    bits, n_acc, n_acc_correct = 0.0, 0, 0
    for t in range(rounds):
        Xp = task.prompts(m, r, human=False)
        P = predict_proba(Wb, Xp)
        cum = P.cumsum(1); u = r.random((m, 1))
        yp = np.minimum((u > cum).sum(1), KC - 1)        # sample from own softmax
        correct = yp == task.label(Xp)
        q = correct.mean()
        bits += m * mutual_info_cv(q, a, b)
        acc_mask = np.where(correct, r.random(m) < a, r.random(m) < b)
        n_acc += acc_mask.sum(); n_acc_correct += (acc_mask & correct).sum()
        if mode == "accumulate":
            Xs = np.vstack([Xs, Xp[acc_mask]]); ys = np.concatenate([ys, yp[acc_mask]])
        else:
            Xs = Xp[acc_mask]; ys = yp[acc_mask]
        Wb = fit(np.vstack([Xh, Xs]), np.concatenate([yh, ys]))
        accs.append(accuracy(Wb, Xt, yt, wt))
    return dict(kind=kind, n0=n0, a=a, b=b, seed=seed, mode=mode, accs=accs, bits=bits,
                n_accepted=int(n_acc), precision=float(n_acc_correct / max(n_acc, 1)))

def plain_curve(kind, seed, grid):
    r = np.random.default_rng(seed); task = Task(kind, seed)
    Xt, yt, wt = task.test_set(np.random.default_rng(7 + seed))
    out = []
    for n in grid:
        X = task.prompts(n, np.random.default_rng(seed * 100 + n), human=True)
        out.append(accuracy(fit(X, task.label(X)), Xt, yt, wt))
    return out

def _star(args):
    return run(*args)

if __name__ == "__main__":
    t0 = time.time()
    SEEDS = [0, 1, 2]
    jobs = []
    for kind, n0s in [("skill", [24, 64, 256]), ("knowledge", [200, 800])]:
        for n0 in n0s:
            for (a, b) in [(1, 0), (1, 0.05), (1, 0.15), (1, 0.3), (1, 0.6), (1, 1.0), (0.5, 0.5), (0.7, 0.1)]:
                for s in SEEDS:
                    jobs.append((kind, n0, a, b, s, "accumulate"))
    for kind, n0 in [("skill", 64), ("knowledge", 800)]:
        for (a, b) in [(1, 0), (1, 0.15), (1, 0.6), (1, 1.0)]:
            for s in SEEDS:
                jobs.append((kind, n0, a, b, s, "replace"))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_star, jobs, chunksize=2))
    grids = {"skill": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384],
             "knowledge": [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]}
    curves = {k: [plain_curve(k, s, g) for s in SEEDS] for k, g in grids.items()}
    json.dump(dict(runs=res, curves=curves, grids=grids, rounds=ROUNDS, m=M_PROP),
              open(os.path.join(OUT, "e2_selftrain.json"), "w"), indent=1)
    print("done in %.1fs, %d runs" % (time.time() - t0, len(res)))
