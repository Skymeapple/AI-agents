"""Minimal numpy embedding-MLP for (entity, relation) -> value, trained with Adam."""
import numpy as np

class FactMLP:
    def __init__(self, n_ent, n_rel, n_val, d=32, h=256, seed=0):
        rng = np.random.default_rng(1000 + seed)
        self.p = dict(E=rng.normal(0, 0.3, (n_ent, d)), R=rng.normal(0, 0.3, (n_rel, d)),
                      W1=rng.normal(0, np.sqrt(2 / (2 * d)), (2 * d, h)), b1=np.zeros(h),
                      W2=rng.normal(0, np.sqrt(1 / h), (h, n_val)), b2=np.zeros(n_val))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, e, r):
        x = np.concatenate([self.p['E'][e], self.p['R'][r]], 1)
        pre = x @ self.p['W1'] + self.p['b1']
        hid = np.maximum(pre, 0)
        logits = hid @ self.p['W2'] + self.p['b2']
        return x, pre, hid, logits

    def predict(self, e, r, bs=8192):
        P, H = [], []
        for i in range(0, len(e), bs):
            _, _, hid, lg = self.forward(e[i:i+bs], r[i:i+bs])
            lg = lg - lg.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
            P.append(pr); H.append(hid)
        return np.concatenate(P), np.concatenate(H)

    def step(self, e, r, y, lr=2e-3, wd=1e-4):
        x, pre, hid, lg = self.forward(e, r)
        lg = lg - lg.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(pr[np.arange(n), y] + 1e-12).mean()
        g = pr; g[np.arange(n), y] -= 1; g /= n
        grads = {'W2': hid.T @ g, 'b2': g.sum(0)}
        gh = g @ self.p['W2'].T; gh[pre <= 0] = 0
        grads['W1'] = x.T @ gh; grads['b1'] = gh.sum(0)
        gx = gh @ self.p['W1'].T; d = self.p['E'].shape[1]
        gE = np.zeros_like(self.p['E']); np.add.at(gE, e, gx[:, :d]); grads['E'] = gE
        gR = np.zeros_like(self.p['R']); np.add.at(gR, r, gx[:, d:]); grads['R'] = gR
        self.t += 1; b1, b2 = 0.9, 0.999
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if k in ('W1', 'W2'):  # decoupled weight decay on dense weights only
                self.p[k] -= lr * wd * self.p[k]
        return loss

def train(world, seed=0, epochs=30, bs=512, verbose=False):
    m = FactMLP(world['n_ent'], world['n_rel'], world['n_val'], seed=seed)
    rng = np.random.default_rng(2000 + seed)
    es, rs, ys = world['es'], world['rs'], world['ys']
    for ep in range(epochs):
        idx = rng.permutation(len(es)); L = []
        for i in range(0, len(es), bs):
            j = idx[i:i+bs]; L.append(m.step(es[j], rs[j], ys[j]))
        if verbose: print(f'  seed {seed} epoch {ep} loss {np.mean(L):.4f}', flush=True)
    return m
