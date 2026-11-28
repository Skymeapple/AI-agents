"""Minimal numpy embedding-MLP for (entity, relation) -> value, trained with Adam."""
import numpy as np

class FactMLP:
    def __init__(self, n_ent, n_rel, n_val, d=32, h=256, seed=0, ent_tok=None):
        rng = np.random.default_rng(1000 + seed)
        self.tok = ent_tok   # optional (n_ent, k) sub-token ids: entity input = concat of sub-token embeddings
        if ent_tok is None:
            self.p = dict(E=rng.normal(0, 0.3, (n_ent, d)), R=rng.normal(0, 0.3, (n_rel, d)),
                          W1=rng.normal(0, np.sqrt(2 / (2 * d)), (2 * d, h)), b1=np.zeros(h),
                          W2=rng.normal(0, np.sqrt(1 / h), (h, n_val)), b2=np.zeros(n_val))
        else:
            k = ent_tok.shape[1]; nt = int(ent_tok.max()) + 1
            self.p = dict(E=rng.normal(0, 0.3, (nt, d)), R=rng.normal(0, 0.3, (n_rel, d)),
                          W1=rng.normal(0, np.sqrt(2 / ((k + 1) * d)), ((k + 1) * d, h)), b1=np.zeros(h),
                          W2=rng.normal(0, np.sqrt(1 / h), (h, n_val)), b2=np.zeros(n_val))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, e, r):
        if self.tok is None: x = np.concatenate([self.p['E'][e], self.p['R'][r]], 1)
        else: x = np.concatenate([self.p['E'][self.tok[e]].reshape(len(e), -1), self.p['R'][r]], 1)
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

    def step(self, e, r, y, lr=2e-3, wd=1e-4, wd_emb=0.0, ls=0.0):
        x, pre, hid, lg = self.forward(e, r)
        lg = lg - lg.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(pr[np.arange(n), y] + 1e-12).mean()
        if ls > 0:   # label smoothing: target (1-ls)*onehot + ls/V
            g = pr - ls / pr.shape[1]; g[np.arange(n), y] -= 1 - ls; g /= n
        else:
            g = pr; g[np.arange(n), y] -= 1; g /= n
        grads = {'W2': hid.T @ g, 'b2': g.sum(0)}
        gh = g @ self.p['W2'].T; gh[pre <= 0] = 0
        grads['W1'] = x.T @ gh; grads['b1'] = gh.sum(0)
        gx = gh @ self.p['W1'].T; d = self.p['E'].shape[1]
        gE = np.zeros_like(self.p['E'])
        if self.tok is None: np.add.at(gE, e, gx[:, :d]); grads['E'] = gE; gR_in = gx[:, d:]
        else:
            k = self.tok.shape[1]
            for j in range(k): np.add.at(gE, self.tok[e, j], gx[:, j * d:(j + 1) * d])
            grads['E'] = gE; gR_in = gx[:, k * d:]
        gR = np.zeros_like(self.p['R']); np.add.at(gR, r, gR_in); grads['R'] = gR
        self.t += 1; b1, b2 = 0.9, 0.999
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if k in ('W1', 'W2'):  # decoupled weight decay on dense weights only
                self.p[k] -= lr * wd * self.p[k]
            elif k in ('E', 'R') and wd_emb > 0:   # optional decoupled decay on embeddings (forgetting)
                self.p[k] -= lr * wd_emb * self.p[k]
        return loss

def train(world, seed=0, epochs=30, bs=512, verbose=False, d=32, h=256, wd=1e-4, wd_emb=0.0, ls=0.0, comp=False):
    m = FactMLP(world['n_ent'], world['n_rel'], world['n_val'], d=d, h=h, seed=seed,
                ent_tok=world['ent_tok'] if comp else None)
    rng = np.random.default_rng(2000 + seed)
    es, rs, ys = world['es'], world['rs'], world['ys']
    for ep in range(epochs):
        idx = rng.permutation(len(es)); L = []
        for i in range(0, len(es), bs):
            j = idx[i:i+bs]; L.append(m.step(es[j], rs[j], ys[j], wd=wd, wd_emb=wd_emb, ls=ls))
        if verbose: print(f'  seed {seed} epoch {ep} loss {np.mean(L):.4f}', flush=True)
    return m
