"""Minimal numpy MLP with ReLU, softmax cross-entropy and AdamW (no autograd library available)."""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, scale=1.0):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, scale / np.sqrt(sizes[i]), (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, X, keep=False):
        """Return logits and list of representations [input, h1, ..., h_{L-1}]."""
        reps = [X]
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0.0)
            reps.append(h)
        logits = h @ self.W[-1] + self.b[-1]
        return logits, reps

    def head(self, h, layer):
        """Apply the network from representation layer `layer` onward."""
        for i in range(layer, len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0.0)
        return h @ self.W[-1] + self.b[-1]

    def step(self, X, y, lr=1e-3, wd=1.0, b1=0.9, b2=0.98, eps=1e-8):
        logits, reps = self.forward(X)
        n = X.shape[0]
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        g = p.copy(); g[np.arange(n), y] -= 1; g /= n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = reps[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (reps[i] > 0)
        grads = gW + gb
        self.t += 1
        for k, (P, G) in enumerate(zip(self.params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * G
            self.v[k] = b2 * self.v[k] + (1 - b2) * G * G
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            P *= (1 - lr * wd)
            P -= lr * mh / (np.sqrt(vh) + eps)
        return loss

    def predict(self, X):
        return self.forward(X)[0].argmax(1)
