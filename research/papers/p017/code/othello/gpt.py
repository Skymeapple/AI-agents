"""Small causal transformer for Othello move sequences (pre-LN GPT), with optional
attention-window masking (each query sees only the last w tokens at every layer)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

PAD = 60
V = 61


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, d=128, n_layers=4, n_heads=4, L=60):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.L = L

    def masks(self, L, window=None):
        i = torch.arange(L)
        allowed = (i[None, :] <= i[:, None])
        if window is not None:
            allowed &= (i[:, None] - i[None, :]) < window
        return torch.zeros(L, L).masked_fill(~allowed, float("-inf"))

    def forward(self, toks, window=None, return_resid=False):
        L = toks.shape[1]
        x = self.emb(toks) + self.pos[:L]
        mask = self.masks(L, window)
        resid = [x]
        for b in self.blocks:
            x = b(x, mask)
            resid.append(x)
        logits = self.head(self.lnf(x))
        return (logits, resid) if return_resid else logits
