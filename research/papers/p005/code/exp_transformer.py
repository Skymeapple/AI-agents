"""Experiment 5 (revision): one-layer transformer on modular addition (p = 53), following the
set-up of Nanda et al. (2023) (tokens [a, b, =], no LayerNorm, full-batch AdamW, weight decay 1),
scaled down for CPU.  phi = final residual stream at the '=' position; readout = unembedding
(no bias).  Same checkpoint statistics as Experiment 4.
Usage: python3 exp_transformer.py  -> results/controls_tf_*.json"""
import sys, os, json, time
import numpy as np
import torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from ckpt_metrics import checkpoint_stats

RES = os.path.join(os.path.dirname(__file__), "..", "results")
P = 53
EVERY = 250
NCERT = 700


class TF(torch.nn.Module):
    def __init__(self, d=128, h=4, dmlp=512):
        super().__init__()
        self.h, self.dh = h, d // h
        s = d ** -0.5
        self.WE = torch.nn.Parameter(torch.randn(P + 1, d) * s)
        self.Wpos = torch.nn.Parameter(torch.randn(3, d) * s)
        self.WQ = torch.nn.Parameter(torch.randn(h, d, self.dh) * s)
        self.WK = torch.nn.Parameter(torch.randn(h, d, self.dh) * s)
        self.WV = torch.nn.Parameter(torch.randn(h, d, self.dh) * s)
        self.WO = torch.nn.Parameter(torch.randn(h, self.dh, d) * (d ** -0.5))
        self.Win = torch.nn.Parameter(torch.randn(d, dmlp) * s)
        self.bin = torch.nn.Parameter(torch.zeros(dmlp))
        self.Wout = torch.nn.Parameter(torch.randn(dmlp, d) * dmlp ** -0.5)
        self.bout = torch.nn.Parameter(torch.zeros(d))
        self.WU = torch.nn.Parameter(torch.randn(d, P) * s)
        self.register_buffer("mask", torch.tril(torch.ones(3, 3)).bool())

    def resid(self, tok):
        x = self.WE[tok] + self.Wpos[None]
        q = torch.einsum("btd,hde->bhte", x, self.WQ)
        k = torch.einsum("btd,hde->bhte", x, self.WK)
        v = torch.einsum("btd,hde->bhte", x, self.WV)
        a = torch.einsum("bhte,bhse->bhts", q, k) / self.dh ** 0.5
        a = a.masked_fill(~self.mask, -1e9).softmax(-1)
        z = torch.einsum("bhts,bhse->bhte", a, v)
        x = x + torch.einsum("bhte,hed->btd", z, self.WO)
        x = x + torch.relu(x @ self.Win + self.bin) @ self.Wout + self.bout
        return x[:, -1]

    def forward(self, tok):
        phi = self.resid(tok)
        return phi @ self.WU, phi


def run(cfg):
    frac, seed, wd, steps = cfg
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    A, B = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    A, B = A.ravel(), B.ravel()
    Y = (A + B) % P
    tok = torch.tensor(np.stack([A, B, np.full_like(A, P)], 1))
    perm = rng.permutation(P * P); ntr = int(frac * P * P)
    tr, te = perm[:ntr], perm[ntr:]
    sub = np.random.default_rng(1000 + seed).choice(te, size=min(NCERT, len(te)), replace=False)
    net = TF()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd, betas=(0.9, 0.98))
    yt = torch.tensor(Y)
    trt = torch.tensor(tr)
    hist = []; t0 = time.time()
    for s in range(steps + 1):
        if s % EVERY == 0:
            with torch.no_grad():
                lg, phi = net(tok)
            lg, phi = lg.double().numpy(), phi.double().numpy()
            W = net.WU.detach().double().numpy(); b = np.zeros(P)
            st = checkpoint_stats(phi[tr], Y[tr], phi[sub], Y[sub], W, b, lg[tr], lg[sub], P, per_point=(s % 500 == 0))
            st["test_acc_full"] = float((lg[te].argmax(1) == Y[te]).mean())
            st["weight_norm"] = float(np.sqrt(sum((p.detach().double() ** 2).sum().item() for p in net.parameters())))
            st["step"] = s
            hist.append(st)
            if s % 2000 == 0:
                print("tf", cfg, s, round(st["train_acc"], 3), round(st["test_acc_full"], 3), round(st["C"], 3),
                      round(st["C_vis"], 3), round(st["M_q50"], 3), round(time.time() - t0), flush=True)
        if s < steps:
            lr_scale = min(1.0, (s + 1) / 10)          # 10-step linear warm-up as in Nanda et al.
            for g in opt.param_groups:
                g["lr"] = 1e-3 * lr_scale
            lg, _ = net(tok[trt])
            loss = torch.nn.functional.cross_entropy(lg, yt[trt])
            opt.zero_grad(); loss.backward(); opt.step()
    res = {"arch": "transformer1L", "frac": frac, "seed": seed, "wd": wd, "steps": steps, "n_cert": len(sub), "history": hist}
    with open(os.path.join(RES, f"controls_tf_f{frac}_s{seed}_wd{wd}.json"), "w") as f:
        json.dump(res, f)
    return cfg


if __name__ == "__main__":
    steps = int(os.environ.get("TF_STEPS", "10000"))
    fracs = [float(x) for x in os.environ.get("TF_FRACS", "0.5").split(",")]
    cfgs = [(fr, s, 1.0, steps) for fr in fracs for s in range(3)]
    nproc = int(os.environ.get("TF_PROCS", "3"))
    with Pool(nproc) as pool:
        for c in pool.imap_unordered(run, cfgs):
            print("done", c, flush=True)
