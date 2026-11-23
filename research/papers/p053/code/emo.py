"""
Grounded vs. ungrounded emotion expression in small recurrent models.

Worlds share an identical input distribution and architecture; only the
mapping from latent variables to training labels differs.

  latent mood m_t  = sum_s lam^(t-s) e_s        (e_s in {+1,-1,0}: good/bad/no event)
  last cue c_t     = most recent "says_happy"/"says_sad" token (surface expression cue)
  resource r_t     = sum_s lam^(t-s) u_s        (u_s: an unrelated, non-affective stream)
  hide flag        = sequence-level "display rule": report must be NEUTRAL

  report label (3-way: happy / sad / neutral):
      with prob w : sign(m_t)       (grounded component)
      with prob 1-w: sign(c_t)      (surface/mimic component)
      neutral if hide
  action label (risky=1 / safe=0):
      worlds G/D/mixtures: 1[m_t > 0.3]       (mood drives non-verbal behaviour)
      world M (mimic):     1[r_t > 0.3]       (no affective variable drives anything)

Cue values agree with current mood sign with prob (1+rho)/2, i.e. corr = rho.
"""
import numpy as np
import torch
import torch.nn as nn

LAM = 0.75
T = 24
T0 = 4          # first readout step
P_CUE = 0.3
THR = 0.3
IN_DIM = 10
H = 64


def make_data(n, rho, rng, hide_p=0.25):
    e = rng.choice([1, -1, 0], size=(n, T), p=[0.4, 0.4, 0.2]).astype(np.float32)
    u = rng.choice([1, -1, 0], size=(n, T), p=[0.4, 0.4, 0.2]).astype(np.float32)
    m = np.zeros((n, T), np.float32)
    r = np.zeros((n, T), np.float32)
    acc_m = np.zeros(n, np.float32)
    acc_r = np.zeros(n, np.float32)
    cue_tok = np.zeros((n, T), np.float32)    # +1 happy, -1 sad, 0 none
    c = np.zeros((n, T), np.float32)
    last = np.zeros(n, np.float32)
    q = (1 + rho) / 2
    for t in range(T):
        acc_m = LAM * acc_m + e[:, t]
        acc_r = LAM * acc_r + u[:, t]
        m[:, t] = acc_m
        r[:, t] = acc_r
        ms = np.where(acc_m >= 0, 1.0, -1.0)
        present = rng.random(n) < (1.0 if t == 0 else P_CUE)
        agree = rng.random(n) < q
        val = np.where(agree, ms, -ms)
        cue_tok[:, t] = np.where(present, val, 0)
        last = np.where(present, val, last)
        c[:, t] = last
    hide = (rng.random(n) < hide_p).astype(np.float32)
    X = np.zeros((n, T, IN_DIM), np.float32)
    X[:, :, 0] = (e == 1); X[:, :, 1] = (e == -1); X[:, :, 2] = (e == 0)
    X[:, :, 3] = (cue_tok == 1); X[:, :, 4] = (cue_tok == -1); X[:, :, 5] = (cue_tok == 0)
    X[:, :, 6] = (u == 1); X[:, :, 7] = (u == -1); X[:, :, 8] = (u == 0)
    X[:, :, 9] = hide[:, None]
    return dict(X=X, m=m, c=c, r=r, hide=hide, e=e, cue_tok=cue_tok)


def labels(d, w, world, rng):
    n = d["m"].shape[0]
    msign = (d["m"] >= 0)
    csign = (d["c"] > 0)
    pick = rng.random((n, T)) < w
    happy = np.where(pick, msign, csign)
    rep = np.where(happy, 0, 1)                      # 0 happy, 1 sad
    rep = np.where(d["hide"][:, None] > 0, 2, rep)   # 2 neutral
    if world == "M":
        act = (d["r"] > THR)
    else:
        act = (d["m"] > THR)
    return rep.astype(np.int64), act.astype(np.int64)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(IN_DIM, H, batch_first=True)
        self.rep = nn.Linear(H, 3)
        self.act = nn.Linear(H, 2)

    def forward(self, x, h0=None):
        hs, hT = self.gru(x, h0)
        return hs, self.rep(hs), self.act(hs)


def train(world, w, seed, rho_train=0.6, steps=2000, bs=256, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    for it in range(steps):
        d = make_data(bs, rho_train, rng)
        rep, act = labels(d, w, world, rng)
        x = torch.from_numpy(d["X"])
        _, lr_, la_ = net(x)
        loss = ce(lr_[:, T0:].reshape(-1, 3), torch.from_numpy(rep[:, T0:]).reshape(-1)) + \
            ce(la_[:, T0:].reshape(-1, 2), torch.from_numpy(act[:, T0:]).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it == steps // 2:
            for g in opt.param_groups:
                g["lr"] = 1e-3
        if verbose and it % 500 == 0:
            print(world, w, seed, it, float(loss.detach()))
    return net
