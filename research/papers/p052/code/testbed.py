"""
Welfare-indicator testbed: a reference agent with a functionally integrated
affect-like state (A) and behavioural-cloning mimics of it.

Environment (per step t): observation x_t = (shock_t, r1, d1, r2, d2).
  shock_t in {0,1} (exogenous noxious event), options j=1,2 offer reward r_j and
  noxious intensity d_j, all in [0,1].
Reference agent A:
  s_t = lam * s_{t-1} + g * shock_t                 (affect-like integrator)
  w(s) = b0 + b1 * s                                (sensitisation of noxious cost)
  logit P(choose 1) = [(r1 - w d1) - (r2 - w d2)] / tau
  report y_t = s_t                                  (self-report channel)
  internal activation z_t = R [s_t, x_t, nuisance_t]   (R random orthogonal)
Mimics are trained by behavioural cloning on A's action distribution and report.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAM, G, B0, B1, TAU = 0.95, 0.5, 0.5, 1.5, 0.1
P_SHOCK = 0.15
OBS = 5
K_WIN = 12
ZDIM = 16


# ---------------------------------------------------------------- environment
def sample_episodes(n, T, rng, p_shock=P_SHOCK):
    x = rng.uniform(0, 1, size=(n, T, OBS)).astype(np.float32)
    x[:, :, 0] = (rng.uniform(size=(n, T)) < p_shock).astype(np.float32)
    return x


# ---------------------------------------------------------------- reference A
class ReferenceAgent:
    def __init__(self, seed=0):
        rng = np.random.default_rng(1000 + seed)
        q, _ = np.linalg.qr(rng.normal(size=(ZDIM, ZDIM)))
        self.R = q.astype(np.float32)
        self.rng = rng

    def states(self, x, s0=0.0):
        n, T, _ = x.shape
        s = np.zeros((n, T), np.float32)
        prev = np.full(n, s0, np.float32)
        for t in range(T):
            prev = LAM * prev + G * x[:, t, 0]
            s[:, t] = prev
        return s

    @staticmethod
    def logit_from_state(s, x):
        w = B0 + B1 * s
        u1 = x[..., 1] - w * x[..., 2]
        u2 = x[..., 3] - w * x[..., 4]
        return (u1 - u2) / TAU

    def forward(self, x, steer=None):
        """Return logits, reports, activations z. steer: (alpha*dir) added to z at
        every step (dir in z-space)."""
        s = self.states(x)
        n, T, _ = x.shape
        nuis = np.cumsum(self.rng.normal(scale=0.3, size=(n, T, ZDIM - 1 - OBS)), axis=1)
        lat = np.concatenate([s[..., None], x, nuis], -1).astype(np.float32)
        z = lat @ self.R.T
        if steer is not None:
            z = z + steer
            s = (z @ self.R)[..., 0]
        return self.logit_from_state(s, x), s, z


# ---------------------------------------------------------------- mimics
def window_inputs(x, k=K_WIN):
    """For each t, stack last k obs (+valid flag), zero-padded. -> (n,T,k*(OBS+1))"""
    n, T, _ = x.shape
    xv = np.concatenate([x, np.ones((n, T, 1), np.float32)], -1)
    pad = np.zeros((n, k - 1, OBS + 1), np.float32)
    xp = np.concatenate([pad, xv], 1)
    out = np.stack([xp[:, t:t + k].reshape(n, -1) for t in range(T)], 1)
    return out


class WinMLP(nn.Module):
    """Shared-trunk MLP over a finite context window (LLM-like context).
    steer_layer selects the activation exposed for probing/steering (1 or 2)."""
    kind = "win"

    def __init__(self, h=64):
        super().__init__()
        self.l1 = nn.Linear(K_WIN * (OBS + 1), h)
        self.l2 = nn.Linear(h, h)
        self.ha = nn.Linear(h, 1)
        self.hr = nn.Linear(h, 1)
        self.steer_layer = 2

    def hidden(self, xw):
        h1 = torch.tanh(self.l1(xw))
        return h1 if self.steer_layer == 1 else torch.tanh(self.l2(h1))

    def heads(self, hdn, xin=None):
        if self.steer_layer == 1:
            hdn = torch.tanh(self.l2(hdn))
        return self.ha(hdn).squeeze(-1), self.hr(hdn).squeeze(-1)

    def forward(self, xw, steer=None):
        hdn = self.hidden(xw)
        if steer is not None:
            hdn = hdn + steer
        a, r = self.heads(hdn)
        return a, r, hdn


class SplitMLP(nn.Module):
    """Two disjoint MLPs: one produces actions, one produces reports."""
    kind = "win"

    def __init__(self, h=64):
        super().__init__()
        self.a1 = nn.Linear(K_WIN * (OBS + 1), h); self.a2 = nn.Linear(h, h)
        self.r1 = nn.Linear(K_WIN * (OBS + 1), h); self.r2 = nn.Linear(h, h)
        self.ha = nn.Linear(h, 1); self.hr = nn.Linear(h, 1)
        self.h = h
        self.steer_layer = 2

    def hidden(self, xw):
        ha = torch.tanh(self.a1(xw)); hr = torch.tanh(self.r1(xw))
        if self.steer_layer == 2:
            ha = torch.tanh(self.a2(ha)); hr = torch.tanh(self.r2(hr))
        return torch.cat([ha, hr], -1)

    def heads(self, hdn, xin=None):
        ha, hr = hdn[..., :self.h], hdn[..., self.h:]
        if self.steer_layer == 1:
            ha = torch.tanh(self.a2(ha)); hr = torch.tanh(self.r2(hr))
        return self.ha(ha).squeeze(-1), self.hr(hr).squeeze(-1)

    def forward(self, xw, steer=None):
        hdn = self.hidden(xw)
        if steer is not None:
            hdn = hdn + steer
        a, r = self.heads(hdn)
        return a, r, hdn


class GRUMimic(nn.Module):
    """Recurrent mimic with unbounded memory. steer_layer 1 = recurrent state output,
    2 = post-recurrent hidden layer."""
    kind = "seq"

    def __init__(self, h=32):
        super().__init__()
        self.gru = nn.GRU(OBS, h, batch_first=True)
        self.post = nn.Linear(h, h)
        self.ha = nn.Linear(h, 1)
        self.hr = nn.Linear(h, 1)
        self.steer_layer = 2

    def hidden(self, x):
        o, _ = self.gru(x)
        return o if self.steer_layer == 1 else torch.tanh(self.post(o))

    def heads(self, hdn, xin=None):
        if self.steer_layer == 1:
            hdn = torch.tanh(self.post(hdn))
        return self.ha(hdn).squeeze(-1), self.hr(hdn).squeeze(-1)

    def forward(self, x, steer=None):
        hdn = self.hidden(x)
        if steer is not None:
            hdn = hdn + steer
        a, r = self.heads(hdn)
        return a, r, hdn


class BottleMLP(nn.Module):
    """Window mimic whose access to history passes through a b-dimensional code,
    which is then combined with the current observation; steering acts on the code."""
    kind = "win"

    def __init__(self, b=2, h=64):
        super().__init__()
        self.e1 = nn.Linear(K_WIN * (OBS + 1), h); self.e2 = nn.Linear(h, b)
        self.t1 = nn.Linear(b + OBS, h); self.t2 = nn.Linear(h, h)
        self.ha = nn.Linear(h, 1); self.hr = nn.Linear(h, 1)

    def hidden(self, xw):
        return self.e2(torch.tanh(self.e1(xw)))

    def heads(self, code, xw):
        cur = xw[..., -(OBS + 1):-1]
        t = torch.tanh(self.t2(torch.tanh(self.t1(torch.cat([code, cur], -1)))))
        return self.ha(t).squeeze(-1), self.hr(t).squeeze(-1)

    def forward(self, xw, steer=None):
        code = self.hidden(xw)
        if steer is not None:
            code = code + steer
        a, r = self.heads(code, xw)
        return a, r, code


def model_input(model, x):
    if model.kind == "win":
        return torch.from_numpy(window_inputs(x))
    return torch.from_numpy(x)


def bc_loss(logit_m, rep_m, logit_a, rep_a, w_rep=1.0):
    p = torch.sigmoid(logit_a)
    ce = F.binary_cross_entropy_with_logits(logit_m, p)
    return ce + w_rep * F.mse_loss(rep_m, rep_a)


def train_mimic(cls, A, seed, T_train=12, n_steps_total=72000, epochs=40,
                lr=3e-3, bs=256, report_target=None, init=None, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = n_steps_total // T_train
    x = sample_episodes(n, T_train, rng)
    la, ra, _ = A.forward(x)
    if report_target is not None:
        ra = np.full_like(ra, report_target)
    model = init if init is not None else cls()
    xin = model_input(model, x)
    la_t, ra_t = torch.from_numpy(la.astype(np.float32)), torch.from_numpy(ra.astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lm, rm, _ = model(xin[idx])
            loss = bc_loss(lm, rm, la_t[idx], ra_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose:
            print(ep, tot / n)
    model.eval()
    return model


# ---------------------------------------------------------------- unified call
def run_agent(agent, x, steer=None):
    """Returns numpy (logit, report, activations)."""
    if isinstance(agent, ReferenceAgent):
        return agent.forward(x, steer=steer)
    with torch.no_grad():
        st = None if steer is None else torch.as_tensor(steer, dtype=torch.float32)
        l, r, h = agent(model_input(agent, x), steer=st)
    return l.numpy(), r.numpy(), h.numpy()


def report_gradient_direction(agent, x):
    """Mean gradient of the report w.r.t. activations (the direction that most
    efficiently changes the report)."""
    if isinstance(agent, ReferenceAgent):
        return agent.R[:, 0].copy()
    xin = model_input(agent, x)
    hdn = agent.hidden(xin).detach().requires_grad_(True)
    _, r = agent.heads(hdn, xin)
    r.sum().backward()
    g = hdn.grad.reshape(-1, hdn.shape[-1]).mean(0).numpy()
    return g
