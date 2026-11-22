"""Zipfian multi-skill testbed for studying how far distillation can compress capabilities.
Each 'skill' t is a stochastic C-way classification rule p(y|t,z)=softmax(beta*W_t z).
Skills occur with Zipf frequency p_t ∝ t^-(1+alpha).
Families: 'random' (W_t i.i.d. Gaussian: incompressible) and 'lowrank' (W_t = sum_k c_tk B_k, r shared bases: compressible).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)

class World:
    def __init__(self, T=400, d=4, C=4, alpha=0.5, beta=2.0, family='random', rank=2, seed=0, m=32):
        g = np.random.default_rng(seed)
        self.T, self.d, self.C, self.alpha, self.beta = T, d, C, alpha, beta
        if family == 'random':
            W = g.standard_normal((T, C, d))
        elif family == 'lowrank':
            B = g.standard_normal((rank, C, d))
            c = g.standard_normal((T, rank)) / np.sqrt(rank)
            W = np.einsum('tk,kcd->tcd', c, B)
        else:
            raise ValueError(family)
        self.W = torch.tensor(W, dtype=torch.float32)
        # fixed (non-learned) random skill codes: all student parameters are shared across skills
        self.code = torch.tensor(g.standard_normal((T, m)) / np.sqrt(m), dtype=torch.float32)
        self.m = m
        f = np.arange(1, T + 1) ** (-(1 + alpha)); self.p = f / f.sum()
        self.p_t = torch.tensor(self.p, dtype=torch.float32)
        # fixed test set: n_test inputs per skill
        gt = torch.Generator().manual_seed(10_000 + seed)
        self.test_t = torch.arange(T).repeat_interleave(200)
        self.test_z = torch.randn(len(self.test_t), d, generator=gt)
        self.test_logp = self.true_logp(self.test_t, self.test_z)

    def true_logp(self, t, z):
        return F.log_softmax(self.beta * torch.einsum('ncd,nd->nc', self.W[t], z), -1)

    def sample(self, n, gen):
        t = torch.multinomial(self.p_t, n, replacement=True, generator=gen)
        z = torch.randn(n, self.d, generator=gen)
        logp = self.true_logp(t, z)
        y = torch.multinomial(logp.exp(), 1, generator=gen).squeeze(1)
        return t, z, y, logp

class MLP(nn.Module):
    def __init__(self, code, d, C, h, depth=2):
        super().__init__()
        self.register_buffer('code', code)     # fixed skill codes (not parameters)
        self.emb = nn.Linear(code.shape[1], h, bias=False)
        self.inp = nn.Linear(d, h)
        layers = []
        for _ in range(depth - 1): layers += [nn.Linear(h, h)]
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(h, C)
    def forward(self, t, z):
        x = F.relu(self.emb(self.code[t]) + self.inp(z))
        for l in self.hidden: x = F.relu(l(x))
        return self.out(x)

def n_params(m): return sum(p.numel() for p in m.parameters())  # buffers (codes) excluded

def evaluate(world, model):
    with torch.no_grad():
        lq = F.log_softmax(model(world.test_t, world.test_z), -1)
        P = world.test_logp.exp()
        kl = (P * (world.test_logp - lq)).sum(-1)
        # null model: uniform prediction
        kl0 = (P * (world.test_logp + np.log(world.C))).sum(-1)
        agree = (lq.argmax(-1) == world.test_logp.argmax(-1)).float()
    T = world.T
    kl_t = kl.view(T, -1).mean(1).numpy(); kl0_t = kl0.view(T, -1).mean(1).numpy()
    acc_t = agree.view(T, -1).mean(1).numpy()
    score_t = np.clip(1 - kl_t / kl0_t, 0, 1)
    return dict(kl_t=kl_t, kl0_t=kl0_t, acc_t=acc_t, score_t=score_t)

def train(world, h, D, method, teacher=None, steps=15000, bs=512, lr=1e-2, seed=0, depth=2, eval_every=500):
    """method: 'hard' (sampled labels), 'soft' (teacher distribution; oracle if teacher None).
    D: number of distinct training inputs (None = fresh samples every step, i.e. unlimited queries).
    Early stopping on a held-out validation set of the SAME label type (size max(D//4, 2000))."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = MLP(world.code, world.d, world.C, h, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    def targets(t, z, y, logp):
        if method == 'hard': return y
        if teacher is None: return logp.exp()
        with torch.no_grad(): return F.softmax(teacher(t, z), -1)
    def loss_fn(out, tgt):
        if method == 'hard': return F.cross_entropy(out, tgt)
        return -(tgt * F.log_softmax(out, -1)).sum(-1).mean()
    if D is not None:
        t, z, y, logp = world.sample(D, gen); tgt = targets(t, z, y, logp)
        nv = max(D // 4, 2000)
        vt, vz, vy, vlp = world.sample(nv, gen); vtgt = targets(vt, vz, vy, vlp)
    best, best_state = 1e9, None
    for s in range(steps):
        if D is None:
            bt, bz, by, blp = world.sample(bs, gen); btgt = targets(bt, bz, by, blp)
        else:
            idx = torch.randint(0, D, (min(bs, D),), generator=gen)
            bt, bz, btgt = t[idx], z[idx], tgt[idx]
        loss = loss_fn(model(bt, bz), btgt)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if D is not None and ((s + 1) % eval_every == 0 or s == steps - 1):
            with torch.no_grad(): vl = loss_fn(model(vt, vz), vtgt).item()
            if vl < best:
                best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model
