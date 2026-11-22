"""Experiment B: trained single-layer sequence mixers on pre-bound multi-query associative recall.

Each sequence: N (key,value) pair tokens with distinct keys from a key vocabulary of size K (K >> d_k,
so keys must be stored in superposition), followed by Q query tokens (keys drawn from the stored set).
Pair token embedding = [E_k(key); E_v(value)], query embedding = [E_k(key); 0].
Writes are masked at query positions for every architecture (isolates memory capacity).

Mixers (state of size d_k x d_v per sequence, except softmax attention whose KV cache grows with N):
  attn   : causal softmax attention over pair positions
  linear : o_t = sum_s (k_s . q_t) v_s                 (additive outer-product state)
  gla    : scalar data-dependent decay a_t in (0,1]     (Mamba-2 / GLA-like)
  delta  : DeltaNet, beta_t in (0,1)                      (parallel UT form)
  gdelta : Gated DeltaNet (decay + delta)
q,k are L2-normalised. Readout: logits = W_o LayerNorm(o_t).
Usage: python train_mqar.py --arch linear --dk 16 --dv 64 [--steps 2000]
Output: results/train/<arch>_dk<dk>_dv<dv>.json
"""
import argparse, json, os, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--arch", default="linear"); p.add_argument("--dk", type=int, default=16)
p.add_argument("--dv", type=int, default=64); p.add_argument("--steps", type=int, default=2000)
p.add_argument("--K", type=int, default=4096); p.add_argument("--V", type=int, default=512)
p.add_argument("--Q", type=int, default=32); p.add_argument("--bs", type=int, default=64)
p.add_argument("--Nmax", type=int, default=256); p.add_argument("--seed", type=int, default=0)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--d", type=int, default=128)  # token embedding width; the value half (d/2) caps the effective value dimension
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
RES = os.path.join(os.path.dirname(__file__), "..", "results", "train"); os.makedirs(RES, exist_ok=True)
NS = [4, 8, 16, 32, 64, 128, 256]


def batch(N, bs, gen):
    keys = torch.stack([torch.randperm(a.K, generator=gen)[:N] for _ in range(bs)])
    vals = torch.randint(0, a.V, (bs, N), generator=gen)
    qi = torch.randint(0, N, (bs, a.Q), generator=gen)
    qk = torch.gather(keys, 1, qi); qv = torch.gather(vals, 1, qi)
    return keys, vals, qk, qv


class Mixer(nn.Module):
    def __init__(s, arch, dk, dv, d=128):
        super().__init__()
        s.arch, s.dk, s.dv = arch, dk, dv
        s.Ek = nn.Embedding(a.K, d // 2); s.Ev = nn.Embedding(a.V, d // 2)
        s.Wq = nn.Linear(d, dk, bias=False); s.Wk = nn.Linear(d, dk, bias=False); s.Wv = nn.Linear(d, dv, bias=False)
        s.Wa = nn.Linear(d, 1); s.Wb = nn.Linear(d, 1)
        nn.init.constant_(s.Wa.bias, 3.0)   # start near no-decay
        s.temp = nn.Parameter(torch.tensor(math.log(10.0)))
        s.ln = nn.LayerNorm(dv); s.Wo = nn.Linear(dv, a.V)

    def forward(s, keys, vals, qk):
        B, N = keys.shape; Q = qk.shape[1]; T = N + Q
        xp = torch.cat([s.Ek(keys), s.Ev(vals)], -1)
        xq = torch.cat([s.Ek(qk), torch.zeros(B, Q, s.Ev.embedding_dim)], -1)
        x = torch.cat([xp, xq], 1)                                   # B,T,d
        write = torch.cat([torch.ones(B, N), torch.zeros(B, Q)], 1)  # B,T
        q = F.normalize(s.Wq(x), dim=-1); k = F.normalize(s.Wk(x), dim=-1); v = s.Wv(x)
        causal = torch.tril(torch.ones(T, T))
        A = q @ k.transpose(1, 2)                                    # B,T,T  (t,s): q_t . k_s
        Wmask = causal * write[:, None, :]
        if s.arch == "attn":
            logits = A * s.temp.exp()
            logits = logits.masked_fill(Wmask == 0, -1e9)
            att = torch.softmax(logits, -1) * (Wmask.sum(-1, keepdim=True) > 0)
            o = att @ v
        else:
            if s.arch in ("gla", "gdelta"):
                la = F.logsigmoid(s.Wa(x)).squeeze(-1) * write        # log decay, 0 at query positions
                c = torch.cumsum(la, 1)
                D = torch.exp((c[:, :, None] - c[:, None, :]).clamp(max=0)) * causal  # D[t,s]=prod_{r=s+1..t} a_r
            else:
                D = causal.expand(B, T, T)
            if s.arch in ("linear", "gla"):
                o = (A * D * write[:, None, :]) @ v
            else:  # delta / gated delta: u = (I + diag(beta) Lstrict)^{-1} diag(beta) v
                beta = torch.sigmoid(s.Wb(x)).squeeze(-1) * write
                KK = (k @ k.transpose(1, 2)) * D * torch.tril(torch.ones(T, T), -1)
                Lm = torch.eye(T) + beta[:, :, None] * KK
                u = torch.linalg.solve_triangular(Lm, beta[:, :, None] * v, upper=False, unitriangular=True)
                o = (A * D) @ u
        return s.Wo(s.ln(o[:, N:]))


m = Mixer(a.arch, a.dk, a.dv, d=a.d)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.steps, pct_start=0.1)
gen = torch.Generator().manual_seed(a.seed)
train_Ns = [n for n in NS if n <= a.Nmax]
t0 = time.time(); losses = []
for step in range(a.steps):
    N = int(np.random.choice(train_Ns))
    keys, vals, qk, qv = batch(N, a.bs, gen)
    loss = F.cross_entropy(m(keys, vals, qk).reshape(-1, a.V), qv.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    losses.append(loss.item())
    if step % 250 == 0:
        print(a.arch, a.dk, step, round(np.mean(losses[-50:]), 3), round(time.time() - t0, 1), flush=True)

m.eval(); evalgen = torch.Generator().manual_seed(1234); acc = {}
with torch.no_grad():
    for N in NS:
        c = 0; n = 0
        for _ in range(8):
            keys, vals, qk, qv = batch(N, 128, evalgen)
            c += (m(keys, vals, qk).argmax(-1) == qv).sum().item(); n += qv.numel()
        acc[N] = c / n
with torch.no_grad():  # learned write strength beta and decay a on pair tokens (N=64 batch)
    keys, vals, qk, qv = batch(64, 128, evalgen)
    xp = torch.cat([m.Ek(keys), m.Ev(vals)], -1)
    gates = {"beta_mean": torch.sigmoid(m.Wb(xp)).mean().item(), "decay_mean": torch.sigmoid(m.Wa(xp)).mean().item(),
             "decay_min": torch.sigmoid(m.Wa(xp)).min().item()}
res = {"args": vars(a), "acc": acc, "gates_on_pair_tokens": gates, "train_time_s": time.time() - t0, "final_loss": float(np.mean(losses[-100:])),
       "state_size": a.dk * a.dv}
json.dump(res, open(os.path.join(RES, f"{a.arch}_dk{a.dk}_dv{a.dv}" + (f"_d{a.d}" if a.d != 128 else "") + ".json"), "w"), indent=1)
print(res)
