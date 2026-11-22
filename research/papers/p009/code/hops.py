"""k-hop in-context pointer chasing for hybrid (DeltaNet / softmax-attention) stacks.

Task: a sequence of N "pair tokens" (x -> pi(x)) for a random permutation pi of N
symbols drawn from a vocabulary of V symbols, in random order, followed by Q query
tokens x. The target at each query is pi^k(x). Loss/accuracy on query positions only.

Each pair is a single token (embedding of key + embedding of value) so that one
retrieval = one lookup, with no need for previous-token/induction circuitry.

Usage: python hops.py --layers DDAA --k 2 --N 32 --steps 4000 --seed 0 --out ../results/runs
Layer codes: A = causal softmax attention, D = DeltaNet (delta-rule fixed-state),
             L = plain linear attention (outer-product fixed state).
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


def make_batch(B, N, V, Q, K, gen):
    """Each query carries its own hop count h in {1..K}; target = pi^h(x)."""
    # symbols: N distinct from V
    sym = torch.argsort(torch.rand(B, V, generator=gen), dim=1)[:, :N]          # (B,N)
    perm = torch.argsort(torch.rand(B, N, generator=gen), dim=1)                # pi on indices
    val = torch.gather(sym, 1, perm)                                            # pi(sym[i]) = sym[perm[i]]
    order = torch.argsort(torch.rand(B, N, generator=gen), dim=1)
    keys = torch.gather(sym, 1, order)
    vals = torch.gather(val, 1, order)
    qi = torch.randint(0, N, (B, Q), generator=gen)                             # index into sym
    qsym = torch.gather(sym, 1, qi)
    hops = torch.randint(1, K + 1, (B, Q), generator=gen)
    t = qi
    for h in range(1, K + 1):
        nt = torch.gather(perm, 1, t)
        t = torch.where(hops >= h, nt, t)
    tgt = torch.gather(sym, 1, t)
    return keys, vals, qsym, hops, tgt


class Attn(nn.Module):
    def __init__(s, d, H):
        super().__init__(); s.H = H; s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)

    def forward(s, x):
        B, T, d = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.H, d // s.H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, d))


class Delta(nn.Module):
    """DeltaNet: S_t = S_{t-1}(I - b k k^T) + b v k^T, o_t = S_t q_t (k,q L2-normalised).
    delta=False gives plain (normalised-key) linear attention S_t = S_{t-1} + v k^T."""
    def __init__(s, d, H, dk, delta=True):
        super().__init__(); s.H, s.dk, s.dv, s.delta = H, dk, d // H, delta
        s.q = nn.Linear(d, H * dk, bias=False); s.k = nn.Linear(d, H * dk, bias=False)
        s.v = nn.Linear(d, d, bias=False); s.b = nn.Linear(d, H); s.o = nn.Linear(d, d, bias=False)
        s.loop = False

    def _loop(s, q, k, v, b):
        B, T = q.shape[:2]
        S = q.new_zeros(B, s.H, s.dv, s.dk); outs = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            pred = torch.einsum('bhvk,bhk->bhv', S, kt) if s.delta else 0
            S = S + torch.einsum('bhv,bhk->bhvk', b[:, t] * (vt - pred), kt)
            outs.append(torch.einsum('bhvk,bhk->bhv', S, qt))
        return torch.stack(outs, 1)

    def forward(s, x):
        B, T, d = x.shape
        q = F.normalize(s.q(x).view(B, T, s.H, s.dk), dim=-1)
        k = F.normalize(s.k(x).view(B, T, s.H, s.dk), dim=-1)
        v = s.v(x).view(B, T, s.H, s.dv)
        b = torch.sigmoid(s.b(x))                                                # (B,T,H)
        if s.loop:
            return s.o(s._loop(q, k, v, b.unsqueeze(-1)).reshape(B, T, d))
        q, k, v, b = (z.transpose(1, 2) for z in (q, k, v, b))                    # (B,H,T,.)
        bv = b.unsqueeze(-1) * v
        if s.delta:
            # exact parallel form of the delta rule (single chunk, S_0 = 0):
            # (I + tril(diag(b) K K^T, -1)) U = diag(b) V ;  O = tril(Q K^T) U
            A = torch.tril(b.unsqueeze(-1) * (k @ k.transpose(-1, -2)), -1)
            I = torch.eye(T, dtype=x.dtype).expand_as(A)
            U = torch.linalg.solve_triangular(I + A, bv, upper=False, unitriangular=True)
        else:
            U = bv
        y = torch.tril(q @ k.transpose(-1, -2)) @ U
        y = y.transpose(1, 2).reshape(B, T, d)
        return s.o(y)


class Block(nn.Module):
    def __init__(s, code, d, H, dk):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.mix = Attn(d, H) if code == 'A' else Delta(d, H, dk, delta=(code == 'D'))
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x):
        x = x + s.mix(s.n1(x))
        return x + s.mlp(s.n2(x))


class Model(nn.Module):
    def __init__(s, layers, V, d, H, dk):
        super().__init__()
        s.ek, s.ev, s.eq = nn.Embedding(V, d), nn.Embedding(V, d), nn.Embedding(V, d)
        s.eh = nn.Embedding(8, d)
        s.blocks = nn.ModuleList([Block(c, d, H, dk) for c in layers])
        s.nf = nn.LayerNorm(d); s.head = nn.Linear(d, V)

    def forward(s, keys, vals, qsym, hops):
        x = torch.cat([s.ek(keys) + s.ev(vals), s.eq(qsym) + s.eh(hops)], 1)
        for b in s.blocks:
            x = b(x)
        return s.head(s.nf(x[:, keys.shape[1]:]))


def state_floats(layers, d, H, dk):
    return sum(d * dk for c in layers if c in 'DL')  # H heads * dv * dk = d*dk per layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', default='AA'); p.add_argument('--k', type=int, default=1, help='max hop count K (queries mix 1..K)')
    p.add_argument('--N', type=int, default=32); p.add_argument('--V', type=int, default=64)
    p.add_argument('--Q', type=int, default=8); p.add_argument('--d', type=int, default=64)
    p.add_argument('--H', type=int, default=4); p.add_argument('--dk', type=int, default=8)
    p.add_argument('--steps', type=int, default=4000); p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='../results/runs'); p.add_argument('--tag', default='')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(1000 + a.seed)
    m = Model(a.layers, a.V, a.d, a.H, a.dk)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time(); curve = []
    for step in range(a.steps):
        keys, vals, qs, hp, tgt = make_batch(a.bs, a.N, a.V, a.Q, a.k, gen)
        logits = m(keys, vals, qs, hp)
        loss = F.cross_entropy(logits.reshape(-1, a.V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            curve.append((step, loss.item()))
    # evaluation on fresh data
    m.eval(); eg = torch.Generator().manual_seed(999999 + a.seed)
    corr = torch.zeros(a.k + 1); tot = torch.zeros(a.k + 1)
    with torch.no_grad():
        for _ in range(16):
            keys, vals, qs, hp, tgt = make_batch(256, a.N, a.V, a.Q, a.k, eg)
            ok = (m(keys, vals, qs, hp).argmax(-1) == tgt).float()
            for h in range(1, a.k + 1):
                corr[h] += ok[hp == h].sum(); tot[h] += (hp == h).sum()
    acc_by_hop = {h: (corr[h] / tot[h]).item() for h in range(1, a.k + 1)}
    res = dict(vars(a), acc=(corr.sum() / tot.sum()).item(), acc_by_hop=acc_by_hop, final_loss=curve[-1][1], curve=curve, secs=time.time() - t0,
               n_attn=a.layers.count('A'), n_rec=sum(c in 'DL' for c in a.layers),
               state_floats=state_floats(a.layers, a.d, a.H, a.dk),
               params=sum(x.numel() for x in m.parameters()))
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.tag}{a.layers}_k{a.k}_N{a.N}_dk{a.dk}_s{a.seed}.json"
    json.dump(res, open(os.path.join(a.out, name), 'w'))
    print(name, {h: round(v, 3) for h, v in acc_by_hop.items()}, f"acc={res['acc']:.3f} loss={res['final_loss']:.3f} secs={res['secs']:.0f}")


if __name__ == '__main__':
    main()
