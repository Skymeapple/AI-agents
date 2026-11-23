"""
Experiment 1: Do candidate AI identity/continuity criteria agree?
Train tiny decoder-only transformers on a synthetic "language" (order-2 Markov chain A),
apply realistic lineage operations (continued training, fine-tuning, pruning, quantisation,
neuron permutation, distillation, merging, re-training, ...), and measure the similarity of
each successor to its origin under weight, behavioural and representational criteria.
All numbers in the paper's Section 5 come from results/lineage_metrics.json written here.
Run:  python3 lineage_experiment.py   (CPU, ~5-10 min)
"""
import json, math, time, copy, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
V, T = 24, 32           # vocab, context length
D, H, L, FF = 64, 4, 2, 256
STEPS_BASE = int(os.environ.get('STEPS_BASE', 1500))
BATCH = 64
SEEDS = [0, 1, 2]

# ---------------- synthetic languages -----------------
def make_chain(seed, alpha=0.15):
    r = np.random.default_rng(seed)
    P = r.dirichlet(alpha * np.ones(V), size=(V, V))   # P[a,b] = dist over next token
    return P

CHAINS = {'A': make_chain(1001), 'B': make_chain(2002), 'C': make_chain(3003)}

def sample(chain, n, rng):
    P = CHAINS[chain]
    x = np.zeros((n, T + 1), dtype=np.int64)
    x[:, 0] = rng.integers(0, V, n); x[:, 1] = rng.integers(0, V, n)
    cum = P.cumsum(-1)
    for t in range(2, T + 1):
        u = rng.random(n)[:, None]
        c = cum[x[:, t - 2], x[:, t - 1]]
        x[:, t] = np.minimum((u > c).sum(-1), V - 1)
    return torch.from_numpy(x)

# ---------------- model -----------------
class Block(nn.Module):
    def __init__(s, d=D, ff=FF):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
        s.fc1 = nn.Linear(d, ff); s.fc2 = nn.Linear(ff, d); s.d = d
        s.register_buffer('mask', torch.tril(torch.ones(T, T)).bool())
    def forward(s, x):
        B_, T_, d = x.shape; h = s.ln1(x)
        q, k, v = s.qkv(h).view(B_, T_, 3, H, d // H).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(d // H)
        a = a.masked_fill(~s.mask[:T_, :T_], -1e9).softmax(-1)
        x = x + s.proj((a @ v).transpose(1, 2).reshape(B_, T_, d))
        return x + s.fc2(F.gelu(s.fc1(s.ln2(x))))

class TinyLM(nn.Module):
    def __init__(s, d=D, ff=FF, nl=L):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.zeros(T, d))
        s.blocks = nn.ModuleList([Block(d, ff) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, V)
    def forward(s, x, return_h=False):
        h = s.emb(x) + s.pos[:x.shape[1]]
        for b in s.blocks: h = b(h)
        h = s.lnf(h); lo = s.head(h)
        return (lo, h) if return_h else lo

def new_model(seed, **kw):
    torch.manual_seed(seed); return TinyLM(**kw)

def train(model, chain, steps, seed, lr=3e-3, teacher=None, mix=None):
    rng = np.random.default_rng(seed + 777)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for it in range(steps):
        if mix is None: x = sample(chain, BATCH, rng)
        else: x = torch.cat([sample(c, BATCH // len(mix), rng) for c in mix])
        inp, tgt = x[:, :-1], x[:, 1:]
        lo = model(inp)
        if teacher is None:
            loss = F.cross_entropy(lo.reshape(-1, V), tgt.reshape(-1))
        else:
            with torch.no_grad(): tp = teacher(inp).softmax(-1)
            loss = F.kl_div(lo.log_softmax(-1), tp, reduction='batchmean') / T
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); return model

# ---------------- operations -----------------
def op_permute(m, seed):
    m = copy.deepcopy(m); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for b in m.blocks:
            p = torch.randperm(b.fc1.out_features, generator=g)
            b.fc1.weight.copy_(b.fc1.weight[p]); b.fc1.bias.copy_(b.fc1.bias[p])
            b.fc2.weight.copy_(b.fc2.weight[:, p])
    return m

def op_noise(m, seed, rel=0.05):
    m = copy.deepcopy(m); torch.manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters(): p.add_(rel * p.std() * torch.randn_like(p))
    return m

def op_prune(m, frac=0.3):
    m = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2 and 'emb' not in n:
                thr = p.abs().flatten().kthvalue(int(frac * p.numel())).values
                p.mul_((p.abs() > thr).float())
    return m

def op_quant(m, bits=4):
    m = copy.deepcopy(m)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                s = p.abs().max() / (2 ** (bits - 1) - 1)
                p.copy_(torch.round(p / s) * s)
    return m

def op_merge(a, b, w=0.5):
    m = copy.deepcopy(a); sb = dict(b.named_parameters())
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(w * p + (1 - w) * sb[n])
    return m

# ---------------- metrics -----------------
def flat(m): return torch.cat([p.detach().flatten() for p in m.parameters()])

def align_ffn(ref, m):
    """Weight matching (Ainsworth et al. 2023 style) restricted to FFN hidden units."""
    m = copy.deepcopy(m)
    with torch.no_grad():
        for br, bm in zip(ref.blocks, m.blocks):
            Wr = torch.cat([br.fc1.weight, br.fc1.bias[:, None], br.fc2.weight.T], 1)
            Wm = torch.cat([bm.fc1.weight, bm.fc1.bias[:, None], bm.fc2.weight.T], 1)
            _, col = linear_sum_assignment(-(Wr @ Wm.T).numpy())
            p = torch.as_tensor(col)
            bm.fc1.weight.copy_(bm.fc1.weight[p]); bm.fc1.bias.copy_(bm.fc1.bias[p])
            bm.fc2.weight.copy_(bm.fc2.weight[:, p])
    return m

def rel_l2(a, b):
    fa, fb = flat(a), flat(b); return float((fa - fb).norm() / fa.norm())

def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))

EVAL = {}
def eval_sets():
    rng = np.random.default_rng(99)
    for c in 'ABC': EVAL[c] = sample(c, 256, rng)[:, :-1]

@torch.no_grad()
def behav(a, b, c):
    x = EVAL[c]; pa = a(x).softmax(-1); pb = b(x).softmax(-1); m = 0.5 * (pa + pb)
    js = 0.5 * ((pa * (pa / m).log()).sum(-1) + (pb * (pb / m).log()).sum(-1))
    agree = (pa.argmax(-1) == pb.argmax(-1)).float().mean()
    return float(js.mean() / math.log(2)), float(agree)

@torch.no_grad()
def nll(m, c):
    x = sample(c, 256, np.random.default_rng(5))
    return float(F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1)))

@torch.no_grad()
def rep(a, b):
    x = EVAL['A'][:64]
    _, ha = a(x, return_h=True); _, hb = b(x, return_h=True)
    return cka(ha.reshape(-1, ha.shape[-1]), hb.reshape(-1, hb.shape[-1]))

def metrics(base, m, same_arch=True):
    r = {}
    if same_arch:
        r['w_raw'] = rel_l2(base, m); r['w_aligned'] = rel_l2(base, align_ffn(base, m))
    else:
        r['w_raw'] = None; r['w_aligned'] = None
    for c in 'ABC':
        js, ag = behav(base, m, c); r[f'js_{c}'] = js; r[f'agree_{c}'] = ag
    r['cka'] = rep(base, m)
    r['nll_A'] = nll(m, 'A'); r['nll_B'] = nll(m, 'B')
    return r

# ---------------- main -----------------
def main():
    eval_sets(); t0 = time.time(); allres = {}
    for s in SEEDS:
        print(f'== seed {s}', flush=True)
        base = train(new_model(s), 'A', STEPS_BASE, s)
        ops = {}
        ops['identity_copy'] = copy.deepcopy(base)
        ops['continue_A'] = train(copy.deepcopy(base), 'A', 300, s + 10, lr=1e-3)
        ops['permute_ffn'] = op_permute(base, s)
        ops['noise_5pct'] = op_noise(base, s)
        ops['quant_4bit'] = op_quant(base)
        ops['prune_30pct'] = op_prune(base)
        ops['finetune_B_short'] = train(copy.deepcopy(base), 'B', 100, s + 20, lr=1e-3)
        ops['finetune_B_long'] = train(copy.deepcopy(base), 'B', 800, s + 30, lr=1e-3)
        ops['merge_base_ftB'] = op_merge(base, ops['finetune_B_long'])
        ops['distill_newinit'] = train(new_model(s + 100), 'A', STEPS_BASE, s + 40, teacher=base)
        ops['retrain_newseed'] = train(new_model(s + 200), 'A', STEPS_BASE, s + 50)
        ops['stranger_B'] = train(new_model(s + 300), 'B', STEPS_BASE, s + 60)
        small = train(new_model(s + 400, d=32, ff=128), 'A', STEPS_BASE, s + 70, teacher=base)
        res = {k: metrics(base, m) for k, m in ops.items()}
        res['distill_small_arch'] = metrics(base, small, same_arch=False)
        res['_base'] = {'nll_A': nll(base, 'A'), 'nll_B': nll(base, 'B')}
        allres[s] = res
        print(f'  done seed {s} at {time.time()-t0:.0f}s', flush=True)
    # entropy-rate reference for the chains
    ent = {}
    for c, P in CHAINS.items():
        H_ = -(P * np.log(P + 1e-30)).sum(-1)
        ent[c] = float(H_.mean())
    allres['_chain_entropy_uniformcontext'] = ent
    allres['_config'] = dict(V=V, T=T, D=D, H=H, L=L, FF=FF, STEPS_BASE=STEPS_BASE, BATCH=BATCH,
                             SEEDS=SEEDS, runtime_s=time.time() - t0)
    json.dump(allres, open(os.path.join(OUT, 'lineage_metrics.json'), 'w'), indent=1)
    print('total', time.time() - t0)

if __name__ == '__main__':
    main()
