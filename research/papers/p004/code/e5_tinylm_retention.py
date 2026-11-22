"""E5: Does a small neural LM obey the per-exposure retention model of Prop. 2?

A 2-layer causal transformer (CPU, PyTorch) is trained with next-token loss on a
synthetic corpus of planted facts: (person, attribute) -> value, value uniform
over V options. Each corpus sentence states one fact in one of T surface
templates (different filler tokens and word orders); the value is always the
last token. Fact i appears Poisson(N p_i) times, p Zipf(alpha) over facts.

Measured:
 (1) recall of each fact (argmax over the V values, averaged over templates) as a
     function of its exposure count c in the plain corpus -> fit the geometric
     storage law 1-(1-s)^c (Prop. 2's assumption) against threshold laws
     ("stored iff >= m of c Bernoulli(s) trials succeed").
 (2) plain learning curve A(N); amplified corpora at base size N0 where every
     human sentence is accompanied by r extra copies, either in OTHER templates
     ("paraphrase") or in the SAME template ("repetition"); multiplier
     M = N_eff / N0 by inverting A(N).
 (3) invariance: Prop. 2 says M(r) does not depend on N0 or alpha.
All training is a fixed number of epochs E over whatever corpus is given.
"""
import json, os, sys, time, math, numpy as np, torch, torch.nn as nn
from concurrent.futures import ProcessPoolExecutor

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
NP, NA, V, T = 1000, 2, 16, 8           # persons, attributes, values/attribute, templates
NF = NP * NA                            # number of facts
E = 4                                   # epochs over the corpus
BATCH, LR, DM = 64, 2e-3, 128

# vocab layout
BOS = 0
PERS0 = 1
ATTR0 = PERS0 + NP
VAL0 = ATTR0 + NA                       # attribute a uses VAL0 + a*V .. +V
FILL0 = VAL0 + NA * V                   # 3 filler tokens per template
VOCAB = FILL0 + 3 * T
L = 7

def sentence(person, attr, val, t):
    f = [FILL0 + 3 * t + j for j in range(3)]
    if t % 2 == 0:
        mid = [f[0], PERS0 + person, f[1], ATTR0 + attr, f[2]]
    else:
        mid = [f[0], ATTR0 + attr, f[1], PERS0 + person, f[2]]
    return [BOS] + mid + [VAL0 + attr * V + val]

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DM); self.pos = nn.Embedding(L, DM)
        layer = nn.TransformerEncoderLayer(DM, 4, 4 * DM, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.ln = nn.LayerNorm(DM); self.head = nn.Linear(DM, VOCAB)
        self.register_buffer("mask", torch.triu(torch.full((L - 1, L - 1), float("-inf")), 1))
    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        return self.head(self.ln(self.tr(h, mask=self.mask[:x.shape[1], :x.shape[1]], is_causal=True)))

def make_world(seed, alpha):
    r = np.random.default_rng(seed)
    vals = r.integers(0, V, size=(NP, NA))
    w = np.arange(1, NF + 1, dtype=float) ** (-alpha); w /= w.sum()
    p = w[r.permutation(NF)]            # fact f = person*NA + attr
    return vals, p

def build_corpus(vals, p, N, r_extra, kind, seed):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(N * p)
    rows = []
    for f in np.nonzero(counts)[0]:
        pe, at = divmod(int(f), NA)
        for _ in range(counts[f]):
            t0 = int(rng.integers(T))
            rows.append(sentence(pe, at, vals[pe, at], t0))
            if r_extra:
                if kind == "paraphrase":
                    others = rng.permutation([t for t in range(T) if t != t0])
                    ts = [int(others[j % (T - 1)]) for j in range(r_extra)]
                else:
                    ts = [t0] * r_extra
                for t in ts:
                    rows.append(sentence(pe, at, vals[pe, at], t))
    return np.array(rows, dtype=np.int64), counts

def evaluate(model, vals):
    model.eval()
    correct = np.zeros((NF, T))
    with torch.no_grad():
        for t in range(T):
            X = torch.tensor([sentence(pe, at, vals[pe, at], t) for pe in range(NP) for at in range(NA)])
            logits = model(X[:, :-1])[:, -1, :]
            for at in range(NA):
                idx = torch.arange(NF)[at::NA]
                sub = logits[idx][:, VAL0 + at * V: VAL0 + (at + 1) * V]
                pred = sub.argmax(1).numpy()
                correct[idx.numpy(), t] = pred == vals[idx.numpy() // NA, at]
    return correct

def train_eval(args):
    N, r_extra, kind, seed, alpha, world_seed = args
    torch.set_num_threads(1); torch.manual_seed(seed)
    vals, p = make_world(world_seed, alpha)
    data, counts = build_corpus(vals, p, N, r_extra, kind, seed * 7919 + N + 13 * r_extra)
    model = TinyLM(); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    D = torch.tensor(data)
    steps = 0
    for ep in range(E):
        perm = torch.randperm(len(D), generator=g)
        for i in range(0, len(D), BATCH):
            xb = D[perm[i:i + BATCH]]
            logits = model(xb[:, :-1])
            loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); steps += 1
    corr = evaluate(model, vals)
    fact_acc = corr.mean(1)
    return dict(N=N, r=r_extra, kind=kind, seed=seed, alpha=alpha, steps=steps, n_rows=len(data),
                acc_weighted=float(np.sum(p * fact_acc)), acc_uniform=float(fact_acc.mean()),
                counts=counts.tolist(), fact_acc=np.round(fact_acc, 3).tolist())

if __name__ == "__main__":
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    SEEDS = [0, 1, 2]
    jobs = []
    ALPHA = 1.0
    if mode == "smoke":
        jobs = [(4000, 0, "plain", 0, ALPHA, 0)]
    else:
        for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            for s in SEEDS: jobs.append((N, 0, "plain", s, ALPHA, s))
        for N0 in [2000, 8000]:
            for r in [1, 3, 7]:
                for kind in ["paraphrase", "repetition"]:
                    if N0 == 8000 and r == 7: continue
                    for s in SEEDS: jobs.append((N0, r, kind, s, ALPHA, s))
        # alpha invariance: second frequency distribution
        for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            for s in SEEDS: jobs.append((N, 0, "plain", s, 0.6, s))
        for r in [1, 3]:
            for s in SEEDS: jobs.append((2000, r, "paraphrase", s, 0.6, s))
    jobs.sort(key=lambda j: -j[0] * (1 + j[1]))
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(train_eval, jobs))
    for x in sorted(res, key=lambda x: (x["alpha"], x["kind"], x["N"], x["r"], x["seed"])):
        print(x["alpha"], x["kind"], x["N"], x["r"], x["seed"], x["steps"], round(x["acc_weighted"], 4), round(x["acc_uniform"], 4))
    if mode != "smoke":
        json.dump(dict(runs=res, config=dict(NP=NP, NA=NA, V=V, T=T, E=E, BATCH=BATCH, LR=LR, DM=DM)),
                  open(os.path.join(OUT, "e5_tinylm.json"), "w"))
    print("done %.1fs" % (time.time() - t0))
