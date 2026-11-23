"""E3b (revision r1): age-0 revision cost in trained single heads vs. the Proposition-4 ceiling.

Proposition 4 (token-and-age heads): if writes of the queried key get a logit that depends only on
the write token and its age, all logits lie in [-B, B], never-revised facts at ages g, 2g, ..., Mg are
retrieved with weight odds o_c(jg) = alpha/(1-alpha), then in the history whose latest version is at
age 0 and whose M older versions sit at ages g, ..., Mg the weight odds on the latest satisfy
      o_h <= exp(2B) / [ (T-1) * sum_j o_c(jg) + (T-1-M) ].
We evaluate saved clamped-logit models (B finite): measured cold odds at each age jg (n streams per
age), the plug-in ceiling, and the measured weight on the latest version at age 0 (evenly spaced
older versions). Writes results/e3_prop4.json.
"""
import torch, numpy as np, json, os, glob, sys, math
sys.argv = [sys.argv[0]]                      # e3_single_head parses argv at import
import e3_single_head as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
K, V = E.K, E.V


def streams(n, N, pos_list, rng):
    q = rng.integers(0, K, size=n)
    other = rng.integers(0, K - 1, size=(n, N)); keys = other + (other >= q[:, None])
    vals = rng.integers(0, V, size=(n, N))
    keys[:, pos_list] = q[:, None]
    return torch.tensor(np.concatenate([keys * V + vals, (K * V + q)[:, None]], 1))


res = {}
for path in sorted(glob.glob(os.path.join(OUT, "models", "e3_*.pt"))):
    tag = os.path.basename(path)[3:-3]
    B = float(tag.split("_B")[1].split("_")[0])
    if math.isinf(B): continue
    m = E.SingleHead(B); m.load_state_dict(torch.load(path)); m.eval()
    rng = np.random.default_rng(123); rows = []
    with torch.no_grad():
        for N in [64, 128, 256]:
            for R in [2, 4, 8, 16, 32]:
                M = R - 1; g = (N - 1) // M
                oc = []
                for j in range(1, M + 1):
                    _, a, _ = m(streams(400, N, [N - 1 - j * g], rng), return_attn=True)
                    al = a[:, N - 1 - j * g].clamp(1e-9, 1 - 1e-9); oc.append(float((al / (1 - al)).mean()))
                pos = [N - 1 - j * g for j in range(M, -1, -1)]
                _, a, _ = m(streams(400, N, pos, rng), return_attn=True)
                ah = float(a[:, N - 1].mean())
                ceil_odds = math.exp(2 * B) / ((N - 1) * sum(oc) + (N - 1 - M))
                rows.append(dict(N=N, R=R, g=g, alpha_hot=ah, ceiling=ceil_odds / (1 + ceil_odds),
                                 mean_cold_alpha=float(np.mean([o / (1 + o) for o in oc]))))
    res[tag] = rows
    print(tag); [print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
json.dump(res, open(os.path.join(OUT, "e3_prop4.json"), "w"), indent=1)
