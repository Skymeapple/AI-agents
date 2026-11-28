"""Check Proposition 3: for gamma<1 (non-persistent misalignment), chain oversight,
kappa=1, R=inf, P_S(n) -> a/(a+eps0) as n->inf, independent of G, where
a=(1-mu0)(1-gamma), eps0=mu0(1-d0)/(1-mu0 d0)."""
import json
from dataclasses import replace
from bos_model import Params, chain_exact
out = []
for gamma in [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]:
    for G in [400, 1000, 4000]:
        p = replace(Params(), gamma=gamma, G=float(G))
        a = (1 - p.mu0) * (1 - gamma); e0 = p.mu0 * (1 - p.d0) / (1 - p.mu0 * p.d0)
        lim = a / (a + e0) if a + e0 > 0 else float('nan')
        out.append(dict(gamma=gamma, G=G, P_n20000=chain_exact(20000, p)[0],
                        P_best_n_le_200=max(chain_exact(n, p)[0] for n in range(1, 201)), limit=lim))
        print(out[-1])
json.dump(out, open('../results/gamma_limit.json', 'w'), indent=1)
