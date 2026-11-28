"""Simulation: does routing votes to the contested band certify more seats than uniform
exposure for the same vote budget?  Ground truth = empirical-Bayes (beta-binomial)
shrunk CCAI agreement rates (M = 275, k = 58).  Both arms are evaluated with the same
marginal rank-certification procedure (parametric bootstrap, B = 400), and we also
check certification errors against the known truth.
Usage: python3 run_routing_sim.py -> ../results/routing_sim.csv, routing_sim.json"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import betabinom
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
from constlib import load_ccai
from certify import certify
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
V, cm, _ = load_ccai(os.path.join(HERE, '..', 'data', 'ccai'))
S = ~np.isnan(V); seen = S.sum(0); ag = (V == 1).sum(0)
nll = lambda th: -betabinom.logpmf(ag, seen, *np.exp(th)).sum()
al, be = np.exp(minimize(nll, [np.log(5), 0.0], method='Nelder-Mead').x)
p = (ag + al) / (seen + al + be)
M, K = len(p), 58
truth_in = np.zeros(M, bool); truth_in[np.argsort(-p)[:K]] = True
rng = np.random.default_rng(2026)
BASE = int(S.sum())             # CCAI's realised vote budget (34,563)
M0 = 30                         # initial uniform exposure per statement (both arms)
BATCH = 2750

def evaluate(a, m):
    ph = a / m
    T = rng.binomial(np.broadcast_to(m, (400, M)), np.broadcast_to(ph, (400, M))) / m
    r = certify(None, None, K, T=T, full=ph, mode='marginal')
    return dict(cert_in=int(r['cin'].sum()), cert_in_wrong=int((r['cin'] & ~truth_in).sum()),
                cert_out=int(r['cout'].sum()), cert_out_wrong=int((r['cout'] & truth_in).sum()),
                stable_in=int((r['pi'] >= .95).sum()), contested=int(((r['pi'] > .05) & (r['pi'] < .95)).sum()))

rows = []
for mult in (1, 2, 4, 8):
    budget = BASE * mult
    for rep in range(4):
        # uniform arm
        m = np.full(M, budget // M); a = rng.binomial(m, p)
        rows.append(dict(arm='uniform', mult=mult, rep=rep, budget=int(m.sum()), **evaluate(a, m)))
        # adaptive arm: start uniform at M0, then route batches to statements whose
        # z = 2 normal interval contains the estimated cutoff
        for arm in ('routed', 'cutoff'):
            m = np.full(M, M0); a = rng.binomial(m, p); used = m.sum()
            while used < budget:
                ph = a / m
                se = np.sqrt(np.maximum(ph * (1 - ph), 0.01) / m)
                if arm == 'routed':
                    # rank-aware: statements whose analytic marginal rank interval straddles k
                    Dm = ph[:, None] - ph[None, :]
                    sig = Dm > 3.5 * np.sqrt(se[:, None] ** 2 + se[None, :] ** 2)
                    rhi = M - sig.sum(1); rlo = 1 + sig.sum(0)
                    target = ~((rhi <= K) | (rlo > K))
                else:
                    # cutoff-proximity: statements whose z=2 interval contains the estimated cutoff
                    o = np.sort(ph)[::-1]; cut = 0.5 * (o[K - 1] + o[K])
                    target = np.abs(ph - cut) < 2 * se
                if not target.any():
                    target = np.ones(M, bool)
                w = np.where(target, se, 0); w = w / w.sum()
                add = rng.multinomial(min(BATCH, budget - used), w)
                a = a + rng.binomial(add, p); m = m + add; used += add.sum()
            rows.append(dict(arm=arm, mult=mult, rep=rep, budget=int(m.sum()), **evaluate(a, m)))
        print(rows[-3:], flush=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, 'routing_sim.csv'), index=False)
summ = df.groupby(['arm', 'mult']).mean(numeric_only=True).round(2).reset_index()
json.dump(dict(eb_prior=[float(al), float(be)], base_budget=BASE, summary=summ.to_dict('records')),
          open(os.path.join(RES, 'routing_sim.json'), 'w'), indent=1)
print(summ.to_string())
