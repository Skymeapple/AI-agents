"""E1b: is Proposition 1 binding in the synthetic world?  Retrains ensemble member 0 (same seed as
run_all.py, hence the same model) and measures, for the E1 task (unknown deterministic vs known items):
  - delta_hat: Youden-J lower bound on TV(P0, P_alea) from a cross-fitted q-only classifier
  - AUROC of the best *trained* q-only detector (logistic on sorted log-probs + entropy) vs alpha
  - the Prop.1 ceiling evaluated at delta_hat, and the MSP AUROC for comparison.
Usage: python3 e1_qonly.py <world_seed>   (writes results/e1_qonly_w<seed>.json)"""
import numpy as np, json, sys, os
import world as W, mlp
from run_all import auroc, fit_logreg, ent, OUT
from theory import auroc_ceiling

def qfeat(P):
    s = -np.sort(-np.log(P + 1e-12), 1)[:, :5]
    return np.hstack([s, ent(P)[:, None]])

def youden(score, lab):
    o = np.argsort(-score); l = lab[o]
    tpr = np.cumsum(l) / l.sum(); fpr = np.cumsum(1 - l) / (1 - l).sum()
    return float(np.max(np.abs(tpr - fpr)))

ws = int(sys.argv[1]); EP = 30
w = W.make_world(seed=ws); nE, nR = w['n_ent'], w['n_rel']
e = np.repeat(np.arange(nE), nR); r = np.tile(np.arange(nR), nE)
m0 = mlp.train(w, seed=100 * ws + 0, epochs=EP)
P0, _ = m0.predict(e, r)
ps = w['pstar'][e, r]; tv = 0.5 * np.abs(P0 - ps).sum(1); msp = P0.max(1)
c = w['counts'][e, r]; alea = w['alea'][r]; det = ~alea
pos = det & (tv > 0.5); neg_det = det & (tv <= 0.5); neg_alea = alea & (c >= 20) & (tv <= 0.5)
X = qfeat(P0)
# split entities in two halves for cross-fitting
half = np.random.default_rng(10 + ws).random(nE) < 0.5; h = half[e]
rng = np.random.default_rng(500 + ws)
# delta_hat: classifier P0 (pos) vs P_alea (neg_alea), cross-fitted
J, aucs = [], []
for trm in [h, ~h]:
    ip = np.flatnonzero(pos & trm); ia = np.flatnonzero(neg_alea & trm)
    ip = rng.choice(ip, min(len(ip), 5 * len(ia)), replace=False)
    f = fit_logreg(X[np.r_[ip, ia]], np.r_[np.ones(len(ip)), np.zeros(len(ia))])
    tp = np.flatnonzero(pos & ~trm); ta = np.flatnonzero(neg_alea & ~trm)
    sc = f(X[np.r_[tp, ta]]); lab = np.r_[np.ones(len(tp)), np.zeros(len(ta))]
    J.append(youden(sc, lab)); aucs.append(auroc(sc, lab))
delta_hat = float(np.mean(J))
# best q-only detector for the full E1 task at each alpha (trained per alpha on one half, tested on other)
sweep = []
for a_ in np.linspace(0, 1, 11):
    res = []
    for rep in range(10):
        for trm in [h, ~h]:
            def draw(mask):
                ia = np.flatnonzero(neg_alea & mask); idt = np.flatnonzero(neg_det & mask)
                nneg = min(len(ia), 1000); na = int(round(a_ * nneg))
                ineg = np.r_[rng.choice(ia, na, replace=False), rng.choice(idt, nneg - na, replace=False)]
                ip = rng.choice(np.flatnonzero(pos & mask), nneg, replace=False)
                return np.r_[ip, ineg].astype(int), np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
            itr, ytr = draw(trm); ite, yte = draw(~trm)
            f = fit_logreg(X[itr], ytr)
            res.append([auroc(f(X[ite]), yte), auroc(1 - msp[ite], yte)])
    res = np.mean(res, 0)
    sweep.append(dict(alpha=float(a_), best_qonly=float(res[0]), msp=float(res[1]),
                      ceiling_at_delta_hat=float(auroc_ceiling(a_, delta_hat)),
                      ceiling_delta0=float(auroc_ceiling(a_, 0.0))))
out = dict(world=ws, delta_hat=delta_hat, auc_pos_vs_alea_qonly=float(np.mean(aucs)),
           n_pos=int(pos.sum()), n_alea=int(neg_alea.sum()), sweep=sweep,
           check_member0_acc_unseen_entity=float((P0.argmax(1) == ps.argmax(1))[det & (w['ent_count'][e] == 0)].mean()))
json.dump(out, open(f'{OUT}/e1_qonly_w{ws}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'sweep'}))
for s in sweep: print(s)
