"""Main experiment: train ensembles on synthetic worlds, then
 E1 aleatoric-vs-epistemic AUROC sweep, E2 familiarity trap, E3 probe shortcut sweep, E4 Good-Turing.
Usage: python3 run_all.py [n_worlds] [n_members] [epochs]"""
import numpy as np, json, sys, os, time
from scipy.optimize import minimize
import world as W, mlp, theory
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NM = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
FAM, UNF = 50, 10   # familiarity thresholds on entity training count

def auroc(score, label):  # label 1 = positive
    score = np.asarray(score, float); label = np.asarray(label, bool)
    o = score.argsort(kind='mergesort'); s = score[o]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1; i = j + 1
    r = np.empty(len(s)); r[o] = ranks
    npos = label.sum(); nneg = len(label) - npos
    return (r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def fit_logreg(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    def f(w):
        z = Xs @ w; l = np.logaddexp(0, z) - y * z
        g = Xs.T @ (1 / (1 + np.exp(-z)) - y)
        reg = lam * (w[:-1] @ w[:-1]) / 2; greg = np.r_[lam * w[:-1], 0]
        return l.sum() + reg, g + greg
    w = minimize(f, np.zeros(Xs.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=500)).x
    return lambda Z: 1 / (1 + np.exp(-(np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))]) @ w)))

def ent(P): return -(P * np.log(P + 1e-12)).sum(-1)

def run_world(ws):
    t0 = time.time()
    w = W.make_world(seed=ws)
    nE, nR = w['n_ent'], w['n_rel']
    e = np.repeat(np.arange(nE), nR); r = np.tile(np.arange(nR), nE)
    members = [mlp.train(w, seed=100 * ws + m, epochs=EP) for m in range(NM)]
    preds = [m.predict(e, r) for m in members]
    P0, H0 = preds[0]                       # "the model" = member 0
    Pens = np.mean([p for p, _ in preds], 0)
    MI = ent(Pens) - np.mean([ent(p) for p, _ in preds], 0)
    Emb0 = members[0].p['E'][e]
    ps = w['pstar'][e, r]; mode = ps.argmax(1)
    pred = P0.argmax(1); correct = pred == mode
    tv = 0.5 * np.abs(P0 - ps).sum(1)
    msp = P0.max(1); H = ent(P0)
    ne = w['ent_count'][e]; c = w['counts'][e, r]; alea = w['alea'][r]
    out = dict(world=ws)
    # ---------------- E2 familiarity trap: confidence vs accuracy by cell
    det = ~alea
    cells = {'familiar_seen': det & (ne >= FAM) & (c >= 1), 'familiar_unseenfact': det & (ne >= FAM) & (c == 0),
             'rare_seen': det & (ne > 0) & (ne <= UNF) & (c >= 1), 'rare_unseenfact': det & (ne > 0) & (ne <= UNF) & (c == 0),
             'unseen_entity': det & (ne == 0)}
    out['cells'] = {k: dict(n=int(m.sum()), acc=float(correct[m].mean()), msp=float(msp[m].mean()),
                            mi=float(MI[m].mean()), ens_acc=float((Pens.argmax(1) == mode)[m].mean()),
                            ens_msp=float(Pens.max(1)[m].mean()),
                            prior_acc=float((w['default'][w['types'][e], r] == mode)[m].mean()))
                    for k, m in cells.items()}
    # curve: confidence and accuracy vs log entity count, for seen / unseen facts
    bins = np.array([1, 3, 10, 30, 100, 300, 1000, 1e9]); curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        for name, m in [('seen', det & (c >= 1)), ('unseenfact', det & (c == 0))]:
            mm = m & (ne >= lo) & (ne < hi)
            if mm.sum() >= 20:
                curve.append(dict(lo=float(lo), hi=float(hi), kind=name, n=int(mm.sum()),
                                  acc=float(correct[mm].mean()), msp=float(msp[mm].mean()), mi=float(MI[mm].mean())))
    out['curve'] = curve
    # ---------------- E1 aleatoric vs epistemic: target = epistemic gap TV>0.5
    pos = det & (tv > 0.5)                       # model does not know (deterministic)
    neg_det = det & (tv <= 0.5)
    neg_alea = alea & (c >= 20) & (tv <= 0.5)    # well-estimated aleatoric facts
    rng = np.random.default_rng(ws)
    sweep = []
    nneg = min(neg_det.sum(), neg_alea.sum(), 2000); ip = np.flatnonzero(pos)
    for a_ in np.linspace(0, 1, 11):
        na = int(round(a_ * nneg)); A = []
        for rep in range(20):
            ineg = np.r_[rng.choice(np.flatnonzero(neg_alea), na, replace=False),
                         rng.choice(np.flatnonzero(neg_det), nneg - na, replace=False)]
            idx = np.r_[ip, ineg]; lab = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
            A.append([auroc(1 - msp[idx], lab), auroc(H[idx], lab), auroc(MI[idx], lab),
                      auroc(ent(Pens)[idx], lab), auroc(-np.log1p(ne[idx]), lab)])
        A = np.mean(A, 0)
        sweep.append(dict(alpha=float(a_), msp=A[0], entropy=A[1], ens_mi=A[2], ens_entropy=A[3], neg_log_count=A[4]))
    out['alea_sweep'] = sweep
    out['n_pos_E1'] = int(pos.sum()); out['n_neg_alea_E1'] = int(neg_alea.sum())
    out['alea_known_msp'] = float(msp[neg_alea].mean()); out['pos_msp'] = float(msp[pos].mean())
    # ---------------- E3 probe shortcut sweep (label: correct = "knows"), deterministic relations
    ent_split = np.random.default_rng(10 + ws).random(nE) < 0.5
    tr_e = ent_split[e]; te_e = ~ent_split[e]
    nat = (cells['familiar_seen'] | cells['unseen_entity'])
    dis = (cells['familiar_unseenfact'] | cells['rare_seen'])
    srt = -np.sort(-np.log(P0 + 1e-12), 1)[:, :5]
    feats = {'hidden': H0, 'hidden+output': np.hstack([H0, srt, H[:, None]]), 'output_only': np.hstack([srt, H[:, None]])}
    famF = ne >= FAM
    probe = []
    Ntr = 1500
    for fname, X in feats.items():
        for pid in [0.0, 0.02, 0.05, 0.1, 0.25, 0.5]:
            for rep in range(5):
                rr = np.random.default_rng(1000 * ws + rep)
                nd = int(round(pid * Ntr))
                pool_n = np.flatnonzero(nat & tr_e)
                pool_fu = np.flatnonzero(cells['familiar_unseenfact'] & tr_e)
                pool_rs = np.flatnonzero(cells['rare_seen'] & tr_e)
                itr = np.r_[rr.choice(pool_n, Ntr - nd, replace=len(pool_n) < Ntr - nd),
                            rr.choice(pool_fu, nd // 2, replace=len(pool_fu) < nd // 2),
                            rr.choice(pool_rs, nd - nd // 2, replace=False)].astype(int)
                y = correct[itr].astype(float)
                pi_eff = float((famF[itr] == correct[itr]).mean())
                if y.min() == y.max(): continue
                pr = fit_logreg(X[itr], y)
                sc = pr(X)
                mn, md = nat & te_e, dis & te_e
                fu = cells['familiar_unseenfact'] & te_e & ~correct   # familiar & wrong
                rk = cells['rare_seen'] & te_e & correct              # rare & right
                probe.append(dict(feat=fname, pi_dis=pid, rep=rep, pi_eff=pi_eff,
                                  auc_nat=auroc(1 - sc[mn], ~correct[mn]), auc_dis=auroc(1 - sc[md], ~correct[md]),
                                  false_know=float((sc[fu] > 0.5).mean()), missed_know=float((sc[rk] <= 0.5).mean())))
    # baselines on the same test splits
    mn, md = nat & te_e, dis & te_e
    base = {}
    for nm, s in [('msp', 1 - msp), ('entropy', H), ('ens_mi', MI), ('neg_log_count', -np.log1p(ne))]:
        base[nm] = dict(auc_nat=auroc(s[mn], ~correct[mn]), auc_dis=auroc(s[md], ~correct[md]))
    out['probe'] = probe; out['probe_baselines'] = base
    # natural co-occurrence pi = P(F=K) in the natural training pool
    Fam = ne[nat] >= FAM; out['pi_natural'] = float((Fam == correct[nat]).mean())
    # ---------------- Gaussian-geometry estimate (Prop. 2 applied to hidden states)
    X = H0[det & (ne > 0) | (det & (ne == 0))]
    msk = det
    Fm = (ne >= FAM); Um = (ne <= UNF)
    cm = {}
    for F_, fm in [(1, Fm), (0, Um)]:
        for K_, km in [(1, correct), (0, ~correct)]:
            cm[(F_, K_)] = msk & fm & km
    if all(cm[k].sum() >= 30 for k in cm):
        Xall = np.vstack([H0[cm[k]] - H0[cm[k]].mean(0) for k in cm])
        S = np.cov(Xall.T); S = 0.7 * S + 0.3 * np.eye(len(S)) * np.trace(S) / len(S)  # shrinkage
        L = np.linalg.cholesky(np.linalg.inv(S))
        m = {k: H0[cm[k]].mean(0) @ L for k in cm}
        a_vec = ((m[(1, 1)] - m[(0, 1)]) + (m[(1, 0)] - m[(0, 0)])) / 2
        b_vec = ((m[(1, 1)] - m[(1, 0)]) + (m[(0, 1)] - m[(0, 0)])) / 2
        out['geometry'] = dict(a=float(np.linalg.norm(a_vec)), b=float(np.linalg.norm(b_vec)),
                               cos=float(a_vec @ b_vec / np.linalg.norm(a_vec) / np.linalg.norm(b_vec)),
                               pred_fk={str(p): float(theory.lda_rates(a_vec, b_vec, p)[0]) for p in [0.5, 0.75, 0.9, 0.98, 0.999]},
                               cell_n={str(k): int(v.sum()) for k, v in cm.items()})
    if 'geometry' in out:
        for p_ in probe:
            p_['pred_fk'] = float(theory.lda_rates(a_vec, b_vec, p_['pi_eff'])[0])
    # ---------------- E4 Good-Turing: unknown mass of fresh queries vs singleton fraction
    Ntr_ = len(w['es']); cnt = w['counts']
    n1 = (cnt == 1).sum()
    rq = np.random.default_rng(77 + ws); nq = 200_000
    qe = rq.choice(nE, nq, p=w['pop'])
    qr = np.array([rq.choice(np.flatnonzero(w['documented'][x])) for x in qe])
    cq = cnt[qe, qr]; cor_q = correct.reshape(nE, nR)[qe, qr]; detq = ~w['alea'][qr]
    out['good_turing'] = dict(singleton_frac=float(n1 / Ntr_), missing_mass=float((cq == 0).mean()),
                              err_rate_det=float((~cor_q[detq]).mean()),
                              err_given_c0=float((~cor_q[detq & (cq == 0)]).mean()),
                              err_given_c1=float((~cor_q[detq & (cq == 1)]).mean()),
                              err_given_c2plus=float((~cor_q[detq & (cq >= 2)]).mean()),
                              frac_det_err_from_c0=float((~cor_q & detq & (cq == 0)).sum() / max(1, (~cor_q & detq).sum())))
    out['seconds'] = time.time() - t0
    print(f'world {ws} done in {out["seconds"]:.0f}s', flush=True)
    return out

if __name__ == '__main__':
    # optional 4th arg: run a single world index (for parallel runs); then merge with --merge
    if len(sys.argv) > 4 and sys.argv[4] == '--merge':
        allres = [json.load(open(f'{OUT}/world_{s}.json')) for s in range(NW)]
    else:
        todo = [int(sys.argv[4])] if len(sys.argv) > 4 else range(NW)
        for s in todo:
            json.dump(run_world(s), open(f'{OUT}/world_{s}.json', 'w'), indent=1, default=float)
        if len(sys.argv) > 4: sys.exit(0)
        allres = [json.load(open(f'{OUT}/world_{s}.json')) for s in range(NW)]
    json.dump(dict(config=dict(n_worlds=NW, n_members=NM, epochs=EP, FAM=FAM, UNF=UNF), worlds=allres),
              open(f'{OUT}/main_results.json', 'w'), indent=1, default=float)
