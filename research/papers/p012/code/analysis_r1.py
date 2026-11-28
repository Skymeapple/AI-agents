"""Revision R1 analyses on cached ensembles (train_cache.py).  Usage: python3 analysis_r1.py <variant> <world_seed>
Writes results/r1/<variant>_w<seed>.json.  Addresses reviewer points:
  M2  E1 with the -log entity-count baseline, and an exposure-matched E1 (familiar entities only, negatives
      resampled to the positives' entity-count histogram; count baseline ~0.5 by construction)
  M3  delta-hat with a threshold fixed on the training fold (m3) + bootstrap CI; best trained q-only detector in the
      matched design; same quantities for a Bayes-reference learner (exact predictive of the generative model)
  M5  within-familiarity-stratum AUROC (e.g. familiar-unseen-fact: correct vs wrong) for probes and baselines
  M6  probe L2 / threshold ablation; Prop. 2 gap decomposition (balanced theory -> class-prior-corrected theory ->
      empirical LDA on hidden states -> logistic probe)
  m5  Good-Turing in imperfect-memorization regimes."""
import numpy as np, json, sys, os, itertools
import world as W, theory
_argv = sys.argv; sys.argv = sys.argv[:1]          # run_all.py parses argv at import time
from run_all import auroc, fit_logreg, ent
sys.argv = _argv
from train_cache import VARIANTS, CACHE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'r1')
FAM, UNF = 50, 10

def qfeat(P):
    s = -np.sort(-np.log(P + 1e-12), 1)[:, :5]
    return np.hstack([s, ent(P)[:, None]])

def roc_points(score, lab):
    o = np.argsort(-score, kind='mergesort'); l = lab[o].astype(float); s = score[o]
    tpr = np.cumsum(l) / l.sum(); fpr = np.cumsum(1 - l) / (1 - l).sum()
    return s, tpr, fpr

def youden_threshold(score, lab):
    s, tpr, fpr = roc_points(score, lab); i = int(np.argmax(tpr - fpr)); return s[i]

def J_at(score, lab, thr):
    lab = lab.astype(bool); return float((score[lab] >= thr).mean() - (score[~lab] >= thr).mean())

def bayes_reference(w):
    """Predictive distribution of the true generative model given the training counts, with oracle entity types
    (the type of a never-seen entity is unknown -> mixture over the empirical type frequencies)."""
    nE, nR, V = w['n_ent'], w['n_rel'], w['n_val']
    vc = np.zeros((nE, nR, V)); np.add.at(vc, (w['es'], w['rs'], w['ys']), 1)
    prior = w['prior'][w['types']]                                  # (nE, nR, V)
    tfreq = np.bincount(w['types'], minlength=w['prior'].shape[0]) / nE
    mix = np.einsum('t,trv->rv', tfreq, w['prior'])
    q = prior.copy(); q[w['ent_count'] == 0] = mix
    c = w['counts']
    for r in range(nR):
        seen = c[:, r] >= 1
        if not w['alea'][r]:
            q[seen, r] = (vc[seen, r] > 0).astype(float)            # deterministic: observed value is the answer
        elif w['matched']:
            q[:, r] = q[:, r]                                       # aleatoric p* = type prior exactly (known structure)
        else:                                                       # exact posterior over (top, o1, o2) assignments
            idx = np.flatnonzero(seen)
            dflt = w['default'][w['types'][idx], r]; rho = w['rho']
            trip = np.array([t for t in itertools.permutations(range(V), 3)])   # (V*(V-1)*(V-2), 3)
            for i0 in range(0, len(idx), 64):
                ii = idx[i0:i0 + 64]; n = vc[ii, r]                 # (b, V)
                ptop = np.where(trip[None, :, 0] == dflt[i0:i0 + 64, None], rho, 0) + (1 - rho) / V
                tot = n.sum(1, keepdims=True)
                cov = n[:, trip].sum(2)                             # counts covered by the triple
                ll = (n[:, trip[:, 0]] * np.log(.5) + n[:, trip[:, 1]] * np.log(.3) + n[:, trip[:, 2]] * np.log(.2))
                ll = np.where(cov == tot, ll, -np.inf) + np.log(ptop)
                post = np.exp(ll - ll.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
                qq = np.zeros((len(ii), V))
                for k, pk in enumerate([.5, .3, .2]):
                    for v in range(V): qq[:, v] += pk * post[:, trip[:, k] == v].sum(1)
                q[ii, r] = qq
    q = 0.999 * q + 0.001 / V
    return q.reshape(nE * nR, V)

def run(var, ws):
    wkw, _ = VARIANTS[var]
    w = W.make_world(seed=ws, **wkw); nE, nR = w['n_ent'], w['n_rel']
    Z = np.load(f'{CACHE}/{var}_w{ws}.npz'); Pm = Z['P'].astype(float); H0 = Z['H0'].astype(float)
    e = np.repeat(np.arange(nE), nR); r = np.tile(np.arange(nR), nE)
    ps = w['pstar'][e, r]; mode = ps.argmax(1)
    ne = w['ent_count'][e]; c = w['counts'][e, r]; alea = w['alea'][r]; det = ~alea
    dflt = w['default'][w['types'][e], r]
    learners = {'mlp': Pm[0]}
    if var in ('base', 'matched'): learners['bayes'] = bayes_reference(w)
    Pens = Pm.mean(0); MI = ent(Pens) - np.mean([ent(p) for p in Pm], 0)
    out = dict(variant=var, world=ws, n_members=len(Pm))
    rng = np.random.default_rng(900 + ws)
    half = np.random.default_rng(10 + ws).random(nE) < 0.5; h = half[e]
    for ln, P0 in learners.items():
        R = out[ln] = {}
        pred = P0.argmax(1); correct = pred == mode; msp = P0.max(1); H = ent(P0)
        tv = 0.5 * np.abs(P0 - ps).sum(1)
        cells = {'familiar_seen': det & (ne >= FAM) & (c >= 1), 'familiar_unseenfact': det & (ne >= FAM) & (c == 0),
                 'rare_seen': det & (ne > 0) & (ne <= UNF) & (c >= 1), 'rare_unseenfact': det & (ne > 0) & (ne <= UNF) & (c == 0),
                 'unseen_entity': det & (ne == 0)}
        R['cells'] = {k: dict(n=int(m.sum()), acc=float(correct[m].mean()), msp=float(msp[m].mean()), mi=float(MI[m].mean()),
                              prior_acc=float((dflt == mode)[m].mean())) for k, m in cells.items()}
        # calibration (ECE, 15 bins) on deterministic items
        bins = np.minimum((msp[det] * 15).astype(int), 14)
        R['ece_det'] = float(sum(abs(correct[det][bins == b].mean() - msp[det][bins == b].mean()) * (bins == b).mean()
                                 for b in range(15) if (bins == b).any()))
        scores = {'msp': 1 - msp, 'entropy': H, 'ens_mi': MI, 'ens_entropy': ent(Pens), 'neg_log_count': -np.log1p(ne),
                  'neg_log_factcount': -np.log1p(c)}
        X = qfeat(P0)
        # ---------- (M2) original E1 (positives: det & G=1; negatives: alea c>=20 & G=0 / det & G=0), with count baselines
        pos = det & (tv > 0.5); neg_det = det & (tv <= 0.5); neg_alea = alea & (c >= 20) & (tv <= 0.5)
        sweep = []
        if neg_alea.sum() >= 20:
            nneg = min(neg_det.sum(), neg_alea.sum(), 2000); ip = np.flatnonzero(pos)
            for a_ in [0, .25, .5, .75, 1.]:
                na = int(round(a_ * nneg)); A = []
                for rep in range(10):
                    ineg = np.r_[rng.choice(np.flatnonzero(neg_alea), na, replace=False),
                                 rng.choice(np.flatnonzero(neg_det), nneg - na, replace=False)].astype(int)
                    idx = np.r_[ip, ineg]; lab = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
                    A.append([auroc(s[idx], lab) for s in scores.values()])
                sweep.append(dict(alpha=a_, **dict(zip(scores, np.mean(A, 0)))))
        R['E1_original'] = sweep
        R['E1_original_frac_pos_unseen_entity'] = float((ne[pos] == 0).mean())
        # ---------- (M2) exposure-matched E1: familiar entities only; positives = unseen fact & wrong;
        # negatives = known deterministic (seen & correct) or known aleatoric (c>=20, G=0); negatives resampled to the
        # positives' log-entity-count histogram.
        F = ne >= FAM
        mpos = det & F & (c == 0) & ~correct
        mnd = det & F & (c >= 1) & correct
        mna = alea & F & (c >= 20) & (tv <= 0.5)
        eb = np.array([50, 70, 100, 150, 250, 500, 1e9])
        def matched_draw(mask_pos, mask_neg, nneg, rr):
            bp = np.digitize(ne[mask_pos], eb); ineg = []
            cand = np.flatnonzero(mask_neg); bn = np.digitize(ne[cand], eb)
            frac = np.bincount(bp, minlength=len(eb) + 1) / len(bp)
            for b_ in range(len(eb) + 1):
                k = int(round(frac[b_] * nneg))
                if k == 0: continue
                pool = cand[bn == b_]
                if len(pool) == 0: pool = cand[np.argsort(np.abs(bn - b_))[:20]]
                ineg.append(rr.choice(pool, k, replace=True))
            return np.concatenate(ineg).astype(int)
        msweep = []
        if mpos.sum() >= 30 and mna.sum() >= 30:
            ip_all = np.flatnonzero(mpos); nneg = len(ip_all)
            for a_ in [0, .5, 1.]:
                A, Q = [], []
                for rep in range(10):
                    na = int(round(a_ * nneg))
                    parts = []
                    if na: parts.append(matched_draw(mpos, mna, na, rng))
                    if nneg - na: parts.append(matched_draw(mpos, mnd, nneg - na, rng))
                    ineg = np.concatenate(parts); idx = np.r_[ip_all, ineg]
                    lab = np.r_[np.ones(len(ip_all)), np.zeros(len(ineg))]
                    A.append([auroc(s[idx], lab) for s in scores.values()])
                    # best trained q-only detector, cross-fitted over entity halves
                    for trm in [h, ~h]:
                        tr = trm[idx];
                        if lab[tr].min() == lab[tr].max() or lab[~tr].min() == lab[~tr].max(): continue
                        f = fit_logreg(X[idx[tr]], lab[tr]); Q.append(auroc(f(X[idx[~tr]]), lab[~tr]))
                msweep.append(dict(alpha=a_, best_qonly=float(np.mean(Q)), **dict(zip(scores, np.mean(A, 0)))))
        R['E1_matched'] = msweep
        R['E1_matched_n'] = dict(pos=int(mpos.sum()), neg_det=int(mnd.sum()), neg_alea=int(mna.sum()))
        # ---------- (M3, m3) delta-hat: TV lower bound with threshold fixed on the training fold; bootstrap CI
        def delta_hat(mp, ma):
            Js, As, boots = [], [], []
            for trm in [h, ~h]:
                ip = np.flatnonzero(mp & trm); ia = np.flatnonzero(ma & trm)
                if len(ia) < 10 or len(ip) < 10: return None
                ip = rng.choice(ip, min(len(ip), 5 * len(ia)), replace=False)
                lab_tr = np.r_[np.ones(len(ip)), np.zeros(len(ia))]
                f = fit_logreg(X[np.r_[ip, ia]], lab_tr)
                thr = youden_threshold(f(X[np.r_[ip, ia]]), lab_tr)
                tp = np.flatnonzero(mp & ~trm); ta = np.flatnonzero(ma & ~trm)
                sc_p, sc_a = f(X[tp]), f(X[ta])
                Js.append(float((sc_p >= thr).mean() - (sc_a >= thr).mean()))
                As.append(auroc(np.r_[sc_p, sc_a], np.r_[np.ones(len(tp)), np.zeros(len(ta))]))
                boots.append([float((rng.choice(sc_p, len(sc_p)) >= thr).mean() - (rng.choice(sc_a, len(sc_a)) >= thr).mean())
                              for _ in range(300)])
            b = np.mean(boots, 0)
            return dict(delta_hat=float(np.mean(Js)), ci90=[float(np.percentile(b, 5)), float(np.percentile(b, 95))],
                        auc=float(np.mean(As)))
        R['delta_original'] = delta_hat(pos, neg_alea)
        R['delta_matched'] = delta_hat(mpos, mna)
        if ln != 'mlp':
            continue
        # ---------- E3 probes + (M5) within-stratum AUROC + (M6) ablations + Prop. 2 decomposition
        tr_e = h; te_e = ~h
        nat = cells['familiar_seen'] | cells['unseen_entity']; dis = cells['familiar_unseenfact'] | cells['rare_seen']
        srt = -np.sort(-np.log(P0 + 1e-12), 1)[:, :5]
        feats = {'hidden': H0, 'hidden+output': np.hstack([H0, srt, H[:, None]]), 'output_only': np.hstack([srt, H[:, None]])}
        strata = {'FU': cells['familiar_unseenfact'], 'RU': cells['rare_unseenfact'], 'FS': cells['familiar_seen'],
                  'RS': cells['rare_seen'], 'UE': cells['unseen_entity']}
        def within(sc, wrong_score=True):
            res = {}
            for k, m in strata.items():
                mm = m & te_e; y = ~correct[mm]
                res[k] = float(auroc(sc[mm], y)) if 5 <= y.sum() <= len(y) - 5 else None
            mn, md = nat & te_e, dis & te_e
            res['natural'] = float(auroc(sc[mn], ~correct[mn])) if 0 < (~correct[mn]).sum() < mn.sum() else None
            res['dissociated'] = float(auroc(sc[md], ~correct[md])) if 0 < (~correct[md]).sum() < md.sum() else None
            return res
        base = {k: within(s) for k, s in scores.items()}
        base['type_default_disagree'] = within((pred != dflt).astype(float) + 1e-3 * (1 - msp))
        R['baselines'] = base
        R['stratum_n_wrong'] = {k: [int((m & te_e & ~correct).sum()), int((m & te_e).sum())] for k, m in strata.items()}
        famF = ne >= FAM
        # geometry (whitened hidden), as in run_all.py
        cm = {}
        for F_, fm in [(1, ne >= FAM), (0, ne <= UNF)]:
            for K_, km in [(1, correct), (0, ~correct)]: cm[(F_, K_)] = det & fm & km
        geo = None
        if all(cm[k].sum() >= 30 for k in cm):
            Xall = np.vstack([H0[cm[k]] - H0[cm[k]].mean(0) for k in cm])
            S = np.cov(Xall.T); S = 0.7 * S + 0.3 * np.eye(len(S)) * np.trace(S) / len(S)
            L = np.linalg.cholesky(np.linalg.inv(S))
            mu = {k: H0[cm[k]].mean(0) @ L for k in cm}
            a_vec = ((mu[(1, 1)] - mu[(0, 1)]) + (mu[(1, 0)] - mu[(0, 0)])) / 2
            b_vec = ((mu[(1, 1)] - mu[(1, 0)]) + (mu[(0, 1)] - mu[(0, 0)])) / 2
            geo = (a_vec, b_vec, L)
            R['geometry'] = dict(a=float(np.linalg.norm(a_vec)), b=float(np.linalg.norm(b_vec)),
                                 cos=float(a_vec @ b_vec / np.linalg.norm(a_vec) / np.linalg.norm(b_vec)))
        probes = []
        Ntr = 1500
        pool_n = np.flatnonzero(nat & tr_e); pool_fu = np.flatnonzero(cells['familiar_unseenfact'] & tr_e)
        pool_rs = np.flatnonzero(cells['rare_seen'] & tr_e)
        fu_w = cells['familiar_unseenfact'] & te_e & ~correct; rk = cells['rare_seen'] & te_e & correct
        configs = [(f, pid, 1.0) for f in feats for pid in [0.0, 0.1, 0.5]] + \
                  [('hidden', pid, lam) for pid in [0.0, 0.5] for lam in [0.01, 100.0]]
        for fname, pid, lam in configs:
            Xf = feats[fname]
            for rep in range(5):
                rr = np.random.default_rng(1000 * ws + rep)
                nd = int(round(pid * Ntr))
                if len(pool_fu) == 0 or len(pool_rs) == 0: continue
                itr = np.r_[rr.choice(pool_n, Ntr - nd, replace=len(pool_n) < Ntr - nd),
                            rr.choice(pool_fu, nd // 2, replace=len(pool_fu) < nd // 2),
                            rr.choice(pool_rs, nd - nd // 2, replace=len(pool_rs) < nd - nd // 2)].astype(int)
                y = correct[itr].astype(float)
                if y.min() == y.max(): continue
                pr = fit_logreg(Xf[itr], y, lam=lam); sc = pr(Xf)
                thr_y = youden_threshold(sc[itr], y)                 # threshold chosen on the training set
                row = dict(feat=fname, pi_dis=pid, lam=lam, rep=rep,
                           pi_eff=float((famF[itr] == correct[itr]).mean()), p1=float(y.mean()),
                           within=within(1 - sc),
                           false_know=float((sc[fu_w] > 0.5).mean()), missed_know=float((sc[rk] <= 0.5).mean()),
                           false_know_youden=float((sc[fu_w] >= thr_y).mean()),
                           missed_know_youden=float((sc[rk] < thr_y).mean()))
                if fname == 'hidden' and lam == 1.0 and geo is not None:
                    a_vec, b_vec, L = geo
                    Fi = famF[itr]; Ki = correct[itr]
                    pi1 = float(Fi[Ki].mean()); pi0 = float((~Fi[~Ki]).mean()); p1 = float(Ki.mean())
                    row['pred_fk_balanced'] = float(theory.lda_rates(a_vec, b_vec, row['pi_eff'])[0])
                    row['pred_fk_prior'] = theory.lda_rates_general(a_vec, b_vec, pi1, pi0, p1)[0]
                    # empirical LDA on whitened hidden states of the actual training set (shrunk pooled covariance)
                    Zt = H0[itr] @ L; m1, m0 = Zt[Ki].mean(0), Zt[~Ki].mean(0)
                    Sw = np.cov(np.vstack([Zt[Ki] - m1, Zt[~Ki] - m0]).T); Sw = 0.7 * Sw + 0.3 * np.eye(len(Sw)) * np.trace(Sw) / len(Sw)
                    wl = np.linalg.solve(Sw, m1 - m0); cl = wl @ (m1 + m0) / 2 + np.log((1 - p1) / p1)
                    row['lda_emp_fk'] = float(((H0[fu_w] @ L) @ wl > cl).mean())
                probes.append(row)
        R['probes'] = probes
        R['pi_natural'] = float(((ne[nat] >= FAM) == correct[nat]).mean())
        # ---------- E4 Good-Turing per variant
        cnt = w['counts']; n1 = (cnt == 1).sum()
        rq = np.random.default_rng(77 + ws); nq = 100_000
        qe = rq.choice(nE, nq, p=w['pop'])
        qr = np.array([rq.choice(np.flatnonzero(w['documented'][x])) for x in qe])
        cq = cnt[qe, qr]; cor_q = correct.reshape(nE, nR)[qe, qr]; detq = ~w['alea'][qr]
        R['good_turing'] = dict(singleton_frac=float(n1 / len(w['es'])), missing_mass=float((cq == 0).mean()),
                                err_rate_det=float((~cor_q[detq]).mean()),
                                err_given_c0=float((~cor_q[detq & (cq == 0)]).mean()),
                                err_given_c1=float((~cor_q[detq & (cq == 1)]).mean()),
                                err_given_c2plus=float((~cor_q[detq & (cq >= 2)]).mean()),
                                frac_det_err_from_c0=float((~cor_q & detq & (cq == 0)).sum() / max(1, (~cor_q & detq).sum())))
    return out

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    var, ws = sys.argv[1], int(sys.argv[2])
    res = run(var, ws)
    json.dump(res, open(f'{OUT}/{var}_w{ws}.json', 'w'), indent=1, default=float)
    print(var, ws, 'ok', flush=True)
