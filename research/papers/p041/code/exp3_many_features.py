"""E3: many-feature toy model with hierarchical pairs, correlated (non-hierarchical) pairs and independent
features. Train matched-width ReLU SAEs, measure per-latent precision/recall against ground truth, and test
weight-only diagnostics for absorption-induced false negatives. Outputs results/e3_many.json"""
import json, time, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sae import SAE, train

D_AMB, LAM, TAU = 64, 0.1, 0.1


def build_structure(rng):
    feats = []   # list of dicts describing generation
    pairs = []
    k = 0
    # 12 hierarchical pairs with varied q (P(child|parent)), magnitude CV, and violation rate rho
    combos = [(q, cv, rho) for q in (0.3, 0.6) for cv in (0.0, 0.2, 0.4) for rho in (0.0, 0.5)]
    for q, cv, rho in combos:
        pairs.append(dict(kind='hier', parent=k, child=k + 1, q=q, cv=cv, rho=rho, p=0.06)); k += 2
    # 4 correlated but non-hierarchical pairs: each alone w.p. 0.04, jointly w.p. 0.03
    for cv in (0.05, 0.2, 0.05, 0.4):
        pairs.append(dict(kind='corr', parent=k, child=k + 1, cv=cv, p_alone=0.04, p_joint=0.03)); k += 2
    # 4 mutually exclusive pairs with NON-orthogonal directions (cos = 1/sqrt2): distributionally
    # equivalent to a fully absorbed hierarchy (Prop. 4). Ground truth: no false negatives expected.
    for cv in (0.0, 0.0, 0.2, 0.2):
        pairs.append(dict(kind='excl', parent=k, child=k + 1, cv=cv, p_a=0.06 * 0.5, p_b=0.06 * 0.5)); k += 2
    indep = [dict(idx=k + i, p=0.05, cv=float(rng.choice([0.0, 0.2, 0.4]))) for i in range(8)]
    n = k + 8
    Q, _ = np.linalg.qr(rng.standard_normal((D_AMB, D_AMB)))
    F = Q[:, :n].T.copy()
    for pr in pairs:
        if pr['kind'] == 'excl':
            i, j = pr['parent'], pr['child']
            F[j] = (Q[:, i] + Q[:, j]) / np.sqrt(2)
    return dict(pairs=pairs, indep=indep, n=n, F=F)


def sample(S, B, rng):
    n = S['n']
    A = np.zeros((B, n))
    def mag(cv, size):
        return np.clip(1 + cv * rng.standard_normal(size), 0.05, None)
    for pr in S['pairs']:
        i, j = pr['parent'], pr['child']
        if pr['kind'] == 'hier':
            par = rng.random(B) < pr['p']
            ch = par & (rng.random(B) < pr['q'])
            ch_alone = (~par) & (rng.random(B) < pr['p'] * pr['q'] * pr['rho'])
            A[:, i] = par * mag(pr['cv'], B)
            A[:, j] = (ch | ch_alone) * mag(pr['cv'], B)
        elif pr['kind'] == 'excl':
            u = rng.random(B)
            aa = u < pr['p_a']
            bb = (u >= pr['p_a']) & (u < pr['p_a'] + pr['p_b'])
            A[:, i] = aa * mag(pr['cv'], B)
            # magnitude sqrt2 so that the b-feature input equals a P&C input of the hierarchy (a=c=1)
            A[:, j] = bb * np.sqrt(2) * mag(pr['cv'], B)
        else:
            u = rng.random(B)
            ji = u < pr['p_joint']
            ai = (u >= pr['p_joint']) & (u < pr['p_joint'] + pr['p_alone'])
            aj = (u >= pr['p_joint'] + pr['p_alone']) & (u < pr['p_joint'] + 2 * pr['p_alone'])
            A[:, i] = (ji | ai) * mag(pr['cv'], B)
            A[:, j] = (ji | aj) * mag(pr['cv'], B)
    for f in S['indep']:
        A[:, f['idx']] = (rng.random(B) < f['p']) * mag(f['cv'], B)
    return A @ S['F'], A


def auroc(score, label):
    score = np.asarray(score); label = np.asarray(label).astype(bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(gt)


def diagnostics(sae):
    Wd = sae.Wd                                   # (m,d) unit rows
    We = sae.We                                   # (d,m)
    m = Wd.shape[0]
    C = Wd @ Wd.T
    R = (We.T @ Wd.T)                             # R[i,j] = w_i . d_j
    selfr = np.diag(R).copy()
    A = np.full(m, -np.inf); N = np.full(m, -np.inf); Cmax = np.full(m, -np.inf)
    for i in range(m):
        for j in range(m):
            if i == j: continue
            A[i] = max(A[i], C[i, j] - R[i, j] / max(selfr[i], 1e-8))
            dperp = Wd[j] - C[i, j] * Wd[i]
            nrm = np.linalg.norm(dperp)
            if nrm > 1e-6:
                N[i] = max(N[i], -(We[:, i] @ (dperp / nrm)) / max(selfr[i], 1e-8) * C[i, j])
            Cmax[i] = max(Cmax[i], C[i, j])
    return A, N, Cmax


def run(seed, steps=5000):
    rng = np.random.default_rng(seed)
    S = build_structure(rng)
    sampler = lambda B, r: sample(S, B, r)[0]
    best = None
    for init in range(1):
        r = np.random.default_rng(seed * 100 + init)
        sae = SAE(D_AMB, S['n'], r)
        train(sae, sampler, LAM, steps=steps, batch=512, lr=3e-3, rng=r)
        Xv, _ = sample(S, 20000, np.random.default_rng(99))
        L = sae.loss(Xv, LAM)
        if best is None or L < best[1]:
            best = (sae, L)
    sae = best[0]
    X, A = sample(S, 60000, np.random.default_rng(seed + 7))
    Z = sae.encode(X)
    cos = sae.Wd @ S['F'].T                         # (m latents, n features)
    ri, ci = linear_sum_assignment(-cos)
    lat_of = {int(f): int(l) for l, f in zip(ri, ci)}
    Adiag, Ndiag, Cdiag = diagnostics(sae)
    role = {}
    for pr in S['pairs']:
        role[pr['parent']] = ('parent_' + pr['kind'], pr)
        role[pr['child']] = ('child_' + pr['kind'], pr)
    for f in S['indep']:
        role[f['idx']] = ('indep', f)
    rows = []
    for f in range(S['n']):
        l = lat_of[f]
        act = A[:, f] > 0
        fire = Z[:, l] > TAU
        rec = float(fire[act].mean()); prec = float(act[fire].mean()) if fire.any() else float('nan')
        r_, pr = role[f]
        row = dict(seed=seed, feature=f, latent=l, role=r_, cos=float(cos[l, f]), recall=rec, precision=prec,
                   diag_A=float(Adiag[l]), diag_N=float(Ndiag[l]), diag_Cmax=float(Cdiag[l]))
        if r_.endswith('hier'):
            row.update(q=pr['q'], cv=pr['cv'], rho=pr['rho'])
            # recall restricted to co-occurrence inputs (parent & child both active)
            other = pr['child'] if r_.startswith('parent') else pr['parent']
            both = act & (A[:, other] > 0)
            row['recall_cooc'] = float(fire[both].mean()) if both.any() else float('nan')
        elif r_.endswith('corr') or r_.endswith('excl'):
            row.update(cv=pr['cv'])
            other = pr['child'] if r_.startswith('parent') else pr['parent']
            both = act & (A[:, other] > 0)
            row['recall_cooc'] = float(fire[both].mean()) if both.any() else float('nan')
        rows.append(row)
    # linear probe baseline for parent concepts: least-squares readout on raw activations
    probe = []
    Xtr, Atr = sample(S, 20000, np.random.default_rng(seed + 11))
    for pr in S['pairs']:
        if pr['kind'] != 'hier': continue
        y = (Atr[:, pr['parent']] > 0).astype(float)
        w, *_ = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], y, rcond=None)
        yh = np.c_[X, np.ones(len(X))] @ w > 0.5
        yt = A[:, pr['parent']] > 0
        probe.append(dict(seed=seed, q=pr['q'], cv=pr['cv'], rho=pr['rho'], probe_recall=float(yh[yt].mean()),
                          probe_precision=float(yt[yh].mean())))
    return rows, probe, float(best[1])


if __name__ == '__main__':
    t0 = time.time()
    all_rows, all_probe, losses = [], [], []
    for seed in range(5):
        rows, probe, L = run(seed)
        all_rows += rows; all_probe += probe; losses.append(L)
        print('seed', seed, 'loss', L, 'time', time.time() - t0, flush=True)
    rec = np.array([r['recall'] for r in all_rows])
    lab = rec < 0.9
    summ = dict(n_latents=len(all_rows), n_low_recall=int(lab.sum()),
                auroc_A=auroc([r['diag_A'] for r in all_rows], lab),
                auroc_N=auroc([r['diag_N'] for r in all_rows], lab),
                auroc_Cmax=auroc([r['diag_Cmax'] for r in all_rows], lab),
                spearman_A=float(spearmanr([r['diag_A'] for r in all_rows], 1 - rec).correlation),
                spearman_N=float(spearmanr([r['diag_N'] for r in all_rows], 1 - rec).correlation),
                spearman_Cmax=float(spearmanr([r['diag_Cmax'] for r in all_rows], 1 - rec).correlation),
                auroc_A_no_excl=auroc([r['diag_A'] for r in all_rows if 'excl' not in r['role']], [r['recall'] < 0.9 for r in all_rows if 'excl' not in r['role']]),
                auroc_Cmax_no_excl=auroc([r['diag_Cmax'] for r in all_rows if 'excl' not in r['role']], [r['recall'] < 0.9 for r in all_rows if 'excl' not in r['role']]),
                auroc_N_no_excl=auroc([r['diag_N'] for r in all_rows if 'excl' not in r['role']], [r['recall'] < 0.9 for r in all_rows if 'excl' not in r['role']]),
                median_precision=float(np.nanmedian([r['precision'] for r in all_rows])),
                min_precision=float(np.nanmin([r['precision'] for r in all_rows])),
                frac_prec_ge_095_among_low_recall=float(np.mean([r['precision'] >= 0.95 for r in all_rows if r['recall'] < 0.9])) if lab.any() else float('nan'),
                sae_losses=losses)
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=all_rows, probe=all_probe), open('../results/e3_many.json', 'w'), indent=1)
    print('total', time.time() - t0)
