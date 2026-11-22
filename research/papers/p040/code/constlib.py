"""Core library: loading Polis exports, aggregation rules, designer multiverse,
representation audits. All functions are deterministic given a numpy Generator.

Vote matrix convention: V is (n participants x m statements) float array with
1 = agree, -1 = disagree, 0 = pass, NaN = not seen.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

META = ['participant', 'xid', 'group-id', 'n-comments', 'n-votes', 'n-agree', 'n-disagree']


# ----------------------------------------------------------------- loading
def load_polis(votes_csv, comments_csv, min_votes=7, min_seen=10, moderated_only=True):
    pv = pd.read_csv(votes_csv)
    cm = pd.read_csv(comments_csv)
    cm['comment-id'] = cm['comment-id'].astype(str)
    stmt_cols = [c for c in pv.columns if c not in META]
    if moderated_only and 'moderated' in cm.columns:
        keep = set(cm.loc[cm['moderated'] > 0, 'comment-id'])
        stmt_cols = [c for c in stmt_cols if c in keep]
    pv = pv[pv['n-votes'] >= min_votes]
    V = pv[stmt_cols].to_numpy(dtype=float)
    seen = (~np.isnan(V)).sum(0)
    ok = seen >= min_seen
    stmt_cols = [c for c, o in zip(stmt_cols, ok) if o]
    V = V[:, ok]
    # drop participants with no remaining votes
    r = (~np.isnan(V)).sum(1) > 0
    V = V[r]
    cm = cm.set_index('comment-id').loc[stmt_cols].reset_index()
    return V, cm


def load_ccai(base):
    """CCAI data as released by Huang et al. (github.com/saffronh/ccai):
    clean_votes.csv (1002 x 275) and clean_comments.csv."""
    pv = pd.read_csv(f'{base}/clean_votes.csv')
    cm = pd.read_csv(f'{base}/clean_comments.csv')
    cm['comment-id'] = cm['comment-id'].astype(str)
    stmt_cols = [c for c in pv.columns if c not in META]
    V = pv[stmt_cols].to_numpy(dtype=float)
    cm = cm.set_index('comment-id').loc[stmt_cols].reset_index()
    return V, cm, pv['group-id'].to_numpy()


# ----------------------------------------------------------------- grouping
def polis_groups(V, K, impute='zero', seed=0, center='stmt'):
    """Polis-style opinion groups: impute, PCA to 2D, k-means.
    center='stmt': PCA of participants in statement space (columns centred; the
      usual Polis convention).
    center='part': PCA fitted on the transposed matrix, participant coordinates
      taken from the components (the convention in the CCAI notebook), i.e.
      rows centred.
    (Polis additionally pre-clusters into 100 centres; we cluster directly.)"""
    X = V.copy()
    if impute == 'zero':
        X = np.nan_to_num(X, nan=0.0)
    elif impute == 'mean':
        mu = np.nanmean(X, axis=0)
        mu = np.nan_to_num(mu, nan=0.0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(mu, idx[1])
    else:
        raise ValueError(impute)
    if K == 1:
        return np.zeros(len(V), dtype=int)
    if center == 'stmt':
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        emb = U[:, :2] * S[:2]
    else:
        Xc = X - X.mean(1, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        emb = U[:, :2]
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(emb)
    return km.labels_


# ----------------------------------------------------------------- estimators
def agree_rate(V, est):
    """Per-statement agreement estimate over the rows of V."""
    a = (V == 1).sum(0).astype(float)
    d = (V == -1).sum(0).astype(float)
    s = (~np.isnan(V)).sum(0).astype(float)
    if est == 'seen':        # agree / seen (passes count against) -- CCAI notebook
        return np.divide(a, s, out=np.zeros_like(a), where=s > 0)
    if est == 'decided':     # agree / (agree + disagree): passes ignored
        t = a + d
        return np.divide(a, t, out=np.zeros_like(a), where=t > 0)
    if est == 'laplace':     # (agree+1)/(seen+2) -- Polis group-aware consensus
        return (a + 1) / (s + 2)
    raise ValueError(est)


def group_scores(V, groups, est):
    G = np.unique(groups)
    return np.vstack([agree_rate(V[groups == g], est) for g in G])  # (|G| x m)


# ----------------------------------------------------------------- rules (fixed size k)
def topk(score, k):
    # deterministic tie-break by index
    order = np.lexsort((np.arange(len(score)), -score))
    return np.sort(order[:k])


def rule_consmin(V, groups, est, k):
    return topk(group_scores(V, groups, est).min(0), k)


def rule_gac(V, groups, est, k):
    return topk(np.prod(group_scores(V, groups, est), axis=0), k)


def rule_av(V, est, k):
    return topk(agree_rate(V, est), k)


def approvals(V):
    return (V == 1)


def rule_seqpav(V, k):
    A = approvals(V).astype(float)
    n, m = A.shape
    sat = np.zeros(n)
    chosen = []
    avail = np.ones(m, bool)
    for _ in range(k):
        gain = (A / (sat[:, None] + 1)).sum(0)
        gain[~avail] = -1
        c = int(np.argmax(gain))
        chosen.append(c); avail[c] = False; sat += A[:, c]
    return np.sort(np.array(chosen))


def rule_mes(V, k):
    """Method of Equal Shares (approval utilities), completed by approval count."""
    A = approvals(V)
    n, m = A.shape
    budget = np.full(n, k / n)
    chosen = []
    avail = np.ones(m, bool)
    supporters = [np.where(A[:, c])[0] for c in range(m)]
    while len(chosen) < k:
        best, best_rho = None, np.inf
        for c in np.where(avail)[0]:
            sup = supporters[c]
            if len(sup) == 0:
                continue
            b = np.sort(budget[sup])
            if b.sum() < 1 - 1e-12:
                continue
            # smallest rho with sum(min(b_i, rho)) >= 1
            paid = 0.0
            L = len(b)
            rho = None
            for j in range(L):
                rem = L - j
                if paid + b[j] * rem >= 1 - 1e-12:
                    rho = (1 - paid) / rem
                    break
                paid += b[j]
            if rho is not None and rho < best_rho - 1e-15:
                best, best_rho = c, rho
        if best is None:
            break
        sup = supporters[best]
        budget[sup] -= np.minimum(budget[sup], best_rho)
        chosen.append(best); avail[best] = False
    if len(chosen) < k:  # completion by approval count
        cnt = A.sum(0).astype(float)
        cnt[~avail] = -np.inf
        extra = topk(cnt, k - len(chosen))
        chosen += list(extra)
    return np.sort(np.array(chosen[:k]))


def rule_greedycc(V, k, core=None):
    """Greedy Chamberlin-Courant (approval coverage), optionally after a fixed core."""
    A = approvals(V)
    n, m = A.shape
    chosen = list(core) if core is not None else []
    covered = A[:, chosen].any(1) if chosen else np.zeros(n, bool)
    avail = np.ones(m, bool); avail[chosen] = False
    while len(chosen) < k:
        gain = A[~covered].sum(0).astype(float)
        gain[~avail] = -1
        if gain.max() <= 0:   # everyone covered: fill by approval count
            cnt = A.sum(0).astype(float); cnt[~avail] = -1
            c = int(np.argmax(cnt))
        else:
            c = int(np.argmax(gain))
        chosen.append(c); avail[c] = False; covered |= A[:, c]
    return np.sort(np.array(chosen))


def rule_hybrid(V, groups, est, k, k0):
    """Consensus core (k0 statements by min-group agreement) + greedy coverage completion."""
    core = rule_consmin(V, groups, est, k0)
    return rule_greedycc(V, k, core=core)


# ----------------------------------------------------------------- designer multiverse
def design_space(include_proportional=True, Ks=(2, 3, 4), imputes=('zero', 'mean'),
                 ests=('seen', 'decided', 'laplace'), centers=('stmt', 'part')):
    D = []
    for c in centers:
        for K in Ks:
            for imp in imputes:
                for est in ests:
                    for rule in ('consmin', 'gac'):
                        D.append(dict(family='consensus', rule=rule, K=K, impute=imp, est=est, center=c))
    for est in ests:
        D.append(dict(family='majoritarian', rule='av', K=1, impute='-', est=est))
    if include_proportional:
        D.append(dict(family='proportional', rule='seqpav', K=1, impute='-', est='-'))
        D.append(dict(family='proportional', rule='mes', K=1, impute='-', est='-'))
        D.append(dict(family='proportional', rule='greedycc', K=1, impute='-', est='-'))
        D.append(dict(family='hybrid', rule='hybrid', K=2, impute='zero', est='seen', center='stmt'))
    return D


def run_design(V, d, k, group_cache, seed=0, k0_frac=0.7):
    r = d['rule']
    if r in ('consmin', 'gac', 'hybrid'):
        key = (d['K'], d['impute'], d['center'])
        if key not in group_cache:
            group_cache[key] = polis_groups(V, d['K'], d['impute'], seed=seed, center=d['center'])
        g = group_cache[key]
        if r == 'consmin':
            return rule_consmin(V, g, d['est'], k)
        if r == 'gac':
            return rule_gac(V, g, d['est'], k)
        return rule_hybrid(V, g, d['est'], k, int(round(k0_frac * k)))
    if r == 'av':
        return rule_av(V, d['est'], k)
    if r == 'seqpav':
        return rule_seqpav(V, k)
    if r == 'mes':
        return rule_mes(V, k)
    if r == 'greedycc':
        return rule_greedycc(V, k)
    raise ValueError(r)


def label(d):
    if d['family'] in ('consensus',):
        return f"{d['rule']}|{d['center']}|K={d['K']}|{d['impute']}|{d['est']}"
    if d['rule'] == 'av':
        return f"av|{d['est']}"
    return d['rule']


# ----------------------------------------------------------------- variance decomposition
def decompose(I):
    """I: (D designs x B samples x m statements) 0/1 inclusion array.
    Two-way decomposition of inclusion variance, summed over statements:
      total = designer main effect + sample main effect + interaction.
    Returns shares."""
    I = I.astype(float)
    mu = I.mean((0, 1), keepdims=True)
    md = I.mean(1, keepdims=True)     # design means
    mb = I.mean(0, keepdims=True)     # sample means
    tot = ((I - mu) ** 2).mean((0, 1)).sum()
    des = ((md - mu) ** 2).mean((0, 1)).sum()
    sam = ((mb - mu) ** 2).mean((0, 1)).sum()
    inter = tot - des - sam
    return dict(total=tot, designer=des / tot, sample=sam / tot, interaction=inter / tot)


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / max(1, len(a | b))


# ----------------------------------------------------------------- representation audit
def jr_violation(V, W, k):
    """Justified representation on observed approvals (unseen = not approved).
    Returns size of the largest group of voters with no approved statement in W who
    all approve a common statement outside W, divided by n/k (>=1 means JR violated)."""
    A = approvals(V)
    n = len(A)
    unc = ~A[:, W].any(1)
    out = np.ones(A.shape[1], bool); out[W] = False
    if unc.sum() == 0:
        return 0.0
    best = A[unc][:, out].sum(0).max() if out.any() else 0
    return best / (n / k)


def coverage_stats(V, W, groups=None):
    A = approvals(V)
    S = ~np.isnan(V)
    nA = A[:, W].sum(1)
    nS = S[:, W].sum(1)
    frac = np.divide(nA, nS, out=np.full(len(V), np.nan), where=nS > 0)
    out = dict(uncovered=float((nA == 0).mean()),
               mean_endorse=float(np.nanmean(frac)))
    if groups is not None:
        gs = {}
        for g in np.unique(groups):
            gs[int(g)] = float(np.nanmean(frac[groups == g]))
        out['group_endorse'] = gs
        vals = list(gs.values())
        out['group_gap'] = float(max(vals) - min(vals))
    return out


# ----------------------------------------------------------------- theory helpers
def partition_floor(p, alpha):
    """Prop. 1: lowest possible within-group agreement of a statement with overall
    agreement p, over all groups containing at least a fraction alpha of voters."""
    return np.maximum(0.0, 1.0 - (1.0 - p) / alpha)


def swap_distances(I):
    """Mean Jaccard distance between constitutions when (a) the designer is swapped
    holding the participant sample fixed, (b) the sample is swapped holding the
    design fixed. I: (D x B x m)."""
    D, B, m = I.shape
    sets = [[np.where(I[d, b])[0] for b in range(B)] for d in range(D)]
    des, sam = [], []
    for b in range(B):
        for i in range(D):
            for j in range(i + 1, D):
                des.append(1 - jaccard(sets[i][b], sets[j][b]))
    for d in range(D):
        for b in range(B):
            for c in range(b + 1, B):
                sam.append(1 - jaccard(sets[d][b], sets[d][c]))
    return dict(designer_swap=float(np.mean(des)), sample_swap=float(np.mean(sam)))
