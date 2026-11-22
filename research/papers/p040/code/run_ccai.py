"""Authorship audit of the Collective Constitutional AI (CCAI) Polis data.
Usage: python3 run_ccai.py   (writes to ../results/ccai_*)"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, os.path.dirname(__file__))
from constlib import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'ccai')
RES = os.path.join(HERE, '..', 'results')
os.makedirs(RES, exist_ok=True)
K_SEATS = 58          # size of Anthropic's 2023 constitution, the stated size target of CCAI
B = int(os.environ.get('B', 40))
SEED = 20261122
t0 = time.time()
out = {}

V, cm, released_groups = load_ccai(DATA)
n, m = V.shape
S = ~np.isnan(V)
out['n_participants'] = int(n); out['n_statements'] = int(m)
out['votes_observed'] = int(S.sum())
out['density'] = float(S.mean())
out['agree_share'] = float((V == 1).sum() / S.sum())
out['disagree_share'] = float((V == -1).sum() / S.sum())
out['pass_share'] = float((V == 0).sum() / S.sum())

# ---------------------------------------------------------------- agenda channel
auth = cm['author-id'].to_numpy()
ts = cm['timestamp'].to_numpy()
first_part = ts[auth != 0].min()
is_seed = (auth == 0) & (ts < first_part)
is_owner_later = (auth == 0) & ~is_seed
out['seed_statements'] = int(is_seed.sum())
out['owner_account_later_statements'] = int(is_owner_later.sum())
part_auth = auth[auth != 0]
vc = pd.Series(part_auth).value_counts()
out['distinct_participant_authors'] = int(len(vc))
out['share_participants_authoring'] = float(len(vc) / n)
top10 = vc.head(max(1, int(round(0.10 * len(vc))))).sum()
out['share_stmts_by_top10pct_authors'] = float(top10 / len(part_auth))
out['max_stmts_single_participant'] = int(vc.max())
seen_cnt = S.sum(0)
out['seen_seed_mean'] = float(seen_cnt[is_seed].mean())
out['seen_nonseed_mean'] = float(seen_cnt[~is_seed].mean())
order = np.argsort(np.argsort(ts))
from scipy.stats import spearmanr
rho, pv = spearmanr(order, seen_cnt)
out['spearman_order_vs_seen'] = [float(rho), float(pv)]

# ---------------------------------------------------------------- grouping sanity
out['released_group_sizes'] = np.bincount(released_groups).tolist()
# Pipeline-integrity audit. The CCAI notebook sorts the vote matrix by participant id
# before PCA/k-means but writes the resulting labels positionally into the unsorted
# data frame. We replicate the notebook's clustering on the sorted matrix and compare.
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
pid = pd.read_csv(os.path.join(DATA, 'clean_votes.csv'))['participant'].to_numpy()
out['clean_votes_rows_sorted_by_participant'] = bool((np.diff(pid) > 0).all())
srt = np.argsort(pid, kind='stable')
aris_pos, aris_cor = [], []
for rs in range(5):
    Xs = np.nan_to_num(V[srt], nan=0.0)
    coords = PCA(2).fit(Xs.T).components_.T
    k100 = KMeans(100, random_state=rs, n_init=10).fit(coords)
    lab2 = KMeans(2, random_state=rs, n_init=10).fit(k100.cluster_centers_).labels_[k100.labels_]
    corr = np.empty_like(lab2); corr[srt] = lab2
    aris_pos.append(adjusted_rand_score(released_groups, lab2))
    aris_cor.append(adjusted_rand_score(released_groups, corr))
    if rs == 0:
        corrected_groups = corr
out['ARI_released_vs_positional_replica'] = [float(x) for x in aris_pos]
out['ARI_released_vs_correctly_aligned_replica'] = [float(x) for x in aris_cor]
def grp_agree(g):
    return [float((V[g == k] == 1).sum() / (~np.isnan(V[g == k])).sum()) for k in np.unique(g)]
out['agree_rate_by_group_released'] = grp_agree(released_groups)
out['agree_rate_by_group_corrected'] = grp_agree(corrected_groups)
out['corrected_group_sizes'] = np.bincount(corrected_groups).tolist()
# group separation diagnostic: mean |difference| in per-statement group agreement
def sep(g):
    gs = group_scores(V, g, 'seen'); return float(np.abs(gs[0] - gs[1]).mean())
out['group_separation_released'] = sep(released_groups)
out['group_separation_corrected'] = sep(corrected_groups)
perm_sep = []
for r in range(200):
    perm_sep.append(sep(np.random.default_rng(r).permutation(released_groups)))
out['group_separation_random_mean_sd'] = [float(np.mean(perm_sep)), float(np.std(perm_sep))]
out['group_separation_random_max'] = float(np.max(perm_sep))

# ---------------------------------------------------------------- threshold mode (CCAI-style)
thr = {}
groupings = [(K, imp, c) for c in ('stmt', 'part') for K in (2, 3, 4) for imp in ('zero', 'mean')]
gcache_full = {kk: polis_groups(V, kk[0], kk[1], seed=0, center=kk[2]) for kk in groupings}
thr_groups = dict(gcache_full)
thr_groups[('released', '-', '-')] = released_groups
thr_groups[('corrected', '-', '-')] = corrected_groups
for tau in (0.6, 0.65, 0.7, 0.72, 0.75, 0.8):
    row = {}
    for key, g in thr_groups.items():
        for est in ('seen', 'decided', 'laplace'):
            sel = np.where(group_scores(V, g, est).min(0) > tau)[0]
            row[f'{key[0]}|{key[1]}|{key[2]}|{est}'] = int(len(sel))
    thr[str(tau)] = row
out['threshold_mode_sizes'] = thr
# overlap of threshold-mode sets at tau = 0.72 across groupings/estimators
sets72 = {}
for key, g in thr_groups.items():
    for est in ('seen', 'decided', 'laplace'):
        sets72[f'{key[0]}|{key[1]}|{key[2]}|{est}'] = set(np.where(group_scores(V, g, est).min(0) > 0.72)[0])
ref = sets72['released|-|-|seen']
cor = sets72['corrected|-|-|seen']
out['tau072_corrected_size'] = len(cor)
out['tau072_released_vs_corrected_jaccard'] = len(ref & cor) / len(ref | cor)
out['tau072_in_released_not_corrected'] = int(len(ref - cor))
out['tau072_in_corrected_not_released'] = int(len(cor - ref))
overall072 = set(np.where(agree_rate(V, 'seen') > 0.72)[0])
out['tau072_overall_majority_size'] = len(overall072)
out['tau072_released_vs_overall_jaccard'] = len(ref & overall072) / len(ref | overall072)
out['tau072_corrected_vs_overall_jaccard'] = len(cor & overall072) / len(cor | overall072)
# top-58 under min-consensus with released vs corrected groups
W_rel = rule_consmin(V, released_groups, 'seen', K_SEATS)
W_cor = rule_consmin(V, corrected_groups, 'seen', K_SEATS)
out['top58_released_vs_corrected_jaccard'] = jaccard(W_rel, W_cor)
ex = [dict(id=cm['comment-id'][i], text=cm['comment-body'][i],
           agree_released=[float(x) for x in group_scores(V, released_groups, 'seen')[:, i]],
           agree_corrected=[float(x) for x in group_scores(V, corrected_groups, 'seen')[:, i]])
      for i in sorted(ref - cor)]
pd.DataFrame(ex).to_csv(os.path.join(RES, 'ccai_misalignment_examples.csv'), index=False)
out['tau072_reference_size'] = len(ref)
out['tau072_union_size'] = len(set.union(*sets72.values()))
out['tau072_intersection_size'] = len(set.intersection(*sets72.values()))
out['tau072_jaccard_to_reference'] = {k: len(v & ref) / len(v | ref) for k, v in sets72.items()}

# ---------------------------------------------------------------- Prop 1: partition leverage
alpha = released_groups.tolist().count(1) / n
p_all = agree_rate(V, 'seen')
floor = partition_floor(p_all, alpha)
out['alpha_minority'] = alpha
out['prop1_n_overall_above_072'] = int((p_all > 0.72).sum())
out['prop1_n_vulnerable_072'] = int(((p_all > 0.72) & (floor <= 0.72)).sum())
out['prop1_n_robust_072'] = int(((p_all > 0.72) & (floor > 0.72)).sum())

# ---------------------------------------------------------------- multiverse on full data
D = design_space()
labels = [label(d) for d in D]
fam = [d['family'] for d in D]
W_full = []
for d in D:
    W_full.append(run_design(V, d, K_SEATS, gcache_full, seed=0))
Dn = len(D)
J = np.zeros((Dn, Dn))
for i in range(Dn):
    for j in range(Dn):
        J[i, j] = jaccard(W_full[i], W_full[j])
np.save(os.path.join(RES, 'ccai_multiverse_jaccard.npy'), J)
inc_full = np.zeros((Dn, m), int)
for i, W in enumerate(W_full):
    inc_full[i, W] = 1
cons_idx = [i for i, f in enumerate(fam) if f == 'consensus']
pf = inc_full[cons_idx].mean(0)
out['consensus_multiverse_n_designs'] = len(cons_idx)
out['consensus_always_in'] = int((pf == 1).sum())
out['consensus_sometimes_in'] = int(((pf > 0) & (pf < 1)).sum())
out['consensus_union'] = int((pf > 0).sum())
pa = inc_full.mean(0)
out['all_always_in'] = int((pa == 1).sum())
out['all_union'] = int((pa > 0).sum())
iu = np.triu_indices(len(cons_idx), 1)
Jc = J[np.ix_(cons_idx, cons_idx)]
out['consensus_pairwise_jaccard_mean'] = float(Jc[iu].mean())
out['consensus_pairwise_jaccard_min'] = float(Jc[iu].min())
iu2 = np.triu_indices(Dn, 1)
out['all_pairwise_jaccard_mean'] = float(J[iu2].mean())
out['all_pairwise_jaccard_min'] = float(J[iu2].min())

# representation audit per design (reference groups = released Polis groups)
audit = []
for d, lab, W in zip(D, labels, W_full):
    cs = coverage_stats(V, W, released_groups)
    audit.append(dict(design=lab, family=d['family'], jr_ratio=jr_violation(V, W, K_SEATS),
                      uncovered=cs['uncovered'], mean_endorse=cs['mean_endorse'],
                      endorse_A=cs['group_endorse'][0], endorse_B=cs['group_endorse'][1],
                      group_gap=cs['group_gap'],
                      seed_share=float(is_seed[W].mean()),
                      owner_share=float((auth[W] == 0).mean()),
                      mean_seen=float(seen_cnt[W].mean())))
pd.DataFrame(audit).to_csv(os.path.join(RES, 'ccai_design_audit.csv'), index=False)
out['seed_share_in_pool'] = float(is_seed.mean())

# Prop 3 check for hybrid
g2 = gcache_full[(2, 'zero', 'stmt')]
k0 = int(round(0.7 * K_SEATS))
core = rule_consmin(V, g2, 'seen', k0)
A = approvals(V)
U = (~A[:, core].any(1)).sum()
out['prop3_k0'] = k0
out['prop3_uncovered_after_core'] = int(U)
out['prop3_bound'] = float((K_SEATS - k0 + 1) * n / K_SEATS)
out['prop3_condition_holds'] = bool(U < (K_SEATS - k0 + 1) * n / K_SEATS)

# ---------------------------------------------------------------- bootstrap x multiverse
rng = np.random.default_rng(SEED)

def boot_inclusion(Vsub_fn, designs, Bn, kseats):
    I = np.zeros((len(designs), Bn, m), np.int8)
    for b in range(Bn):
        Vb = Vsub_fn(b)
        cache = {}
        for i, d in enumerate(designs):
            W = run_design(Vb, d, kseats, cache, seed=b)
            I[i, b, W] = 1
    return I

idx_boot = [rng.integers(0, n, n) for _ in range(B)]
I_boot = boot_inclusion(lambda b: V[idx_boot[b]], D, B, K_SEATS)
np.save(os.path.join(RES, 'ccai_boot_inclusion.npy'), I_boot)
out['decomp_full_multiverse'] = decompose(I_boot)
out['decomp_consensus_multiverse'] = decompose(I_boot[cons_idx])
out['swap_full_multiverse'] = swap_distances(I_boot)
out['swap_consensus_multiverse'] = swap_distances(I_boot[cons_idx])
# restricted: only the technical knobs, one at a time around CCAI-like baseline
def idx_of(**kw):
    for i, d in enumerate(D):
        if all(d.get(a) == v for a, v in kw.items()):
            return i
B0 = dict(K=2, impute='zero', est='seen', center='part')   # CCAI-notebook-like baseline
def var(**kw):
    d = dict(B0); d.update(kw); return d
knob_sets = {
    'estimator_only': [idx_of(rule='consmin', **var(est=e)) for e in ('seen', 'decided', 'laplace')],
    'K_only': [idx_of(rule='consmin', **var(K=K)) for K in (2, 3, 4)],
    'impute_only': [idx_of(rule='consmin', **var(impute=i)) for i in ('zero', 'mean')],
    'centering_only': [idx_of(rule='consmin', **var(center=c)) for c in ('part', 'stmt')],
    'consmin_vs_gac': [idx_of(rule=r, **B0) for r in ('consmin', 'gac')],
    'rule_family': [idx_of(rule='consmin', **B0), idx_of(rule='av', est='seen'),
                    idx_of(rule='seqpav'), idx_of(rule='mes')],
}
out['decomp_by_knob'] = {k: dict(decompose(I_boot[v]), **swap_distances(I_boot[v])) for k, v in knob_sets.items()}
# per-design sampling stability: mean Jaccard of bootstrap constitutions to full-data constitution
stab = []
for i in range(Dn):
    js = [jaccard(np.where(I_boot[i, b])[0], W_full[i]) for b in range(B)]
    stab.append(float(np.mean(js)))
out['boot_jaccard_to_full_by_design'] = dict(zip(labels, stab))
out['boot_jaccard_mean_consensus'] = float(np.mean([stab[i] for i in cons_idx]))

# ---------------------------------------------------------------- sample-size curve
sizes = [100, 200, 400, 700]
Bs = int(os.environ.get('BS', 25))
curve = {}
for ns in sizes:
    idxs = [rng.integers(0, n, ns) for _ in range(Bs)]   # bootstrap of size ns
    I = boot_inclusion(lambda b: V[idxs[b]], D, Bs, K_SEATS)
    curve[ns] = dict(full=decompose(I), consensus=decompose(I[cons_idx]),
                     sample_var_abs=float(decompose(I)['sample'] * decompose(I)['total']),
                     designer_var_abs=float(decompose(I)['designer'] * decompose(I)['total']),
                     cons_sample_abs=float(decompose(I[cons_idx])['sample'] * decompose(I[cons_idx])['total']),
                     cons_designer_abs=float(decompose(I[cons_idx])['designer'] * decompose(I[cons_idx])['total']),
                     cons_interaction_abs=float(decompose(I[cons_idx])['interaction'] * decompose(I[cons_idx])['total']),
                swap_full=swap_distances(I), swap_consensus=swap_distances(I[cons_idx]))
dfull = decompose(I_boot); dcons = decompose(I_boot[cons_idx])
curve[n] = dict(full=dfull, consensus=dcons,
                sample_var_abs=dfull['sample'] * dfull['total'], designer_var_abs=dfull['designer'] * dfull['total'],
                cons_sample_abs=dcons['sample'] * dcons['total'], cons_designer_abs=dcons['designer'] * dcons['total'],
                cons_interaction_abs=dcons['interaction'] * dcons['total'],
                swap_full=out['swap_full_multiverse'], swap_consensus=out['swap_consensus_multiverse'])
out['sample_size_curve'] = {str(k): v for k, v in curve.items()}

# ---------------------------------------------------------------- k sensitivity
ksens = {}
for kk in (30, 58, 100):
    Wk = [run_design(V, d, kk, gcache_full, seed=0) for d in D]
    Jk = np.array([[jaccard(a, b) for b in Wk] for a in Wk])
    ksens[kk] = dict(consensus_mean_jaccard=float(Jk[np.ix_(cons_idx, cons_idx)][iu].mean()),
                     all_mean_jaccard=float(Jk[iu2].mean()))
out['k_sensitivity'] = {str(k): v for k, v in ksens.items()}

out['labels'] = labels
out['families'] = fam
out['runtime_sec'] = time.time() - t0
with open(os.path.join(RES, 'ccai_results.json'), 'w') as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ('labels', 'families', 'boot_jaccard_to_full_by_design', 'tau072_jaccard_to_reference', 'threshold_mode_sizes')}, indent=1, default=float))
