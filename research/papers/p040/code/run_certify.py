"""Certification analysis: which seats of a public-input constitution do the votes
statistically determine?  (CCAI + public Polis conversations; planning simulation.)
Usage: python3 run_certify.py   (needs ../results/ccai_realigned_groups.npy and
ccai_boot_inclusion.npy from run_ccai.py)
Outputs: ../results/certify_ccai.json, certify_opendata.csv, certify_planning.csv,
certify_ccai_statements.csv"""
import json, os, sys, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, gaussian_kde, betabinom
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
from constlib import load_ccai, load_polis, design_space, label
from certify import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data'); RES = os.path.join(HERE, '..', 'results')
BB = int(os.environ.get('BB', 1000))
K = 58
t0 = time.time()
out = {}

def summarize(r, k):
    return dict(cert_in=int(r['cin'].sum()), cert_out=int(r['cout'].sum()), uncertified=int(r['unc'].sum()),
                crit_median=float(np.median(r['crit'])),
                stable_in=int((r['pi'] >= 0.95).sum()), stable_out=int((r['pi'] <= 0.05).sum()),
                contested=int(((r['pi'] > 0.05) & (r['pi'] < 0.95)).sum()), k=k)

# ======================================================================= CCAI
V, cm, released = load_ccai(os.path.join(DATA, 'ccai'))
n, M = V.shape
S = ~np.isnan(V)
seen = S.sum(0)
realigned = np.load(os.path.join(RES, 'ccai_realigned_groups.npy'))
auth = cm['author-id'].to_numpy(); ts = cm['timestamp'].to_numpy()
is_seed = (auth == 0) & (ts < ts[auth != 0].min())
scores = {'av|seen': make_score('av', 'seen'), 'av|decided': make_score('av', 'decided'),
          'consmin|realigned|seen': make_score('consmin', 'seen', realigned),
          'consmin|released|seen': make_score('consmin', 'seen', released)}
cc = {}
stmt_rows = {}
for name, f in scores.items():
    full, T = bootstrap_scores(V, f, BB, np.random.default_rng(11))
    res = {}
    for mode in ('marginal', 'simultaneous'):
        r = certify(V, None, K, T=T, full=full, mode=mode)
        res[mode] = summarize(r, K)
        if mode == 'marginal':
            rm = r
    # seat-count curve over k (marginal)
    res['k_curve'] = {}
    for kk in (10, 20, 30, 58, 100, 150):
        r = certify(V, None, kk, T=T, full=full, mode='marginal')
        res['k_curve'][kk] = summarize(r, kk)
    th = certify_threshold(full, T, 0.72)
    res['threshold_072'] = dict(point_above=int((full > 0.72).sum()), cert_above=int(th['above'].sum()),
                                cert_below=int(th['below'].sum()),
                                uncertain=int(((~th['above']) & (~th['below'])).sum()), crit=float(th['crit']))
    stab_in = rm['pi'] >= 0.95
    res['seed_share_stable_in'] = float(is_seed[stab_in].mean()) if stab_in.any() else None
    res['n_seed_stable_in'] = int(is_seed[stab_in].sum())
    res['mean_seen_stable_in'] = float(seen[stab_in].mean()) if stab_in.any() else None
    res['mean_seen_contested'] = float(seen[(rm['pi'] > .05) & (rm['pi'] < .95)].mean())
    res['spearman_seen_vs_rankCIwidth'] = float(spearmanr(seen, rm['rhi'] - rm['rlo'])[0])
    cc[name] = res
    stmt_rows[name] = rm
    print(name, res['marginal'], res['simultaneous'], res['threshold_072'], f'{time.time()-t0:.0f}s', flush=True)
out['ccai'] = cc
r0 = stmt_rows['av|seen']
pd.DataFrame(dict(comment_id=cm['comment-id'], seed=is_seed, seen=seen, agree_seen=r0['full'],
                  rank_lo=r0['rlo'], rank_hi=r0['rhi'], boot_top58=r0['pi'],
                  text=cm['comment-body'])).sort_values('agree_seen', ascending=False).to_csv(
    os.path.join(RES, 'certify_ccai_statements.csv'), index=False)

# ------------------------------------------------ per-design stability on the designer multiverse
I = np.load(os.path.join(RES, 'ccai_boot_inclusion.npy'))     # D x B x M
o = json.load(open(os.path.join(RES, 'ccai_results.json')))
fam = o['families']; labs = o['labels']
Dn, Bn, _ = I.shape
pi_d = I.mean(1)                                              # D x M
stable = pi_d >= 0.95
per_design = []
cons = [i for i, f in enumerate(fam) if f == 'consensus']
W_full = np.zeros((Dn, M), bool)
D = design_space()
from constlib import run_design
cache = {}
for i, d in enumerate(D):
    W_full[i, run_design(V, d, K, cache, seed=0)] = True
always_cons = W_full[cons].all(0)
for i in range(Dn):
    W = W_full[i]
    per_design.append(dict(design=labs[i], family=fam[i], stable_in_seats=int((W & stable[i]).sum()),
                           noise_decided_seats=int((W & ~stable[i]).sum()),
                           stable_and_in_all_consensus=int((W & stable[i] & always_cons).sum())))
pdf = pd.DataFrame(per_design)
pdf.to_csv(os.path.join(RES, 'certify_ccai_by_design.csv'), index=False)
out['ccai_by_design'] = {f: dict(stable_in_mean=float(g.stable_in_seats.mean()), stable_in_min=int(g.stable_in_seats.min()),
                                 stable_in_max=int(g.stable_in_seats.max()))
                         for f, g in pdf.groupby('family')}
out['ccai_by_design']['B'] = int(Bn)
# designer disagreements: are they on statements the votes leave undetermined?
# for each pair of consensus designs, the symmetric difference of their full-data sets;
# share of those statements that are stable-in (pi>=.95) under *both* designs' bootstraps
sd_tot, sd_unstable = 0, 0
for a in cons:
    for b in cons:
        if a < b:
            diff = W_full[a] ^ W_full[b]
            sd_tot += diff.sum()
            sd_unstable += (diff & ~(stable[a] & stable[b])).sum()
out['ccai_designer_disagreement_on_unstable_share'] = float(sd_unstable / max(1, sd_tot))
# statements stable-in under at least one consensus design but excluded (full data) by some other design
out['ccai_stable_in_any_consensus'] = int(stable[cons].any(0).sum())
out['ccai_stable_in_all_consensus'] = int(stable[cons].all(0).sum())

# ------------------------------------------------ lower bound (Theorem 1) on CCAI, EB truth
def fit_betabinom(a, s):
    def nll(th):
        al, be = np.exp(th)
        return -betabinom.logpmf(a, s, al, be).sum()
    r = minimize(nll, [np.log(5), np.log(1)], method='Nelder-Mead')
    return np.exp(r.x)
ag = (V == 1).sum(0)
al, be = fit_betabinom(ag, seen)
p_eb = (ag + al) / (seen + al + be)
out['ccai_eb_prior'] = dict(alpha=float(al), beta=float(be), prior_mean=float(al / (al + be)))
def lb_band(p_true, m_s, k, alpha=0.05, beta=0.2):
    """Theorem 1: statement s in the top-k with p_s - q_{k+1} < (1-alpha-beta) sqrt(2 q(1-q)/m_s)
    cannot be certified with power 1-beta by any level-alpha procedure."""
    o_ = np.sort(p_true)[::-1]
    q = o_[k]                          # (k+1)-th largest
    delta = (1 - alpha - beta) * np.sqrt(2 * q * (1 - q) / m_s)
    inside = p_true > q
    return int((inside & (p_true - q < delta)).sum()), float(q)
for lab_, pt in (('eb', p_eb), ('raw', ag / seen)):
    cnt, q = lb_band(pt, seen, K)
    out[f'ccai_theorem1_uncertifiable_seats_{lab_}'] = dict(count=cnt, q=q)

# ------------------------------------------------ planning simulation: votes per statement needed
plan = []
rng = np.random.default_rng(5)
v_pp = S.sum(1).mean()
for truth_name, pt in (('eb', p_eb), ('raw', ag / seen)):
    for m_s in (100, 200, 400, 800, 1600, 3200, 6400):
        for rep in range(3):
            a_ = rng.binomial(m_s, pt)
            ph = a_ / m_s
            T = rng.binomial(m_s, np.broadcast_to(ph, (400, M))) / m_s   # parametric bootstrap
            r = certify(None, None, K, T=T, full=ph, mode='marginal')
            lb, _ = lb_band(pt, np.full(M, m_s), K)
            plan.append(dict(truth=truth_name, m=m_s, rep=rep, n_equiv=m_s * M / v_pp,
                             cert_in=int(r['cin'].sum()), stable_in=int((r['pi'] >= .95).sum()),
                             contested=int(((r['pi'] > .05) & (r['pi'] < .95)).sum()),
                             theorem1_uncertifiable=lb))
pl = pd.DataFrame(plan)
pl.to_csv(os.path.join(RES, 'certify_planning.csv'), index=False)
out['planning'] = pl.groupby(['truth', 'm']).mean(numeric_only=True).round(2).reset_index().to_dict('records')
print('planning done', f'{time.time()-t0:.0f}s', flush=True)

# ======================================================================= openData
rows = []
for d in sorted(glob.glob(os.path.join(DATA, 'openData', '*'))):
    pvf, cmf = os.path.join(d, 'participants-votes.csv'), os.path.join(d, 'comments.csv')
    if not os.path.exists(pvf):
        continue
    Vd, cmd = load_polis(pvf, cmf, min_votes=7, min_seen=10)
    nd, Md = Vd.shape
    if nd < 100 or Md < 40:
        continue
    k = int(np.clip(round(0.2 * Md), 10, 58))
    Sd = ~np.isnan(Vd)
    full, T = bootstrap_scores(Vd, make_score('av', 'seen'), BB, np.random.default_rng(12))
    rm = certify(Vd, None, k, T=T, full=full, mode='marginal')
    rs = certify(Vd, None, k, T=T, full=full, mode='simultaneous')
    sd_ = Sd.sum(0)
    q = np.sort(full)[::-1][k]
    try:
        fq = float(gaussian_kde(full)(q)[0])
    except Exception:
        fq = np.nan
    a_ = (Vd == 1).sum(0)
    al_, be_ = fit_betabinom(a_, sd_)
    peb = (a_ + al_) / (sd_ + al_ + be_)
    lb, qeb = lb_band(peb, sd_, k)
    rows.append(dict(conversation=os.path.basename(d), n=nd, M=Md, k=k, votes=int(Sd.sum()),
                     m_mean=float(sd_.mean()), m_median=float(np.median(sd_)), votes_per_participant=float(Sd.sum(1).mean()),
                     agree_share=float((Vd == 1).sum() / Sd.sum()), cutoff=float(q), density_at_cutoff=fq,
                     cert_in_marginal=int(rm['cin'].sum()), cert_in_simult=int(rs['cin'].sum()),
                     cert_out_marginal=int(rm['cout'].sum()),
                     stable_in=int((rm['pi'] >= .95).sum()), contested=int(((rm['pi'] > .05) & (rm['pi'] < .95)).sum()),
                     theorem1_uncertifiable=lb))
    print(rows[-1], f'{time.time()-t0:.0f}s', flush=True)
# add CCAI as a row (same statistics, av|seen)
rc = stmt_rows['av|seen']
q = np.sort(rc['full'])[::-1][K]
lbc, _ = lb_band(p_eb, seen, K)
rows.append(dict(conversation='CCAI', n=n, M=M, k=K, votes=int(S.sum()), m_mean=float(seen.mean()),
                 m_median=float(np.median(seen)), votes_per_participant=float(S.sum(1).mean()),
                 agree_share=float((V == 1).sum() / S.sum()), cutoff=float(q),
                 density_at_cutoff=float(gaussian_kde(rc['full'])(q)[0]),
                 cert_in_marginal=int(rc['cin'].sum()), cert_in_simult=cc['av|seen']['simultaneous']['cert_in'],
                 cert_out_marginal=int(rc['cout'].sum()),
                 stable_in=int((rc['pi'] >= .95).sum()), contested=int(((rc['pi'] > .05) & (rc['pi'] < .95)).sum()),
                 theorem1_uncertifiable=lbc))
od = pd.DataFrame(rows)
od['stable_share'] = od.stable_in / od.k
od['cert_share'] = od.cert_in_marginal / od.k
od['contested_per_seat'] = od.contested / od.k
od['kappa'] = od.density_at_cutoff * np.sqrt(od.cutoff * (1 - od.cutoff) / od.m_median) * od.M / od.k
od.to_csv(os.path.join(RES, 'certify_opendata.csv'), index=False)
law = {}
lg = np.log
law['loglog_M_on_n'] = np.polyfit(lg(od.n), lg(od.M), 1).tolist()
law['loglog_mmedian_on_n'] = np.polyfit(lg(od.n), lg(od.m_median), 1).tolist()
law['spearman_n_mmedian'] = list(map(float, spearmanr(od.n, od.m_median)))
for y in ('stable_share', 'cert_share', 'contested_per_seat'):
    for x in ('n', 'm_median', 'kappa'):
        law[f'spearman_{y}_vs_{x}'] = list(map(float, spearmanr(od[x], od[y])))
# OLS: logit-ish stable share on log n and log m
import numpy.linalg as la
X = np.column_stack([np.ones(len(od)), lg(od.n), lg(od.m_median)])
y = od.contested_per_seat.values
beta, res_, *_ = la.lstsq(X, lg(y), rcond=None)
resid = lg(y) - X @ beta
s2 = resid @ resid / (len(y) - 3)
se = np.sqrt(np.diag(s2 * la.inv(X.T @ X)))
law['ols_log_contested_per_seat'] = dict(coef=beta.tolist(), se=se.tolist(), t=(beta / se).tolist(),
                                         terms=['const', 'log n', 'log m_median'])
out['opendata_law'] = law
out['runtime_sec'] = time.time() - t0
json.dump(out, open(os.path.join(RES, 'certify_ccai.json'), 'w'), indent=1, default=float)
print(json.dumps(law, indent=1))
print(od[['conversation', 'n', 'M', 'k', 'm_median', 'stable_in', 'contested', 'cert_in_marginal', 'theorem1_uncertifiable', 'kappa']].to_string())
