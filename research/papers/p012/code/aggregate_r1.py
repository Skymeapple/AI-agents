"""Aggregate results/r1/*.json into results/r1/summary_r1.txt and fig_r1_*.png.
All intervals are 95% t-intervals across worlds (n = 5 for base, 3 otherwise)."""
import json, os, glob, collections, numpy as np
from scipy.stats import t as tdist
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from theory import auroc_ceiling
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'r1')
runs = collections.defaultdict(list)
for f in sorted(glob.glob(f'{D}/*_w*.json')):
    d = json.load(open(f)); runs[d['variant']].append(d)
ORDER = [v for v in ['base', 'early', 'small', 'comp', 'matched', 'matched_early', 'matched_ls'] if v in runs]
L = []
def P(*a): L.append(' '.join(str(x) for x in a))
def ci(v):
    v = np.array([x for x in v if x is not None], float)
    if len(v) == 0: return '   n/a     '
    hw = tdist.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
    return f'{v.mean():.3f}±{hw:.3f}'
def mean(v): return float(np.mean([x for x in v if x is not None]))
P('variants and #worlds:', {v: len(runs[v]) for v in ORDER})

P('\n== T-G cells / memorization regime (member 0; acc and mean MSP) ==')
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    P(f'{v:14s}', ' '.join(f"{k}: acc {ci([r['cells'][k]['acc'] for r in R])} msp {ci([r['cells'][k]['msp'] for r in R])} |"
                           for k in R[0]['cells']), f"ECE_det {ci([r['ece_det'] for r in R])}")
    gt = R[0]['good_turing']
    P(' ' * 15 + 'Good-Turing:', ' '.join(f'{k} {ci([r["good_turing"][k] for r in R])}' for k in gt))
for v in ['base', 'matched']:
    if v in runs and 'bayes' in runs[v][0]:
        R = [d['bayes'] for d in runs[v]]
        P(f'{v+"/bayes":14s}', ' '.join(f"{k}: acc {ci([r['cells'][k]['acc'] for r in R])} msp {ci([r['cells'][k]['msp'] for r in R])} |"
                                       for k in R[0]['cells']), f"ECE_det {ci([r['ece_det'] for r in R])}")

P('\n== T-A original E1 design (positives det & G=1, negatives alea c>=20 / det G=0) incl. count baselines ==')
for v in ORDER:
    R = [d['mlp'] for d in runs[v] if d['mlp']['E1_original']]
    if not R: continue
    P(f'-- {v} (n_worlds={len(R)}; frac of positives from never-seen entities {ci([r["E1_original_frac_pos_unseen_entity"] for r in R])})')
    for k in ['msp', 'entropy', 'ens_entropy', 'ens_mi', 'neg_log_count', 'neg_log_factcount']:
        P(f'   {k:18s}', ' '.join(f'a={s["alpha"]:.2f}:{ci([r["E1_original"][i][k] for r in R])}' for i, s in enumerate(R[0]['E1_original'])))

P('\n== T-B exposure-matched E1 (familiar entities, negatives matched on entity-count histogram) ==')
for v in ORDER:
    for ln in ['mlp', 'bayes']:
        R = [d[ln] for d in runs[v] if ln in d and d[ln]['E1_matched']]
        if not R: continue
        P(f'-- {v}/{ln}  n(pos,neg_det,neg_alea) per world ~ {[int(np.mean([r["E1_matched_n"][k] for r in R])) for k in ["pos","neg_det","neg_alea"]]}')
        keys = ['msp', 'entropy', 'best_qonly', 'ens_mi', 'ens_entropy', 'neg_log_count', 'neg_log_factcount'] if ln == 'mlp' else ['msp', 'entropy', 'best_qonly', 'neg_log_count', 'neg_log_factcount']
        for k in keys:
            P(f'   {k:18s}', ' '.join(f'a={s["alpha"]:.1f}:{ci([r["E1_matched"][i][k] for r in R])}' for i, s in enumerate(R[0]['E1_matched'])))

P('\n== T-C is Prop. 1 binding? delta-hat (train-fold threshold), q-only AUROC pos-vs-aleatoric, ceiling at alpha=1 ==')
for v in ORDER:
    for ln in ['mlp', 'bayes']:
        R = [d[ln] for d in runs[v] if ln in d and d[ln]['delta_matched']]
        if not R: continue
        dh = [r['delta_matched']['delta_hat'] for r in R]
        lo = [r['delta_matched']['ci90'][0] for r in R]; hi = [r['delta_matched']['ci90'][1] for r in R]
        bq = [r['E1_matched'][-1]['best_qonly'] for r in R]; a0 = [r['E1_matched'][0]['best_qonly'] for r in R]
        ceil = [a0_ * 0 + auroc_ceiling(1.0, d_) for a0_, d_ in zip(a0, dh)]
        do = [r['delta_original']['delta_hat'] for r in R if r['delta_original']]
        P(f'{v+"/"+ln:22s} delta_hat(matched) {ci(dh)} [mean per-world 90% boot CI {np.mean(lo):.3f},{np.mean(hi):.3f}]'
          f'  auc(pos vs alea) {ci([r["delta_matched"]["auc"] for r in R])}  best q-only @a=1 {ci(bq)}  ceiling@a=1 {ci(ceil)}'
          f'  | delta_hat(original E1) {ci(do)}  ECE {ci([r["ece_det"] for r in R])}')

P('\n== T-D within-stratum AUROC for detecting wrong answers (held-out entities) ==')
P('   strata: FU familiar&unseen-fact, RU rare&unseen-fact, UE unseen entity, FS familiar&seen, RS rare&seen; n_wrong/n per world:')
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    P(f'   {v:14s}', {k: [int(np.mean([r['stratum_n_wrong'][k][0] for r in R])), int(np.mean([r['stratum_n_wrong'][k][1] for r in R]))] for k in R[0]['stratum_n_wrong']})
cols = ['natural', 'dissociated', 'FU', 'RU', 'UE', 'FS', 'RS']
def probe_rows(R, feat, pid, lam=1.0):
    return [[p for p in r['probes'] if p['feat'] == feat and p['pi_dis'] == pid and p['lam'] == lam] for r in R]
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    P(f'-- {v}')
    P('   ' + ' ' * 26 + ' '.join(f'{c:>12s}' for c in cols))
    for feat, pid in [('hidden', 0.0), ('hidden', 0.5), ('hidden+output', 0.0), ('hidden+output', 0.5), ('output_only', 0.0)]:
        pr = probe_rows(R, feat, pid)
        vals = {c: [mean([p['within'][c] for p in prs]) if any(p['within'][c] is not None for p in prs) else None for prs in pr] for c in cols}
        P(f'   probe {feat:13s} pi_d={pid:<4}', ' '.join(f'{ci(vals[c]):>12s}' for c in cols))
    for b in ['msp', 'ens_mi', 'ens_entropy', 'neg_log_count', 'neg_log_factcount', 'type_default_disagree']:
        P(f'   {b:26s}', ' '.join(f'{ci([r["baselines"][b][c] for r in R]):>12s}' for c in cols))

P('\n== T-E Prop. 2 decomposition (hidden-state probe; false-"knows" on familiar & wrong) ==')
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    if 'geometry' not in R[0]: continue
    P(f'-- {v}: |a| {ci([r["geometry"]["a"] for r in R])} |b| {ci([r["geometry"]["b"] for r in R])} cos {ci([r["geometry"]["cos"] for r in R])} pi_natural {ci([r["pi_natural"] for r in R])}')
    for pid in [0.0, 0.1, 0.5]:
        pr = probe_rows(R, 'hidden', pid)
        if not pr[0] or 'pred_fk_prior' not in pr[0][0]: continue
        g = lambda k: [mean([p[k] for p in prs]) for prs in pr]
        P(f'   pi_d={pid:<4} pi_eff {ci(g("pi_eff"))} p1 {ci(g("p1"))} | observed(logistic) {ci(g("false_know"))}  empirical-LDA {ci(g("lda_emp_fk"))}'
          f'  Prop2(balanced) {ci(g("pred_fk_balanced"))}  Prop2(class-prior corrected) {ci(g("pred_fk_prior"))}')

obs, bal, lda, pri = [], [], [], []
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    for pid in [0.0, 0.1, 0.5]:
        for r in R:
            ps_ = [p for p in r['probes'] if p['feat'] == 'hidden' and p['pi_dis'] == pid and p['lam'] == 1.0 and 'pred_fk_prior' in p]
            if ps_:
                obs.append(mean([p['false_know'] for p in ps_])); bal.append(mean([p['pred_fk_balanced'] for p in ps_]))
                lda.append(mean([p['lda_emp_fk'] for p in ps_])); pri.append(mean([p['pred_fk_prior'] for p in ps_]))
obs, bal, lda, pri = map(np.array, (obs, bal, lda, pri))
P(f'   pooled over all variants x worlds x pi_d (n={len(obs)}): r(obs,Prop2 balanced)={np.corrcoef(obs,bal)[0,1]:.3f} MAE={np.abs(obs-bal).mean():.3f} mean signed={np.mean(obs-bal):+.3f};'
  f' r(obs,prior-corrected)={np.corrcoef(obs,pri)[0,1]:.3f} MAE={np.abs(obs-pri).mean():.3f}; r(obs,empirical LDA)={np.corrcoef(obs,lda)[0,1]:.3f} MAE={np.abs(obs-lda).mean():.3f}')
mono = []
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    fk = [np.mean([mean([p['false_know'] for p in r['probes'] if p['feat'] == 'hidden' and p['pi_dis'] == pid and p['lam'] == 1.0]) for r in R]) for pid in [0.0, 0.1, 0.5]]
    mono.append(f'{v}:{"/".join(f"{x:.2f}" for x in fk)}')
P('   FK at pi_d = 0/0.1/0.5 (monotone decrease predicted):', ' '.join(mono))

P('\n== T-F probe ablation (hidden; L2 strength, threshold 0.5 vs Youden-on-train) ==')
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    for pid in [0.0, 0.5]:
        for lam in [0.01, 1.0, 100.0]:
            pr = probe_rows(R, 'hidden', pid, lam)
            if not pr[0]: continue
            g = lambda k: [mean([p[k] for p in prs]) for prs in pr]
            gw = lambda c: [mean([p['within'][c] for p in prs]) for prs in pr]
            P(f'   {v:14s} pi_d={pid:<4} lam={lam:<6} AUROC nat {ci(gw("natural"))} dis {ci(gw("dissociated"))} | FK@0.5 {ci(g("false_know"))} FK@Youden {ci(g("false_know_youden"))}'
              f' | MK@0.5 {ci(g("missed_know"))} MK@Youden {ci(g("missed_know_youden"))}')

open(f'{D}/summary_r1.txt', 'w').write('\n'.join(L)); print('\n'.join(L))

# ---------------- figures
# Fig R1-a: Prop. 1 test: best q-only AUROC vs alpha in the exposure-matched design, with ceilings at delta_hat
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
sty = {('base', 'mlp'): ('#c0392b', 'MLP, original world'), ('matched', 'mlp'): ('#e67e22', 'MLP, shape-matched world'),
       ('matched_ls', 'mlp'): ('#8e44ad', 'MLP + label smoothing, matched'), ('base', 'bayes'): ('#2c7fb8', 'Bayes ref., original world'),
       ('matched', 'bayes'): ('#16a085', 'Bayes ref., shape-matched world')}
for (v, ln), (col, lab) in sty.items():
    if v not in runs or ln not in runs[v][0]: continue
    R = [d[ln] for d in runs[v]]
    al = [s['alpha'] for s in R[0]['E1_matched']]
    ax[0].plot(al, [np.mean([r['E1_matched'][i]['best_qonly'] for r in R]) for i in range(len(al))], 'o-', c=col, label=lab)
    dh = np.mean([r['delta_matched']['delta_hat'] for r in R]); A0 = np.mean([r['E1_matched'][0]['best_qonly'] for r in R])
    ax[0].plot(al, [(1 - a) * A0 + a * auroc_ceiling(1, dh) for a in al], ':', c=col, lw=1)
ax[0].set_xlabel(r'aleatoric share $\alpha$ of known items'); ax[0].set_ylabel('AUROC of best trained q-only detector')
ax[0].set_title('Exposure-matched E1: Prop. 1 binds only for\nthe calibrated learner in the shape-matched world', fontsize=8.5)
ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_ylim(0.4, 1.02)
# right: within-stratum (FU) vs dissociated AUROC across variants
labs, dis_h, fu_h, fu_o, fu_mi, fu_td = [], [], [], [], [], []
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    pr = probe_rows(R, 'hidden', 0.0); po = probe_rows(R, 'output_only', 0.0)
    labs.append(v); dis_h.append(np.mean([mean([p['within']['dissociated'] for p in prs]) for prs in pr]))
    fu_h.append(np.mean([mean([p['within']['FU'] for p in prs]) for prs in pr]))
    fu_o.append(np.mean([mean([p['within']['FU'] for p in prs]) for prs in po]))
    fu_mi.append(mean([r['baselines']['ens_mi']['FU'] for r in R])); fu_td.append(mean([r['baselines']['type_default_disagree']['FU'] for r in R]))
x = np.arange(len(labs)); wd = 0.16
for i, (vals, lab, col) in enumerate([(dis_h, 'hidden probe: dissociated pool', '#c0392b'), (fu_h, 'hidden probe: within FU', '#e6a0a0'),
                                       (fu_o, 'output probe: within FU', '#2c7fb8'), (fu_mi, 'ensemble MI: within FU', '#7f8c3a'),
                                       (fu_td, 'type-default disagreement: within FU', '#555555')]):
    ax[1].bar(x + (i - 2) * wd, vals, wd, color=col, label=lab)
ax[1].axhline(0.5, c='k', lw=.6, ls=':'); ax[1].set_xticks(x); ax[1].set_xticklabels(labs, rotation=30, fontsize=7)
ax[1].set_ylabel('AUROC (detect wrong answers)'); ax[1].legend(fontsize=6, frameon=False, loc='upper left', ncol=2); ax[1].set_ylim(0, 1.15)
ax[1].set_title('Pooled dissociated AUROC vs within-stratum AUROC', fontsize=8.5)
for a_ in ax: a_.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{D}/fig_r1_prop1_strata.png', dpi=160); plt.close()

# Fig R1-b: Prop. 2 decomposition
fig, ax = plt.subplots(figsize=(5.4, 3.5))
xs = []; k = 0
for v in ORDER:
    R = [d['mlp'] for d in runs[v]]
    pr = probe_rows(R, 'hidden', 0.0)
    if not pr[0] or 'pred_fk_prior' not in pr[0][0]: continue
    g = lambda kk: np.mean([mean([p[kk] for p in prs]) for prs in pr])
    for j, (kk, col, lab) in enumerate([('false_know', '#c0392b', 'observed (logistic probe)'), ('lda_emp_fk', '#e67e22', 'empirical LDA on hidden states'),
                                         ('pred_fk_balanced', '#2c7fb8', 'Prop. 2, two-vector geometry'), ('pred_fk_prior', '#16a085', 'Prop. 2, class-prior corrected')]):
        ax.bar(k + (j - 1.5) * 0.2, g(kk), 0.2, color=col, label=lab if k == 0 else None)
    xs.append(v); k += 1
ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=30, fontsize=7); ax.axhline(.5, c='k', lw=.6, ls=':')
ax.set_ylabel('false-"knows" on familiar & wrong'); ax.legend(fontsize=6.5, frameon=False); ax.spines[['top', 'right']].set_visible(False)
ax.set_title(r'Natural-data probe ($\pi_d$=0): where the Prop. 2 gap comes from', fontsize=8.5)
plt.tight_layout(); plt.savefig(f'{D}/fig_r1_prop2_gap.png', dpi=160); plt.close()
