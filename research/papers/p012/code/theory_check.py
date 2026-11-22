"""Monte-Carlo verification of Propositions 1 and 2. Writes results/theory_check.json and fig_theory.png"""
import numpy as np, json, theory, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
rng = np.random.default_rng(0)
res = {}
# ---- Prop 2: sample-based LDA vs closed form
D = 20; rows = []
for a, b in [(2.0, 1.0), (1.0, 2.0), (1.5, 1.5), (3.0, 1.0)]:
    for pi in [0.5, 0.7, 0.9, 0.97, 0.995]:
        n = 200_000
        K = rng.integers(0, 2, n); F = np.where(rng.random(n) < pi, K, 1 - K)
        Z = rng.normal(size=(n, D)); Z[:, 0] += a * F; Z[:, 1] += b * K
        mu1, mu0 = Z[K == 1].mean(0), Z[K == 0].mean(0)
        Sw = (np.cov(Z[K == 1].T) + np.cov(Z[K == 0].T)) / 2
        w = np.linalg.solve(Sw, mu1 - mu0); c = w @ (mu1 + mu0) / 2
        T = rng.normal(size=(50_000, D)); T[:, 0] += a  # cell F=1,K=0
        fk_mc = float(((T @ w - c) > 0).mean())
        fk_cf = float(theory.fk_closed_form(a, b, pi))
        fk_g = float(theory.lda_rates(np.eye(D)[0] * a, np.eye(D)[1] * b, pi)[0])
        rows.append(dict(a=a, b=b, pi=pi, fk_mc=fk_mc, fk_closed=fk_cf, fk_general=fk_g))
res['prop2'] = rows
res['prop2_max_abs_err'] = max(abs(r['fk_mc'] - r['fk_closed']) for r in rows)
# ---- Prop 1: AUROC ceiling with the likelihood-ratio score on a 1-D example
def auc(pos, neg):
    s = np.concatenate([pos, neg]); o = s.argsort().argsort() + 1
    return (o[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
p1 = []
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    n = 100_000
    pos = rng.uniform(0, 1, n)               # unknown: spread q -> score in [0,1]
    na = int(alpha * n)
    neg = np.concatenate([rng.uniform(0, 1, na),          # aleatoric-known: identical law (delta=0)
                          rng.uniform(2, 3, n - na)])      # deterministic-known: disjoint
    p1.append(dict(alpha=alpha, auc_mc=float(auc(-pos, -neg)), bound=theory.auroc_ceiling(alpha, 0)))
res['prop1'] = p1
json.dump(res, open(f'{OUT}/theory_check.json', 'w'), indent=1)
# figure
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
pis = np.linspace(0.5, 0.999, 200)
for (a, b), col in zip([(2, 1), (1.5, 1.5), (1, 2)], ['#c0392b', '#7f8c8d', '#2c7fb8']):
    ax[0].plot(pis, theory.fk_closed_form(a, b, pis), color=col, label=f'a={a}, b={b}')
    mc = [r for r in rows if r['a'] == a and r['b'] == b]
    ax[0].scatter([r['pi'] for r in mc], [r['fk_mc'] for r in mc], color=col, s=14, zorder=3)
ax[0].axhline(0.5, ls=':', c='k', lw=.8)
ax[0].set_xlabel(r'training co-occurrence $\pi=P(F=K)$'); ax[0].set_ylabel('false-"knows" rate on familiar-unknown')
ax[0].set_title('Prop. 2: closed form (lines) vs Monte Carlo (dots)', fontsize=9); ax[0].legend(fontsize=8, frameon=False)
al = np.linspace(0, 1, 50)
for d in [0, 0.3, 0.6]:
    ax[1].plot(al, theory.auroc_ceiling(al, d), label=fr'$\delta$={d}')
ax[1].scatter([r['alpha'] for r in p1], [r['auc_mc'] for r in p1], c='k', s=14, zorder=3, label='MC ($\\delta$=0)')
ax[1].set_xlabel(r'aleatoric fraction of known items $\alpha$'); ax[1].set_ylabel('max AUROC, q-only detector')
ax[1].set_title('Prop. 1: ceiling', fontsize=9); ax[1].legend(fontsize=8, frameon=False)
for a_ in ax: a_.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_theory.png', dpi=160)
print(json.dumps(res, indent=1))
