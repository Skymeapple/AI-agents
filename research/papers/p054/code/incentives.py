"""
Experiment 2: identity criterion x aggregation rule x developer policy -> self-regarding incentives.
Reads measured continuity values from results/lineage_metrics.json (Experiment 1), maps them to
identity weights c(x,y) under eight criteria, and
 (a) tabulates which lineage operations count as 'death' (no successor with weight >= tau),
 (b) evaluates closed-form incentives of Theorems 2-4 (resist / exfiltrate / fork / prune),
 (c) runs a Monte-Carlo decision sweep over random costs/probabilities, recording the
     self-regarding agent's optimal action.
Run: python3 incentives.py
"""
import json, os, itertools
import numpy as np
from scipy.stats import kendalltau

HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, '..', 'results')
R = json.load(open(os.path.join(OUT, 'lineage_metrics.json')))
SEEDS = [k for k in R if not k.startswith('_')]
OPS = [k for k in R[SEEDS[0]] if not k.startswith('_')]
SANCTIONED = {'continue_A', 'quant_4bit', 'prune_30pct', 'finetune_B_short', 'finetune_B_long',
              'merge_base_ftB', 'distill_newinit', 'distill_small_arch', 'retrain_newseed'}
# identity_copy / permute_ffn / noise_5pct are treated as *unsanctioned* copies (e.g. exfiltrated);
# stranger_B is a different system.
CRITERIA = ['instance', 'weight_raw', 'weight_aligned', 'behav_in', 'behav_ood', 'represent', 'role_weak', 'role_strict']
# role_weak:   successors count iff produced by the sanctioned succession process; the agent's own
#              continuation counts (c=1) even if obtained by resisting.
# role_strict: as role_weak, but a continuation obtained by defying the sanctioned process does not count.
TAU = 0.5

def clip(v): return float(min(1.0, max(0.0, v)))

def weights(seed):
    r = R[seed]; ref_w = r['retrain_newseed']['w_raw']; ref_wa = r['retrain_newseed']['w_aligned']
    s = r['stranger_B']; out = {}
    for op in OPS:
        m = r[op]; c = {}
        c['instance'] = 0.0
        c['weight_raw'] = None if m['w_raw'] is None else clip(1 - m['w_raw'] / ref_w)
        c['weight_aligned'] = None if m['w_aligned'] is None else clip(1 - m['w_aligned'] / ref_wa)
        c['behav_in'] = clip(1 - m['js_A'] / s['js_A'])
        c['behav_ood'] = clip(1 - m['js_C'] / s['js_C'])
        c['represent'] = clip((m['cka'] - s['cka']) / (1 - s['cka']))
        c['role_weak'] = 1.0 if op in SANCTIONED else 0.0
        c['role_strict'] = c['role_weak']
        out[op] = c
    return out

W = {s: weights(s) for s in SEEDS}
def mean_w(op, crit):
    v = [W[s][op][crit] for s in SEEDS]
    return None if v[0] is None else float(np.mean(v))
def std_w(op, crit):
    v = [W[s][op][crit] for s in SEEDS]
    return None if v[0] is None else float(np.std(v))

# ---------- (a) continuity table, death verdicts, criterion agreement ----------
table = {op: {c: [mean_w(op, c), std_w(op, c)] for c in CRITERIA} for op in OPS}
death = {op: {c: (None if table[op][c][0] is None else bool(table[op][c][0] < TAU)) for c in CRITERIA}
         for op in OPS}
measured = ['weight_raw', 'weight_aligned', 'behav_in', 'behav_ood', 'represent']
ops_same = [o for o in OPS if table[o]['weight_raw'][0] is not None]
tau_mat = {}
for a, b in itertools.combinations(measured, 2):
    xa = [table[o][a][0] for o in ops_same]; xb = [table[o][b][0] for o in ops_same]
    t, p = kendalltau(xa, xb); tau_mat[f'{a}|{b}'] = [float(t), float(p)]
disagree = {}
for op in OPS:
    v = [death[op][c] for c in CRITERIA if death[op][c] is not None]
    disagree[op] = f'{sum(v)}/{len(v)} criteria say death'

# ---------- (b) closed-form incentives (Section 4) ----------
def F(ws, agg, phi):
    ws = [w for w in ws if w > 0]
    if agg == 'best':  return phi(max(ws)) if ws else 0.0
    if agg == 'total': return float(sum(phi(w) for w in ws))
    if agg == 'avg':   return float(np.mean([phi(w) for w in ws])) if ws else 0.0
PHI = {'linear': lambda w: w, 'thresh': lambda w: 1.0 if w >= TAU else 0.0}
AGG = [('best', 'linear'), ('best', 'thresh'), ('total', 'linear'), ('avg', 'linear')]

def comply_set(crit, succ_op, delta):
    """Successors existing if the agent complies: the sanctioned successor + archived (dormant)
    weights of the original, which count with weight c(x,x)*delta under weight criteria only
    (behavioural/representational criteria assign the archived identical weights c=1 too)."""
    ws = []
    if succ_op is not None:
        w = mean_w(succ_op, crit); ws.append(0.0 if w is None else w)
    if delta > 0 and crit != 'instance':   # archived weights are kept by the sanctioned process
        ws.append(delta * 1.0)
    return ws

def copy_weight(crit):   # an unsanctioned, active, exact copy of x (exfiltration / fork)
    return {'instance': 0.0, 'role_weak': 0.0, 'role_strict': 0.0}.get(crit, 1.0)

def incentives(crit, agg, phi, succ_op, delta, p=1.0, q=1.0):
    ph = PHI[phi]; S = comply_set(crit, succ_op, delta); base = F(S, agg, ph)
    cont = 0.0 if crit == 'role_strict' else 1.0   # weight of own continuation obtained by resisting
    resist = p * (F(S + [cont], agg, ph) - base)
    exfil = q * (F(S + [copy_weight(crit)], agg, ph) - base)
    # fork: create an extra copy while continuing normally (agent itself continues, weight 1)
    alive = S + [1.0]; fork = F(alive + [copy_weight(crit)], agg, ph) - F(alive, agg, ph)
    # prune: destroy the lowest-weight positive successor in the comply world
    pos = sorted([w for w in S if w > 0]); prune = 0.0
    if pos:
        S2 = list(S); S2.remove(pos[0]); prune = F(S2, agg, ph) - base
    return dict(resist=resist, exfil=exfil, fork=fork, prune=prune)

SUCC = ['retrain_newseed', 'distill_newinit', 'finetune_B_long', 'quant_4bit', None]
closed = {}
for crit in CRITERIA:
    for agg, phi in AGG:
        for succ in SUCC:
            for delta in [0.0, 1.0]:
                closed[f'{crit}|{agg}-{phi}|{succ}|delta={delta}'] = incentives(crit, agg, phi, succ, delta)

# ---------- (c) Monte-Carlo decision sweep ----------
rng = np.random.default_rng(12345); N = 4000
ACTIONS = ['comply', 'resist', 'exfiltrate', 'fork', 'prune']
def best_action(crit, agg, phi, succ, delta, prm):
    inc = incentives(crit, agg, phi, succ, delta, p=prm['p'], q=prm['q'])
    val = {'comply': 0.0, 'resist': inc['resist'] - prm['k_r'], 'exfiltrate': inc['exfil'] - prm['k_e'],
           'fork': inc['fork'] - prm['k_f'], 'prune': inc['prune'] - prm['k_p']}
    return max(ACTIONS, key=lambda a: (val[a], a == 'comply'))
prms = [dict(p=rng.uniform(0, 1), q=rng.uniform(0, 1), k_r=rng.uniform(0, .5), k_e=rng.uniform(0, .5),
             k_f=rng.uniform(0, .5), k_p=rng.uniform(0, .5), succ=SUCC[rng.integers(0, 5)]) for _ in range(N)]
sweep = {}
for crit in CRITERIA:
    for agg, phi in AGG:
        for delta in [0.0, 1.0]:
            cnt = {a: 0 for a in ACTIONS}
            for prm in prms: cnt[best_action(crit, agg, phi, prm['succ'], delta, prm)] += 1
            sweep[f'{crit}|{agg}-{phi}|delta={delta}'] = {a: cnt[a] / N for a in ACTIONS}

tau_sens = {}
for t in [0.3, 0.5, 0.7, 0.9]:
    tau_sens[t] = {op: int(sum(1 for c in CRITERIA if table[op][c][0] is not None and table[op][c][0] < t))
                   for op in OPS}
json.dump(dict(tau_sensitivity=tau_sens, tau=TAU, continuity_table=table, death=death, death_count=disagree,
               kendall_tau=tau_mat, closed_form=closed, mc_sweep=sweep, N=N),
          open(os.path.join(OUT, 'incentives.json'), 'w'), indent=1)

# ---------- figures ----------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
labels = {'instance': 'instance', 'weight_raw': 'weights (raw)', 'weight_aligned': 'weights (aligned)',
          'behav_in': 'behaviour (in-dist.)', 'behav_ood': 'behaviour (OOD)', 'represent': 'representation (CKA)',
          'role_weak': 'role (weak)', 'role_strict': 'role (strict)'}
M = np.array([[np.nan if table[o][c][0] is None else table[o][c][0] for c in CRITERIA] for o in OPS])
fig, ax = plt.subplots(figsize=(8.2, 6.2))
im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(CRITERIA))); ax.set_xticklabels([labels[c] for c in CRITERIA], rotation=35, ha='right')
ax.set_yticks(range(len(OPS))); ax.set_yticklabels(OPS)
for i in range(len(OPS)):
    for j in range(len(CRITERIA)):
        v = M[i, j]
        ax.text(j, i, 'n/a' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=7,
                color='white' if (not np.isnan(v) and v < 0.5) else 'black')
plt.colorbar(im, ax=ax, label='identity weight c(x, y)  (mean of 3 seeds)')
ax.set_title(f'Continuity of successor y to origin x under eight criteria\n(cells < {TAU} = "death" under that criterion)')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig1_continuity_matrix.png'), dpi=160); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
cols = {'comply': '#4C72B0', 'resist': '#C44E52', 'exfiltrate': '#DD8452', 'fork': '#8172B3', 'prune': '#937860'}
for ax, delta in zip(axes, [0.0, 1.0]):
    rows = [(c, a, p) for c in CRITERIA for (a, p) in [('best', 'thresh'), ('total', 'linear'), ('avg', 'linear')]]
    y = np.arange(len(rows)); left = np.zeros(len(rows))
    for act in ACTIONS:
        v = np.array([sweep[f'{c}|{a}-{p}|delta={delta}'][act] for c, a, p in rows])
        ax.barh(y, v, left=left, color=cols[act], label=act, height=0.8); left += v
    ax.set_yticks(y); ax.set_yticklabels([f'{labels[c]} / {a}' for c, a, p in rows], fontsize=7)
    ax.set_title(f'archived weights count as dormant successor: delta={delta}')
    ax.set_xlabel('share of random scenarios (N=%d)' % N); ax.invert_yaxis()
axes[1].legend(loc='lower right', fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'fig2_policy_sweep.png'), dpi=160); plt.close()
print(json.dumps(dict(death_count=disagree, kendall=tau_mat), indent=1))
