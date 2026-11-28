"""Exp 3: division of labour (routing) vs conflict resolution (aggregation), on held-out items.
All selection / fitting uses a random 50% train split (stratified by category); metrics on the test half.
Repeated over 20 splits."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, plurality, weighted_plurality

import sys
POOL = sys.argv[1] if len(sys.argv) > 1 else "all46"
d = load()
A, y, nopt, cat, models = d["A"], d["y"], d["nopt"], d["cat"], d["models"]
_acc = (A == y[:, None]).mean(0)
if POOL == "excl_top1":          # remove the single dominant agent
    keep = np.argsort(-_acc)[1:]
elif POOL == "peer_0.60-0.80":   # near-peer frontier-ish agents only
    keep = np.nonzero((_acc >= 0.60) & (_acc <= 0.80))[0]
else:
    keep = np.arange(A.shape[1])
A = A[:, keep]; models = [models[i] for i in keep]
n, m = A.shape
K = 10
C = A == y[:, None]
cats = np.unique(cat)


def onehot_votes(Asub):
    V = np.zeros((Asub.shape[0], Asub.shape[1], K), dtype=np.float32)
    ii, jj = np.nonzero(Asub >= 0)
    V[ii, jj, Asub[ii, jj]] = 1
    return V


def fit_trust(Atr, ytr, steps=400, lr=0.5, l2=1e-3):
    """Supervised 'learned trust' aggregator: score(option) = sum_j w_j * 1[a_j = option] + b * 0;
    softmax over options, fitted by full-batch gradient descent on train items."""
    V = onehot_votes(Atr)                      # (n, m, K)
    w = np.ones(V.shape[1]) * 0.5
    Y = np.zeros((len(ytr), K)); Y[np.arange(len(ytr)), ytr] = 1
    for _ in range(steps):
        s = np.einsum("nmk,m->nk", V, w)
        s -= s.max(1, keepdims=True)
        P = np.exp(s); P /= P.sum(1, keepdims=True)
        g = np.einsum("nmk,nk->m", V, P - Y) / len(ytr) + l2 * w
        w -= lr * g
    return w


def dawid_skene_onecoin(Asub, nopt_sub, iters=50):
    """Unsupervised one-coin Dawid-Skene: agent j correct w.p. p_j, otherwise uniform over the K_i-1
    wrong options; abstentions ignored. Returns posterior argmax."""
    V = onehot_votes(Asub)
    nonab = (Asub >= 0)
    p = np.full(Asub.shape[1], 0.6)
    for _ in range(iters):
        # E-step: log posterior over options
        lw = np.log(p / (1 - p)) + np.log(nopt_sub[:, None] - 1)   # (n, m) weight of agreeing
        s = np.einsum("nmk,nm->nk", V, lw)
        mask = np.arange(K)[None, :] < nopt_sub[:, None]
        s = np.where(mask, s, -np.inf)
        s -= s.max(1, keepdims=True)
        T = np.exp(s); T /= T.sum(1, keepdims=True)
        # M-step
        agree = np.einsum("nmk,nk->m", V, T)
        p = np.clip(agree / np.maximum(nonab.sum(0), 1), 1e-3, 1 - 1e-3)
    return T.argmax(1), p


rng = np.random.default_rng(2)
SPLITS = 20
records = {}
def add(k, v):
    records.setdefault(k, []).append(float(v))

for s in range(SPLITS):
    tr = np.zeros(n, bool)
    for c in cats:
        idx = np.nonzero(cat == c)[0]
        tr[rng.choice(idx, len(idx) // 2, replace=False)] = True
    te = ~tr
    acc_tr = C[tr].mean(0)
    order = np.argsort(-acc_tr)
    best = order[0]
    add("best_single", C[te, best].mean())
    add("mean_single", C[te].mean())
    # Division of labour: category router (best agent per category on train)
    pred = np.empty(te.sum(), dtype=int)
    ct = cat[te]
    At = A[te]
    for c in cats:
        b = np.argmax(C[tr & (cat == c)].mean(0))
        pred[ct == c] = At[ct == c, b]
    add("category_router", (pred == y[te]).mean())
    # Conflict resolution by voting
    for k in [3, 5, 9, m]:
        team = order[:k]
        add(("vote_all" if k == m else f"vote_top{k}"), (plurality(At[:, team], rng) == y[te]).mean())
        pk = np.clip(acc_tr[team], 1e-3, 1 - 1e-3)
        w = np.log((K - 1) * pk / (1 - pk))              # Nitzan-Paroush weights (optimal under independence)
        add(("npweighted_all" if k == m else f"npweighted_top{k}"), (weighted_plurality(At[:, team], w, rng) == y[te]).mean())
        add(("oracle_all" if k == m else f"oracle_top{k}"), C[te][:, team].any(1).mean())
    # Learned trust over all 46
    wt = fit_trust(A[tr], y[tr])
    add("learned_trust_all", (weighted_plurality(At, wt, rng) == y[te]).mean())
    # Unsupervised Dawid-Skene over all 46 (fit on test items themselves: label-free)
    ds, _ = dawid_skene_onecoin(At, nopt[te])
    add("dawid_skene_all_unsup", (ds == y[te]).mean())
    # Greedy vote-optimised team (forward selection on train vote accuracy, up to 9 members)
    team = [best]
    trA = A[tr]
    for _ in range(8):
        cand = [j for j in range(m) if j not in team]
        sc = [(plurality(trA[:, team + [j]], np.random.default_rng(0)) == y[tr]).mean() for j in cand]
        team.append(cand[int(np.argmax(sc))])
    add("greedy_vote_team9", (plurality(At[:, team], rng) == y[te]).mean())
    records.setdefault("greedy_team_members", []).append([models[j] for j in team])
    print(s, {k: round(v[-1], 4) for k, v in records.items() if k != "greedy_team_members"})

summary = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)),
                  paired_diff_vs_best_single=float(np.mean(np.array(v) - np.array(records["best_single"]))),
                  paired_diff_sd=float(np.std(np.array(v) - np.array(records["best_single"]))),
                  splits_better_than_best_single=int(np.sum(np.array(v) > np.array(records["best_single"]))))
           for k, v in records.items() if k != "greedy_team_members"}
summary["pool"] = POOL; summary["n_agents"] = m
summary["greedy_team_members_split0"] = records["greedy_team_members"][0]
# per-category best-agent identity (on full data, descriptive)
summary["best_agent_per_category_fulldata"] = {c: models[int(np.argmax(C[cat == c].mean(0)))] for c in cats}
summary["second_best_gap_per_category_fulldata"] = {
    c: float(np.sort(C[cat == c].mean(0))[-1] - np.sort(C[cat == c].mean(0))[-2]) for c in cats}
json.dump(summary, open(f"../results/exp3_division_{POOL}.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

keys = ["mean_single", "vote_all", "dawid_skene_all_unsup", "vote_top9", "vote_top5", "npweighted_all",
        "vote_top3", "greedy_vote_team9", "category_router", "best_single", "npweighted_top5", "learned_trust_all",
        "oracle_top3", "oracle_all"]
keys = sorted(keys, key=lambda k: summary[k]["mean"])
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(range(len(keys)), [summary[k]["mean"] for k in keys], xerr=[summary[k]["sd"] for k in keys],
        color=["C3" if k.startswith("oracle") else ("C2" if "router" in k or "single" in k else "C0") for k in keys])
ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=8)
ax.axvline(summary["best_single"]["mean"], c="k", ls=":", lw=1)
ax.set_xlabel("held-out accuracy (mean ± sd over 20 splits)"); ax.set_xlim(0.4, 1.0)
plt.tight_layout(); plt.savefig(f"../results/fig4_division_{POOL}.png", dpi=140)
