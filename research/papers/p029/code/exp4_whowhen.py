"""Experiment 4: incentive budgets of real failure-attribution channels (Who&When data).

Data: Who&When (Zhang et al., ICML 2025), cloned by fetch_data.sh into ../ext/ww.
We do NOT run any LLM.  We (i) compute confusion matrices of log-only heuristic
attributors on the 58 hand-crafted (Magentic-One) failure logs, whose agent roles are fixed,
(ii) compute accuracy of the same heuristics on the 126 algorithm-generated logs, and
(iii) translate the agent-level accuracies PUBLISHED in Zhang et al. Table 1 into incentive
terms under a symmetric-noise assumption (labelled as such).
Output: results/exp4_whowhen.json
"""
import json, glob, os, collections
import numpy as np
import model as m

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "ext", "ww", "Who&When")
OUT = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(11)
ROLES = ["Orchestrator", "WebSurfer", "FileSurfer", "Assistant"]
WORKERS = ["WebSurfer", "FileSurfer", "Assistant"]


def base_role(r):
    r = (r or "").strip("'\" ")
    for k in ROLES + ["ComputerTerminal", "human"]:
        if r.startswith(k):
            return k
    return r


def heuristics_hand(hist):
    seq = [base_role(h.get("role")) for h in hist]
    workers = [s for s in seq if s in WORKERS]
    cnt = collections.Counter(workers)
    out = {"always_orchestrator": "Orchestrator", "always_websurfer": "WebSurfer"}
    out["last_worker"] = workers[-1] if workers else "Orchestrator"
    out["first_worker"] = workers[0] if workers else "Orchestrator"
    out["most_active_worker"] = cnt.most_common(1)[0][0] if workers else "Orchestrator"
    # orchestrator blamed unless a single worker did >= 80% of worker turns late in the log
    tail = [s for s in seq[len(seq) // 2:] if s in WORKERS]
    out["last_half_majority_worker"] = collections.Counter(tail).most_common(1)[0][0] if tail else "Orchestrator"
    return out


def confusion(true, pred):
    n = len(ROLES)
    N = np.zeros((n, n))
    for t, p in zip(true, pred):
        N[ROLES.index(p), ROLES.index(t)] += 1
    return N


def channel_stats(N):
    """N[a, j] counts. C = column-normalised (columns with zero count are dropped)."""
    keep = N.sum(axis=0) > 0
    C = N[:, keep] / N[:, keep].sum(axis=0, keepdims=True)
    nk = int(keep.sum())
    return {"accuracy": float(np.trace(N) / N.sum()),
            "balanced_accuracy": float(np.mean([N[j, j] / N[:, j].sum() for j in range(len(ROLES)) if N[:, j].sum() > 0])),
            "budget_B": float(np.max(C, axis=1).sum()),
            "n_parties": nk,
            "budget_per_party": float(np.max(C, axis=1).sum() / nk),
            "theta_attributed_pays_all": [float(C[j, jj]) for jj, j in enumerate(np.where(keep)[0])]}


res = {}
# ------------------------------------------------ (i) hand-crafted logs
true, preds = [], collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(DATA, "Hand-Crafted", "*.json"))):
    d = json.load(open(f))
    true.append(base_role(d["mistake_agent"]))
    for k, v in heuristics_hand(d["history"]).items():
        preds[k].append(v)
res["hand_crafted_n"] = len(true)
res["hand_crafted_true_counts"] = dict(collections.Counter(true))
hc = {}
true = np.array(true)
for k, p in preds.items():
    p = np.array(p)
    N = confusion(true, p)
    st = channel_stats(N)
    st["confusion_counts_rows_pred_cols_true"] = N.astype(int).tolist()
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(true), len(true))
        s = channel_stats(confusion(true[idx], p[idx]))
        boots.append((s["accuracy"], s["budget_per_party"]))
    boots = np.array(boots)
    st["accuracy_ci95"] = np.percentile(boots[:, 0], [2.5, 97.5]).tolist()
    st["budget_per_party_ci95"] = np.percentile(boots[:, 1], [2.5, 97.5]).tolist()
    hc[k] = st
res["hand_crafted_heuristics"] = hc
res["roles"] = ROLES

# ------------------------------------------------ (ii) algorithm-generated logs (accuracy only)
acc = collections.defaultdict(list); nag = []
for f in sorted(glob.glob(os.path.join(DATA, "Algorithm-Generated", "*.json"))):
    d = json.load(open(f))
    names = [h.get("name") for h in d["history"] if h.get("name")]
    agents = [x for x in names if x != "Computer_terminal"] or names
    cnt = collections.Counter(agents)
    nag.append(len(set(names)))
    t = d["mistake_agent"]
    acc["first_speaker"].append(agents[0] == t)
    acc["last_speaker"].append(agents[-1] == t)
    acc["most_active"].append(cnt.most_common(1)[0][0] == t)
    acc["uniform_random_expected"].append(1.0 / len(set(names)))
res["algorithm_generated"] = {"n": len(nag), "mean_n_agents": float(np.mean(nag)),
                              "heuristic_accuracy": {k: float(np.mean(v)) for k, v in acc.items()}}

# ------------------------------------------------ (iii) published accuracies -> incentive terms
published = {  # Zhang et al. (2025) Table 1, agent-level accuracy (%), GPT-4o, WITHOUT ground truth
    "random/hand-crafted": 12.00, "all-at-once/hand-crafted": 53.44, "step-by-step/hand-crafted": 32.75,
    "binary-search/hand-crafted": 36.21, "random/algorithm-generated": 29.10,
    "all-at-once/algorithm-generated": 51.12, "step-by-step/algorithm-generated": 26.02,
    "binary-search/algorithm-generated": 30.11}
rows = {}
for k, a in published.items():
    q = a / 100.0
    row = {"q": q, "internalised_share_attributed_pays_all": q,
           "dwl_factor_per_tau": float(1 / q - 1 + np.log(q))}
    for nn in [2, 3, 4]:
        if q > 1.0 / nn + 1e-9:
            r = (1 - q) / (nn - 1)
            row[f"debiased_payment_over_L_n{nn}"] = float((1 - r) / (q - r))
            row[f"debiased_rebate_over_L_n{nn}"] = float(r / (q - r))
        else:
            row[f"debiased_payment_over_L_n{nn}"] = None
    rows[k] = row
res["published_accuracy_translation_symmetric_noise_ASSUMPTION"] = rows

with open(os.path.join(OUT, "exp4_whowhen.json"), "w") as f:
    json.dump(res, f, indent=1)
print(res["hand_crafted_true_counts"])
for k, v in hc.items():
    print(k, round(v["accuracy"], 3), [round(x, 3) for x in v["accuracy_ci95"]], "bal", round(v["balanced_accuracy"], 3),
          "B", round(v["budget_B"], 3), "B/n", round(v["budget_per_party"], 3), [round(x, 3) for x in v["budget_per_party_ci95"]],
          "theta", [round(x, 2) for x in v["theta_attributed_pays_all"]])
    print("   ", v["confusion_counts_rows_pred_cols_true"])
print(res["algorithm_generated"])
for k, v in rows.items():
    print(k, {a: (round(b_, 3) if b_ is not None else None) for a, b_ in v.items()})
