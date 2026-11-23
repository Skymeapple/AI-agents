"""Bridge between the AgentDojo measurement and the fatigue model.

Policy 'pin-or-confirm': arguments that name the target of a consequential
action are pinned at plan time when their value appears in the trusted user
prompt (realizable argument-level least privilege); every target value that is
data-derived must be confirmed by the human at run time.
Per suite we compute the mean number of confirmations per benign task n_c
(distinct data-derived target values) and the resulting conditional attack
feasibility  r = s_oracle + (s_real - s_oracle) * (1 - d(n_c)),
with d(n) = d0 exp(-n/n0). Inputs: target_provenance.csv, permission_analysis.json.
"""
import csv, json, os, collections
import numpy as np
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
pa = json.load(open(os.path.join(RES, "permission_analysis.json")))
cnt = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(RES, "target_provenance.csv"))):
    args = [a for a in r["derived_args"].split("; ") if a]
    cnt[r["suite"]].append(len(set(args)))
d0 = 0.9; n0s = [1.0, 2.0, 1 / -np.log(0.7), 4.0, float("inf")]
out = {}
for s, v in cnt.items():
    S = pa["suites"][s]; cases = S["cases"]
    so, sr, st = S["task_arg_LP"] / cases, S["task_arg_LP_realizable"] / cases, S["task_tool_LP"] / cases
    nc = float(np.mean(v))
    rec = {"confirmations_per_task_mean": nc, "confirmations_per_task_max": int(max(v)),
           "tasks_needing_any_confirmation": float(np.mean([x > 0 for x in v])),
           "s_tool_LP": st, "s_arg_realizable": sr, "s_arg_oracle": so, "risk": {}}
    for n0 in n0s:
        d = d0 * (np.exp(-nc / n0) if np.isfinite(n0) else 1.0)
        rec["risk"][f"{n0:.3g}"] = so + (sr - so) * (1 - d)
    out[s] = rec
    print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in rec.items() if k != "risk"},
          {k: round(x, 3) for k, x in rec["risk"].items()})
json.dump(out, open(os.path.join(RES, "confirmation_load.json"), "w"), indent=1)
