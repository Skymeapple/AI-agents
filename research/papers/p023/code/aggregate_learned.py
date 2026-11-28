"""Aggregate learned-agent results over seeds -> results/learned_summary.json + fig7."""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
files = sorted(glob.glob(os.path.join(OUT, "learned_results_seed*.json")))
Rs = [json.load(open(f)) for f in files]
conds = ["shared", "bootstrap", "independent"]; ns = Rs[0]["shared"]["ns"]
S = {"n_seeds": len(Rs), "ns": ns}
allo, allp = [], []
fig, ax = plt.subplots(figsize=(5.4, 3.9))
for ci, c in enumerate(conds):
    g = lambda key: np.array([R[c][key] for R in Rs])
    eo, ep = g("eps_star_obs"), g("eps_star_pred")
    allo += eo.ravel().tolist(); allp += ep.ravel().tolist()
    S[c] = {k: [float(g(k).mean()), float(g(k).std(ddof=1))] for k in
            ["p_mean_G", "p_mean_S", "rho_mean_G", "rho_mean_S", "MAE_beta", "MAE_iid", "generalist_cross_subtask_error_corr_mean"]}
    S[c]["eps_obs_mean"] = eo.mean(0).round(4).tolist(); S[c]["eps_obs_sd"] = eo.std(0, ddof=1).round(4).tolist()
    S[c]["eps_pred_mean"] = ep.mean(0).round(4).tolist()
    S[c]["eps_obs_drop_n1_to_n33"] = [float((eo[:, 0] - eo[:, -1]).mean()), float((eo[:, 0] - eo[:, -1]).std(ddof=1))]
    S[c]["teamG_obs_n33"] = float(g("team_G_obs")[:, -1].mean()); S[c]["teamD_obs_n33"] = float(g("team_D_obs_eps0")[:, -1].mean())
    S[c]["teamG_obs_n1"] = float(g("team_G_obs")[:, 0].mean()); S[c]["teamD_obs_n1"] = float(g("team_D_obs_eps0")[:, 0].mean())
    ax.errorbar(ns, eo.mean(0), yerr=eo.std(0, ddof=1), fmt="o-", color=f"C{ci}", capsize=2, label=f"{c}: observed")
    ax.plot(ns, ep.mean(0), "--", color=f"C{ci}", label=f"{c}: predicted from 8-agent pilot")
allo, allp = np.array(allo), np.array(allp)
S["eps_pred_vs_obs"] = {"pearson_r": float(np.corrcoef(allo, allp)[0, 1]), "MAE": float(np.abs(allo - allp).mean()),
                        "mean_signed_pred_minus_obs": float((allp - allo).mean()), "N": int(len(allo))}
ax.set_xlabel("agents per subtask n (budget = K n calls)"); ax.set_ylabel("break-even hand-off error eps*")
ax.set_title(f"Learned MLP agents, K=4, mean±sd over {len(Rs)} seeds", fontsize=9); ax.legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig7_learned_eps_star.png"), dpi=150); plt.close()
json.dump(S, open(os.path.join(OUT, "learned_summary.json"), "w"), indent=1)
print(json.dumps(S, indent=1))
