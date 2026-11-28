"""Summarize E2 results (train_toy.json) into a table + figure."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = {}
for fn in ["train_toy.json", "tt_delta0.json", "tt_delta1.json", "tt_tf.json", "tt_tf4b.json"]:
    if os.path.exists(os.path.join(OUT, fn)): R.update(json.load(open(os.path.join(OUT, fn))))
models = ["tf1", "tf2", "tf4", "delta"]; names = {"tf1": "Transformer L=1", "tf2": "Transformer L=2", "tf4": "Transformer L=4", "delta": "Delta-rule (1 layer)"}
EVAL_N = [32, 128, 512]; EVAL_R = [1, 2, 4, 8, 16, 32, 64]
summ = {}
lines = []
for m in models:
    runs = [v for k, v in R.items() if v["model"] == m]
    if not runs: continue
    summ[m] = {"n_seeds": len(runs), "indist": [float(np.mean([r["indist_acc"] for r in runs])), float(np.std([r["indist_acc"] for r in runs]))]}
    for key in runs[0]["grid"]:
        a = np.array([r["grid"][key] for r in runs if key in r["grid"]]); summ[m][key] = [float(a.mean()), float(a.std())]
    if runs[0]["logit_range_per_head"]:
        lr = np.array([r["logit_range_per_head"] for r in runs]); summ[m]["logit_range_maxhead"] = [float(lr.max(1).mean()), float(lr.max(1).std())]; summ[m]["logit_range_all"] = lr.tolist()
    lines.append(f"{names[m]}: seeds={len(runs)} in-dist acc={summ[m]['indist'][0]:.3f}±{summ[m]['indist'][1]:.3f} final train loss=" + ", ".join(f"{r['final_train_loss']:.3f}" for r in runs))
json.dump(summ, open(os.path.join(OUT, "train_toy_summary.json"), "w"), indent=1)
open(os.path.join(OUT, "train_toy_table.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
# age table (R=1, cold fact at varying age)
al = ["| model | N | R=1 age 0 | R=1 age 0.25N | R=1 age 0.5N | R=1 age 0.95N |", "|---|---|---|---|---|---|"]
for m in summ:
    for N in EVAL_N:
        ks = [f"N{N}_R1_fixage0.0", f"N{N}_R1_fixage0.25", f"N{N}_R1_age0.5", f"N{N}_R1_age0.95"]
        al.append(f"| {names[m]} | {N} | " + " | ".join(f"{summ[m][k][0]:.2f}±{summ[m][k][1]:.2f}" if k in summ[m] else "–" for k in ks) + " |")
open(os.path.join(OUT, "train_toy_age_table.md"), "w").write("\n".join(al) + "\n"); print("\n".join(al))
# fixed-age table
fl = ["| model | N | latest age | R=1 | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 |", "|---|---|---|---|---|---|---|---|---|---|"]
for m in summ:
    for N in EVAL_N:
        for af in ["0.0", "0.25"]:
            c = [(f"{summ[m][k][0]:.2f}±{summ[m][k][1]:.2f}" if (k := f"N{N}_R{r}_fixage{af}") in summ[m] else "–") for r in EVAL_R]
            fl.append(f"| {names[m]} | {N} | {'0' if af=='0.0' else 'N/4'} | " + " | ".join(c) + " |")
open(os.path.join(OUT, "train_toy_fixage_table.md"), "w").write("\n".join(fl) + "\n"); print("\n".join(fl))
# Theorem-1 implied revision budget for 1-layer model using measured logit spread (2B := spread)
for m in summ:
    if "logit_range_maxhead" in summ[m]:
        sp = summ[m]["logit_range_maxhead"][0]
        for N in [32, 128, 512]:
            for eps in [0.25, 0.5 - 1e-9]:
                L = np.log((1 - eps) / eps) if eps < 0.49 else float("nan")
            print(m, "spread(max head)=%.2f" % sp, "N=%d" % N, "R_ub(eps=.25)=%.2f" % ((sp - np.log(N - 1)) / np.log(3)))
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
for row, af in enumerate(["0.0", "0.25"]):
    for i, N in enumerate(EVAL_N):
        a = ax[row, i]
        for m in summ:
            rs = [r for r in EVAL_R if f"N{N}_R{r}_fixage{af}" in summ[m]]
            mu = np.array([summ[m][f"N{N}_R{r}_fixage{af}"][0] for r in rs]); sd = np.array([summ[m][f"N{N}_R{r}_fixage{af}"][1] for r in rs])
            a.errorbar(rs, mu, yerr=sd, marker="o", capsize=3, label=names[m])
        a.axhline(1 / 16, ls=":", c="gray"); a.set_xscale("log", base=2)
        a.set_title(f"N={N}{' (train 16-128)' if N <= 128 else ' (4x beyond train)'}, latest age={'0' if af=='0.0' else 'N/4'}", fontsize=9)
        if row == 1: a.set_xlabel("versions R of queried key")
    ax[row, 0].set_ylabel("latest-value accuracy")
ax[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_trained.png"), dpi=150); print("saved")
