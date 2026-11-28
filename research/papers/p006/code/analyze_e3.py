"""Aggregate E3 runs (results/e3/*.json) and the Prop-4 check (results/e3_prop4.json):
tables -> results/e3_tables.md, summary -> results/e3_summary.json, figure -> results/fig3_single_head.png.
Values are means over seeds with [min, max] across seeds; each seed is evaluated on 500 streams per cell."""
import json, os, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
runs = {}
for f in glob.glob(os.path.join(OUT, "e3", "*.json")):
    runs.update(json.load(open(f)))
p4 = json.load(open(os.path.join(OUT, "e3_prop4.json"))) if os.path.exists(os.path.join(OUT, "e3_prop4.json")) else {}
cfgs = {}
for tag, r in runs.items():
    cfgs.setdefault((r["objective"], r["B"], r["p_single"]), []).append(r)
order = sorted(cfgs, key=lambda c: (c[0] != "value", c[1], c[2]))
lab = lambda c: f"{c[0]}, B={'∞' if np.isinf(c[1]) else int(c[1])}, p_single={c[2]}"


def agg(vals):
    v = np.array(vals); return float(v.mean()), float(v.min()), float(v.max())


summ = {}; md = []
md.append("### E3 chain test (N = 256, Δ = 16): measured worst-chain weight m vs Theorem-1 ceiling; accuracy on hot (H_R) and cold (H_1)\n")
md.append("| config | seeds | in-dist acc | A (R_A) | m | Thm-1 ceiling | bound violations | α hot | α cold | acc hot | acc cold |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for c in order:
    rs = cfgs[c]; s = summ.setdefault(lab(c), {"n_seeds": len(rs), "indist": agg([r["indist_acc"] for r in rs]), "chain": {}, "age0_spaced": {}})
    for N in [64, 128, 256]:
        for A in [0, 16, 32, 48, 64, 96, 128, 192]:
            cells = [ch for r in rs for ch in r["chain"] if ch["N"] == N and ch["A"] == A]
            if not cells: continue
            d = {k: agg([ch[k] for ch in cells]) for k in ["m_mean", "ceil_mean", "S_mean", "bound_violations", "alpha_hot", "alpha_cold", "acc_hot", "acc_cold", "acc_all", "frac_m_ge_095"]}
            d["R"] = cells[0]["R"]; s["chain"][f"N{N}_A{A}"] = d
            if N == 256 and A in [0, 16, 48, 96, 192]:
                f = lambda k: f"{d[k][0]:.2f} [{d[k][1]:.2f},{d[k][2]:.2f}]"
                md.append(f"| {lab(c)} | {len(rs)} | {s['indist'][0]:.3f} | {A} ({d['R']}) | {f('m_mean')} | {d['ceil_mean'][0]:.2f} | {d['bound_violations'][2]:.0f} | {d['alpha_hot'][0]:.2f} | {d['alpha_cold'][0]:.2f} | {d['acc_hot'][0]:.2f} | {d['acc_cold'][0]:.2f} |")
    for N in [64, 128, 256]:
        for R in [1, 2, 4, 8, 16, 32]:
            k = f"N{N}_R{R}"
            if k in rs[0]["age0_spaced"]:
                s["age0_spaced"][k] = {"alpha": agg([r["age0_spaced"][k]["alpha"] for r in rs]), "acc": agg([r["age0_spaced"][k]["acc"] for r in rs]),
                                       "acc_uniform": agg([r["age0"][k]["acc"] for r in rs])}
md.append("\n### E3 dilution check at R = 1 (A = 0): measured weight vs ceiling as N grows\n")
md.append("| config | N=64 m / ceiling | N=128 | N=256 |"); md.append("|---|---|---|---|")
for c in order:
    s = summ[lab(c)]["chain"]
    md.append(f"| {lab(c)} | " + " | ".join(f"{s[f'N{N}_A0']['m_mean'][0]:.3f} / {s[f'N{N}_A0']['ceil_mean'][0]:.3f}" for N in [64, 128, 256]) + " |")
md.append("\n### E3b age-0 cell, older versions evenly spaced (N = 256): weight on latest vs Proposition-4 plug-in ceiling (clamped models)\n")
md.append("| config | R=2 α / ceil | R=4 | R=8 | R=16 | R=32 | mean cold α | acc R=2 | acc R=32 (uniform placement as in E2) |")
md.append("|---|---|---|---|---|---|---|---|---|")
for c in order:
    if np.isinf(c[1]): continue
    tags = [t for t in p4 if runs.get(t, {}).get("objective") == c[0] and runs[t]["B"] == c[1] and runs[t]["p_single"] == c[2]]
    if not tags: continue
    cells = []
    for R in [2, 4, 8, 16, 32]:
        rows = [row for t in tags for row in p4[t] if row["N"] == 256 and row["R"] == R]
        cells.append(f"{np.mean([r['alpha_hot'] for r in rows]):.2f} / {np.mean([r['ceiling'] for r in rows]):.2f}")
    viol = sum(row["alpha_hot"] > row["ceiling"] + 1e-3 for t in tags for row in p4[t])
    summ[lab(c)]["prop4_violations"] = int(viol); summ[lab(c)]["prop4_tags"] = tags
    mc = np.mean([row["mean_cold_alpha"] for t in tags for row in p4[t] if row["N"] == 256])
    a0 = summ[lab(c)]["age0_spaced"]
    md.append(f"| {lab(c)} | " + " | ".join(cells) + f" | {mc:.2f} | {a0['N256_R2']['acc'][0]:.2f} | {a0['N256_R32']['acc_uniform'][0]:.2f} |")
md.append("\n(Prop-4 ceiling exceeded in any (model, N, R) cell: " + ", ".join(f"{k}: {v['prop4_violations']}" for k, v in summ.items() if "prop4_violations" in v) + ")")
json.dump(summ, open(os.path.join(OUT, "e3_summary.json"), "w"), indent=1)
open(os.path.join(OUT, "e3_tables.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))

# figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3)); cm = plt.get_cmap("tab10")
for i, c in enumerate(order):
    s = summ[lab(c)]["chain"]; As = [A for A in [0, 16, 32, 48, 64, 96, 128, 192] if f"N256_A{A}" in s]
    Rs = [s[f"N256_A{A}"]["R"] for A in As]
    ax[0].errorbar(Rs, [s[f"N256_A{A}"]["m_mean"][0] for A in As],
                   yerr=[[s[f"N256_A{A}"]["m_mean"][0] - s[f"N256_A{A}"]["m_mean"][1] for A in As], [s[f"N256_A{A}"]["m_mean"][2] - s[f"N256_A{A}"]["m_mean"][0] for A in As]],
                   c=cm(i), marker="o", capsize=2, label=lab(c))
    ax[0].plot(Rs, [s[f"N256_A{A}"]["ceil_mean"][0] for A in As], c=cm(i), ls=":", lw=1)
    ax[2].plot([s[f"N256_A{A}"]["alpha_cold"][0] for A in As[1:]], [s[f"N256_A{A}"]["alpha_hot"][0] for A in As[1:]], c=cm(i), marker="o", label=lab(c))
ax[0].set_xlabel("chain length R_A = floor(A/Δ)+1 (N=256, Δ=16)"); ax[0].set_ylabel("worst-chain weight m = min_j α_{p_j}(H_j)")
ax[0].set_title("(a) trained heads vs Theorem-1 ceiling (dotted)", fontsize=10); ax[0].legend(fontsize=6)
for i, c in enumerate(order):
    if np.isinf(c[1]): continue
    tags = summ[lab(c)].get("prop4_tags", [])
    if not tags: continue
    Rs = [2, 4, 8, 16, 32]
    ah = [np.mean([row["alpha_hot"] for t in tags for row in p4[t] if row["N"] == 256 and row["R"] == R]) for R in Rs]
    ce = [np.mean([row["ceiling"] for t in tags for row in p4[t] if row["N"] == 256 and row["R"] == R]) for R in Rs]
    ax[1].plot(Rs, ah, c=cm(i), marker="o", label=lab(c)); ax[1].plot(Rs, ce, c=cm(i), ls=":", lw=1)
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("versions R (latest at age 0, evenly spaced), N=256"); ax[1].set_ylabel("weight on latest version")
ax[1].set_title("(b) age-0 cell vs Prop-4 plug-in ceiling (dotted)", fontsize=10); ax[1].legend(fontsize=6)
ax[2].set_xlabel("α cold (H_1: single write at age A)"); ax[2].set_ylabel("α hot (H_R: latest at age < Δ)")
ax[2].set_title("(c) hot vs cold weight along the chain, A=16..192, N=256", fontsize=10); ax[2].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_single_head.png"), dpi=150); print("saved")
