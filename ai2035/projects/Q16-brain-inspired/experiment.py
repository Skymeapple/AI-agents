"""Q16 类脑智能与低能耗：ANN→SNN 转换的精度–时延–能耗前沿。

设计（参数在运行前固定）：
- 数据：sklearn digits（8x8，像素/16 ∈ [0,1]），每个种子分层划分 70/30。
- ANN：ReLU MLP 64-128-64-10（sklearn MLPClassifier, adam, alpha=1e-4, max_iter=400）。
- 转换：Rueckauer et al. 2017 的数据驱动归一化：λ_l = 训练集上第 l 层激活的 p 分位数；
  W'_l = W_l λ_{l-1}/λ_l, b'_l = b_l/λ_l；IF 神经元阈值 1；
  对照：不归一化 / max 归一化 (Diehl 2015) / 99.9 分位数归一化；复位：减法 vs 置零；
  神经元：IF vs LIF(τ=50,20,5 步)；输入编码：确定性 IF 速率编码 vs Poisson。
  输出层不发放，累积膜电位取 argmax。
- 计数：SNN 突触操作 = 每个突触前脉冲 × 扇出（累加 AC）；偏置每步每神经元 1 次 AC；
  LIF 泄漏每步每神经元 1 次乘法。ANN：稠密 MAC = Σ fan_in·fan_out；
  零跳过 ANN：Σ nnz(a_{l-1})·fan_out。
- 能耗（每操作，45nm，Horowitz 2014 ISSCC）：FP32 乘 3.7 pJ、加 0.9 pJ；INT8 乘 0.2 pJ、加 0.03 pJ。
  仅计算算术能耗，忽略访存（效度威胁中讨论）。
- 5 个种子；时间步检查点 1..256。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng, save_results  # noqa: E402

from scipy import stats  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
TMAX = 256
CHECK = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ENERGY = {  # pJ per op, Horowitz 2014 (45nm)
    "fp32": {"mult": 3.7, "add": 0.9},
    "int8": {"mult": 0.2, "add": 0.03},
}
for k in ENERGY:
    ENERGY[k]["mac"] = ENERGY[k]["mult"] + ENERGY[k]["add"]

CONFIGS = {
    "IF_p99.9_sub": dict(norm="p99.9", reset="sub", tau=None, enc="if"),
    "IF_max_sub": dict(norm="max", reset="sub", tau=None, enc="if"),
    "IF_none_sub": dict(norm="none", reset="sub", tau=None, enc="if"),
    "IF_p99.9_zero": dict(norm="p99.9", reset="zero", tau=None, enc="if"),
    "LIF50_p99.9_sub": dict(norm="p99.9", reset="sub", tau=50.0, enc="if"),
    "LIF20_p99.9_sub": dict(norm="p99.9", reset="sub", tau=20.0, enc="if"),
    "LIF5_p99.9_sub": dict(norm="p99.9", reset="sub", tau=5.0, enc="if"),
    "IF_p99.9_sub_poisson": dict(norm="p99.9", reset="sub", tau=None, enc="poisson"),
}


def ann_acts(Ws, bs, X):
    acts = [X]
    a = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = a @ W + b
        a = np.maximum(z, 0) if i < len(Ws) - 1 else z
        acts.append(a)
    return acts


def normalize(Ws, bs, Xtr, norm):
    if norm == "none":
        return [W.copy() for W in Ws], [b.copy() for b in bs]
    acts = ann_acts(Ws, bs, Xtr)
    lam = [1.0]  # input already in [0,1]
    for l in range(1, len(Ws) + 1):
        a = acts[l][acts[l] > 0] if l < len(Ws) else np.maximum(acts[l], 0)
        a = a[a > 0]
        lam.append(float(a.max()) if norm == "max" else float(np.percentile(a, 99.9)))
    Wn = [W * lam[l] / lam[l + 1] for l, W in enumerate(Ws)]
    bn = [b / lam[l + 1] for l, b in enumerate(bs)]
    return Wn, bn


def simulate(Wn, bn, X, cfg, g):
    """Returns accuracy-relevant predictions at checkpoints and op counts (per sample)."""
    B = X.shape[0]
    L = len(Wn)
    sizes = [W.shape[1] for W in Wn]
    V = [np.zeros((B, n)) for n in sizes]
    Vin = np.zeros_like(X)
    decay = 1.0 if cfg["tau"] is None else float(np.exp(-1.0 / cfg["tau"]))
    spikes_cum = np.zeros(L)  # spikes emitted by layer 0..L-1 (input + hidden), summed over batch
    out = {}
    for t in range(1, TMAX + 1):
        # input encoding
        if cfg["enc"] == "if":
            Vin += X
            s = (Vin >= 1.0).astype(float)
            Vin -= s
        else:
            s = (g.random(X.shape) < X).astype(float)
        spikes_cum[0] += s.sum()
        for l in range(L):
            inp = s @ Wn[l] + bn[l]
            if l < L - 1:
                V[l] = V[l] * decay + inp
                s = (V[l] >= 1.0).astype(float)
                if cfg["reset"] == "sub":
                    V[l] -= s
                else:
                    V[l] *= (1 - s)
                spikes_cum[l + 1] += s.sum()
            else:
                V[l] += inp  # output integrator (no leak, no spike)
        if t in CHECK:
            fan_out = np.array(sizes, dtype=float)
            syn_ac = float((spikes_cum * fan_out).sum() / B)
            bias_ac = float(sum(sizes) * t)
            leak_mult = float(sum(sizes[:-1]) * t) if cfg["tau"] is not None else 0.0
            out[t] = dict(pred=V[-1].argmax(1), syn_ac=syn_ac, bias_ac=bias_ac,
                          leak_mult=leak_mult,
                          spikes_per_neuron=(spikes_cum / B / np.array([X.shape[1]] + sizes[:-1])).tolist())
    return out


def energy_snn(o, prec):
    e = ENERGY[prec]
    return (o["syn_ac"] + o["bias_ac"]) * e["add"] + o["leak_mult"] * e["mult"]


res = {"config": {"seeds": SEEDS, "tmax": TMAX, "checkpoints": CHECK, "energy_pJ": ENERGY,
                  "arch": [64, 128, 64, 10], "configs": CONFIGS}}
per_seed = {name: {t: [] for t in CHECK} for name in CONFIGS}
per_seed_E = {name: {p: {t: [] for t in CHECK} for p in ENERGY} for name in CONFIGS}
per_seed_spn = {name: {t: [] for t in CHECK} for name in CONFIGS}
ann_acc, ann_dense_ops, ann_sparse_ops, ann_E = [], [], [], {p: {"dense": [], "sparse": []} for p in ENERGY}
ann_nnz_frac = []

X, y = load_digits(return_X_y=True)
X = X / 16.0
for sd in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
                        max_iter=400, random_state=sd)
    clf.fit(Xtr, ytr)
    Ws, bs = clf.coefs_, clf.intercepts_
    acc = float((clf.predict(Xte) == yte).mean())
    ann_acc.append(acc)
    acts = ann_acts(Ws, bs, Xte)
    dense = sum(W.shape[0] * W.shape[1] for W in Ws)
    sparse = float(np.mean(sum((acts[l] > 0).sum(1) * Ws[l].shape[1] for l in range(len(Ws)))))
    nbias = sum(W.shape[1] for W in Ws)
    ann_dense_ops.append(dense)
    ann_sparse_ops.append(sparse)
    ann_nnz_frac.append(sparse / dense)
    for p in ENERGY:
        ann_E[p]["dense"].append(dense * ENERGY[p]["mac"] + nbias * ENERGY[p]["add"])
        ann_E[p]["sparse"].append(sparse * ENERGY[p]["mac"] + nbias * ENERGY[p]["add"])
    for ci, (name, cfg) in enumerate(CONFIGS.items()):
        Wn, bn = normalize(Ws, bs, Xtr, cfg["norm"])
        out = simulate(Wn, bn, Xte, cfg, rng(1000 * sd + ci))
        for t in CHECK:
            per_seed[name][t].append(float((out[t]["pred"] == yte).mean()))
            per_seed_spn[name][t].append(float(np.mean(out[t]["spikes_per_neuron"][1:])))
            for p in ENERGY:
                per_seed_E[name][p][t].append(energy_snn(out[t], p))
    print(f"seed {sd} done, ANN acc {acc:.4f}, {time.time()-T0:.1f}s", flush=True)

res["ann"] = {
    "acc": mean_ci(ann_acc), "acc_per_seed": ann_acc,
    "dense_mac": float(np.mean(ann_dense_ops)),
    "sparse_mac": mean_ci(ann_sparse_ops), "nnz_frac": mean_ci(ann_nnz_frac),
    "energy_nJ": {p: {k: mean_ci(np.array(v) / 1000) for k, v in ann_E[p].items()} for p in ENERGY},
}
res["snn"] = {}
for name in CONFIGS:
    d = {}
    for t in CHECK:
        a = np.array(per_seed[name][t])
        gap = np.array(ann_acc) - a
        d[str(t)] = {
            "acc": mean_ci(a), "gap_pp": mean_ci(100 * gap),
            "hidden_spikes_per_neuron": mean_ci(per_seed_spn[name][t]),
            "energy_nJ": {p: mean_ci(np.array(per_seed_E[name][p][t]) / 1000) for p in ENERGY},
            "ratio_vs_dense": {p: mean_ci(np.array(per_seed_E[name][p][t]) / np.array(ann_E[p]["dense"])) for p in ENERGY},
            "ratio_vs_sparse": {p: mean_ci(np.array(per_seed_E[name][p][t]) / np.array(ann_E[p]["sparse"])) for p in ENERGY},
        }
    res["snn"][name] = d

# iso-accuracy analysis (per seed): smallest T with gap <= 1pp for main config
iso = {}
for name in CONFIGS:
    Ts, rd, rs, ok = [], {p: [] for p in ENERGY}, {p: [] for p in ENERGY}, 0
    for i in range(len(SEEDS)):
        Tsel = next((t for t in CHECK if ann_acc[i] - per_seed[name][t][i] <= 0.01), None)
        if Tsel is None:
            continue
        ok += 1
        Ts.append(Tsel)
        for p in ENERGY:
            e = per_seed_E[name][p][Tsel][i]
            rd[p].append(e / ann_E[p]["dense"][i])
            rs[p].append(e / ann_E[p]["sparse"][i])
    iso[name] = {"n_seeds_reaching_1pp": ok, "T_1pp_per_seed": Ts,
                 "T_1pp_median": float(np.median(Ts)) if Ts else None,
                 "energy_ratio_vs_dense": {p: mean_ci(rd[p]) if rd[p] else None for p in ENERGY},
                 "energy_ratio_vs_sparse": {p: mean_ci(rs[p]) if rs[p] else None for p in ENERGY}}
res["iso_accuracy_1pp"] = iso

# break-even: SNN arithmetic energy = ANN dense when fan-out-weighted spikes per neuron = E_MAC/E_AC
res["break_even"] = {p: {"mac_over_ac": ENERGY[p]["mac"] / ENERGY[p]["add"],
                         "vs_sparse_ann_spikes_per_neuron": float(np.mean(ann_nnz_frac)) * ENERGY[p]["mac"] / ENERGY[p]["add"]}
                     for p in ENERGY}
# empirical T at which SNN (main) energy crosses ANN dense / sparse (mean curves)
cross = {}
for p in ENERGY:
    for base in ["dense", "sparse"]:
        eb = np.mean(ann_E[p][base])
        es = [np.mean(per_seed_E["IF_p99.9_sub"][p][t]) for t in CHECK]
        tcross = next((t for t, e in zip(CHECK, es) if e > eb), None)
        cross[f"{p}_{base}"] = {"first_T_where_snn_costlier": tcross}
res["crossover"] = cross

# paired test: p99.9 vs max normalization at T=32 and LIF5 vs IF at T=64
def paired(a, b, t):
    x, z = np.array(per_seed[a][t]), np.array(per_seed[b][t])
    st = stats.ttest_rel(x, z)
    d = (x - z).mean() / (x - z).std(ddof=1) if (x - z).std(ddof=1) > 0 else float("inf")
    return {"diff_pp": mean_ci(100 * (x - z)), "t": float(st.statistic), "p": float(st.pvalue), "cohen_dz": float(d)}

res["tests"] = {
    "p99.9_vs_max_T32": paired("IF_p99.9_sub", "IF_max_sub", 32),
    "p99.9_vs_none_T32": paired("IF_p99.9_sub", "IF_none_sub", 32),
    "sub_vs_zero_T32": paired("IF_p99.9_sub", "IF_p99.9_zero", 32),
    "IF_vs_LIF5_T64": paired("IF_p99.9_sub", "LIF5_p99.9_sub", 64),
    "IF_vs_LIF20_T64": paired("IF_p99.9_sub", "LIF20_p99.9_sub", 64),
    "IF_vs_poisson_T64": paired("IF_p99.9_sub", "IF_p99.9_sub_poisson", 64),
}
res["runtime_s"] = time.time() - T0

# ---------------- figures
P = plt()
out_dir = results_dir(__file__)
fig, ax = P.subplots(1, 2, figsize=(11, 4))
for name in CONFIGS:
    m = [res["snn"][name][str(t)]["acc"][0] for t in CHECK]
    c = [res["snn"][name][str(t)]["acc"][1] for t in CHECK]
    ax[0].errorbar(CHECK, m, yerr=c, label=name, marker="o", ms=3, capsize=2)
ax[0].axhline(res["ann"]["acc"][0], color="k", ls="--", label="ANN")
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("时间步 T"); ax[0].set_ylabel("测试准确率")
ax[0].set_title("SNN 准确率 vs 仿真时间步"); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1.02)
for p, mk in [("fp32", "o"), ("int8", "s")]:
    e = [res["snn"]["IF_p99.9_sub"][str(t)]["energy_nJ"][p][0] for t in CHECK]
    a = [res["snn"]["IF_p99.9_sub"][str(t)]["acc"][0] for t in CHECK]
    ax[1].plot(e, a, marker=mk, label=f"SNN(IF,p99.9) {p}")
    for t, ei, ai in zip(CHECK, e, a):
        if p == "fp32":
            ax[1].annotate(str(t), (ei, ai), fontsize=7)
    ax[1].scatter([res["ann"]["energy_nJ"][p]["dense"][0]], [res["ann"]["acc"][0]], marker="*", s=120, label=f"ANN 稠密 {p}")
    ax[1].scatter([res["ann"]["energy_nJ"][p]["sparse"][0]], [res["ann"]["acc"][0]], marker="P", s=80, label=f"ANN 零跳过 {p}")
ax[1].set_xscale("log"); ax[1].set_xlabel("每样本算术能耗 (nJ, 对数)"); ax[1].set_ylabel("准确率")
ax[1].set_title("精度–能耗前沿（数字为 T）"); ax[1].legend(fontsize=7); ax[1].set_ylim(0.5, 1.0)
fig.tight_layout(); fig.savefig(out_dir / "accuracy_energy.png"); P.close(fig)

fig, ax = P.subplots(figsize=(5.5, 4))
spn = [res["snn"]["IF_p99.9_sub"][str(t)]["hidden_spikes_per_neuron"][0] for t in CHECK]
ax.plot(CHECK, spn, marker="o", label="隐藏层平均脉冲数/神经元")
for p, ls in [("fp32", "--"), ("int8", ":")]:
    ax.axhline(res["break_even"][p]["mac_over_ac"], ls=ls, color="r", label=f"盈亏平衡 vs 稠密 ANN ({p})")
    ax.axhline(res["break_even"][p]["vs_sparse_ann_spikes_per_neuron"], ls=ls, color="g", label=f"盈亏平衡 vs 零跳过 ANN ({p})")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("T"); ax.set_ylabel("脉冲数 / 神经元 / 推理")
ax.legend(fontsize=7); ax.set_title("脉冲稀疏度与能耗盈亏平衡")
fig.tight_layout(); fig.savefig(out_dir / "break_even.png"); P.close(fig)

save_results(__file__, res)
print(f"done in {time.time()-T0:.1f}s")
