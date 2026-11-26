"""Q16 前沿实验：超低时间步下，直接训练 SNN vs ANN→SNN 转换 vs 量化 ANN；计入访存后的能耗盈亏。

问题（2025–2026 争论）：一时间步/少时间步 SNN（直接训练或多阈值转换）是否仍保有能耗优势？
STEP (arXiv:2505.11151) 指出计入访存与位宽后量化 ANN 可能同样或更省；Sparsity Ceiling
(arXiv:2607.26648) 认为稀疏红利取决于任务。

设计（参数在运行前固定）：
- 数据：sklearn digits，每种子分层 70/30；像素/16。
- 直接训练 SNN：64-128-64-10，隐藏层 LIF（β=0.9，阈值 1，软复位，复位项不回传），
  第一层为直接（模拟值）编码、电流只算一次；输出层不发放，logits = 平均输出电流。
  代理梯度 σ'(v)=1/(1+5|v-1|)^2，BPTT，Adam(lr=2e-3)，批 64，40 轮；
  T ∈ {1,2,4}；发放率正则 λ ∈ {0, 0.02, 0.1}（惩罚每神经元每推理脉冲数）。
- 对照：ANN（同结构 ReLU，sklearn MLPClassifier）；ANN→SNN 转换（p99.9 归一化，IF，
  减法复位，T∈{1,2,4,8}）；激活量化 ANN（b=1,2,3 bit，即 2^b−1 个非零电平，
  对应“多阈值一步 SNN”的时间–空间等价）。
- 能耗模型（每推理，pJ）：算术 [Horowitz 2014, 45nm] FP32 MAC 4.6 / AC 0.9；INT8 MAC 0.23 / AC 0.03。
  访存：SRAM 每 bit 能耗 e_bit ∈ {10/64 (8KB), 20/64 (32KB, 主条件), 100/64 (1MB)} pJ
  [Horowitz 2014 64-bit 访问 10/20/100 pJ]；权重位宽 = 8(INT8) 或 32(FP32)；
  SNN 每个突触事件读 1 个权重（无跨时间步复用）；每隐藏神经元每时间步读+写 16-bit 膜电位；
  ANN 每个权重每推理读 1 次（零跳过 ANN 只读非零激活对应的行），每神经元激活写+读 1 次。
  第一层（模拟输入）两者都按稠密 MAC + 读权重计；另报告“仅隐藏层”能耗比（去掉共同的第一层）。
- 理论：盈亏平衡脉冲数 s* = f·(E_MAC+E_w)/(E_AC+E_w)（f=零跳过 ANN 非零率；稠密 ANN 时 f=1）。
- 5 个种子。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng  # noqa: E402

from scipy import stats  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
TS = [1, 2, 4]
LAMS = [0.0, 0.02, 0.1]
CONV_TS = [1, 2, 4, 8]
QBITS = [1, 2, 3]
BETA, THETA, EPOCHS, BATCH, LR = 0.9, 1.0, 40, 64, 2e-3
SIZES = [64, 128, 64, 10]
ARITH = {"fp32": {"mac": 4.6, "ac": 0.9, "wbits": 32, "abits": 32},
         "int8": {"mac": 0.23, "ac": 0.03, "wbits": 8, "abits": 8}}
EBIT = {"8KB": 10 / 64, "32KB": 20 / 64, "1MB": 100 / 64}
VBITS = 16


# ------------------------------------------------------------------ direct-trained SNN
def sg(v):
    return 1.0 / (1.0 + 5.0 * np.abs(v - THETA)) ** 2


def snn_forward(P, X, T, keep=False):
    W1, b1, W2, b2, W3, b3 = P
    B = X.shape[0]
    I1 = X @ W1 + b1
    V1 = np.zeros((B, SIZES[1])); V2 = np.zeros((B, SIZES[2]))
    S1p = np.zeros_like(V1); S2p = np.zeros_like(V2)
    out = np.zeros((B, SIZES[3]))
    cache = []
    for _ in range(T):
        V1 = BETA * V1 + I1 - THETA * S1p
        S1 = (V1 >= THETA).astype(float)
        V2 = BETA * V2 + S1 @ W2 + b2 - THETA * S2p
        S2 = (V2 >= THETA).astype(float)
        out += S2 @ W3 + b3
        if keep:
            cache.append((V1, S1, V2, S2))
        S1p, S2p = S1, S2
    return out / T, cache


def snn_train(X, y, T, lam, g):
    def init(i, o):
        return g.standard_normal((i, o)) * np.sqrt(2.0 / i), np.zeros(o)
    W1, b1 = init(64, 128); W2, b2 = init(128, 64); W3, b3 = init(64, 10)
    W1 *= 1.5  # start with some activity
    P = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    step = 0
    nh = SIZES[1] + SIZES[2]
    for ep in range(EPOCHS):
        idx = g.permutation(len(X))
        for s in range(0, len(X), BATCH):
            bi = idx[s:s + BATCH]; xb, yb = X[bi], y[bi]; B = len(bi)
            logits, cache = snn_forward(P, xb, T, keep=True)
            z = logits - logits.max(1, keepdims=True)
            pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            gl = pr.copy(); gl[np.arange(B), yb] -= 1; gl /= B
            gW = [np.zeros_like(p) for p in P]
            dV1n = np.zeros((B, SIZES[1])); dV2n = np.zeros((B, SIZES[2]))
            dI1 = np.zeros((B, SIZES[1]))
            reg = lam / nh / B
            for t in range(T - 1, -1, -1):
                V1, S1, V2, S2 = cache[t]
                gW[4] += S2.T @ (gl / T); gW[5] += gl.sum(0) / T
                dS2 = (gl / T) @ P[4].T + reg
                dV2 = dS2 * sg(V2) + BETA * dV2n
                gW[2] += S1.T @ dV2; gW[3] += dV2.sum(0)
                dS1 = dV2 @ P[2].T + reg
                dV1 = dS1 * sg(V1) + BETA * dV1n
                dI1 += dV1
                dV1n, dV2n = dV1, dV2
            gW[0] = xb.T @ dI1; gW[1] = dI1.sum(0)
            step += 1
            for k in range(6):
                m[k] = 0.9 * m[k] + 0.1 * gW[k]; v[k] = 0.999 * v[k] + 0.001 * gW[k] ** 2
                P[k] -= LR * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
    return P


def snn_eval(P, X, y, T):
    logits, cache = snn_forward(P, X, T, keep=True)
    acc = float((logits.argmax(1) == y).mean())
    s1 = sum(c[1].sum() for c in cache) / len(X) / SIZES[1]
    s2 = sum(c[3].sum() for c in cache) / len(X) / SIZES[2]
    return acc, float(s1), float(s2)


# ------------------------------------------------------------------ energy models
def energy_ann(prec, ebit, nnz_in=None):
    """dense if nnz_in None; else zero-skipping with nnz counts per layer input (list of 3 floats; layer-1 input dense)."""
    a = ARITH[prec]; ew = a["wbits"] * ebit; ea = a["abits"] * ebit
    E_ar = E_mem = 0.0
    for l in range(3):
        fi, fo = SIZES[l], SIZES[l + 1]
        rows = fi if (nnz_in is None or l == 0) else nnz_in[l]
        E_ar += rows * fo * a["mac"] + fo * a["ac"]
        E_mem += rows * fo * ew + 2 * fo * ea  # weight reads + activation write/read
    return E_ar, E_mem


def energy_snn(prec, ebit, s1, s2, T):
    a = ARITH[prec]; ew = a["wbits"] * ebit; ev = VBITS * ebit
    # layer 1: analog input, dense MAC once
    E_ar = SIZES[0] * SIZES[1] * a["mac"] + SIZES[1] * a["ac"]
    E_mem = SIZES[0] * SIZES[1] * ew + SIZES[1] * 2 * ev
    # spikes: s1 spikes/neuron (layer1, 128 neurons) -> fan-out 64 ; s2 (64 neurons) -> fan-out 10
    ev1, ev2 = s1 * SIZES[1] * SIZES[2], s2 * SIZES[2] * SIZES[3]
    E_ar += (ev1 + ev2) * a["ac"] + T * (SIZES[2] + SIZES[3]) * a["ac"]  # bias/leak adds per step
    E_mem += (ev1 + ev2) * ew + T * (SIZES[1] + SIZES[2]) * 2 * ev  # weight reads + membrane rd/wr
    return E_ar, E_mem


def l1_ann(prec, ebit):
    a = ARITH[prec]; ew = a["wbits"] * ebit; ea = a["abits"] * ebit
    return SIZES[0] * SIZES[1] * (a["mac"] + ew) + SIZES[1] * a["ac"] + 2 * SIZES[1] * ea


def l1_snn(prec, ebit):
    a = ARITH[prec]; ew = a["wbits"] * ebit
    return SIZES[0] * SIZES[1] * (a["mac"] + ew) + SIZES[1] * a["ac"] + SIZES[1] * 2 * VBITS * ebit


def breakeven(prec, ebit, f=1.0):
    a = ARITH[prec]; ew = a["wbits"] * ebit
    return f * (a["mac"] + ew) / (a["ac"] + ew)


# ------------------------------------------------------------------ conversion & quantized ANN
def ann_acts(Ws, bs, X):
    acts = [X]; a = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = a @ W + b
        a = np.maximum(z, 0) if i < len(Ws) - 1 else z
        acts.append(a)
    return acts


def lam_p999(Ws, bs, Xtr):
    acts = ann_acts(Ws, bs, Xtr)
    lam = [1.0]
    for l in range(1, len(Ws) + 1):
        a = acts[l][acts[l] > 0]
        lam.append(float(np.percentile(a, 99.9)))
    return lam


def convert_eval(Ws, bs, Xtr, X, y, T):
    lam = lam_p999(Ws, bs, Xtr)
    Wn = [W * lam[l] / lam[l + 1] for l, W in enumerate(Ws)]; bn = [b / lam[l + 1] for l, b in enumerate(bs)]
    B = len(X); V = [np.zeros((B, W.shape[1])) for W in Wn]
    I1 = X @ Wn[0] + bn[0]; sp = [0.0, 0.0]
    for _ in range(T):
        V[0] += I1; s = (V[0] >= 1).astype(float); V[0] -= s; sp[0] += s.sum()
        V[1] += s @ Wn[1] + bn[1]; s2 = (V[1] >= 1).astype(float); V[1] -= s2; sp[1] += s2.sum()
        V[2] += s2 @ Wn[2] + bn[2]
    return float((V[2].argmax(1) == y).mean()), sp[0] / B / SIZES[1], sp[1] / B / SIZES[2]


def quant_eval(Ws, bs, Xtr, X, y, bits):
    lam = lam_p999(Ws, bs, Xtr); L = 2 ** bits - 1
    a = X; nnz = [64.0]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = a @ W + b
        if i < len(Ws) - 1:
            a = np.clip(np.round(np.maximum(z, 0) / lam[i + 1] * L), 0, L) * lam[i + 1] / L
            nnz.append(float((a > 0).sum(1).mean()))
        else:
            a = z
    return float((a.argmax(1) == y).mean()), nnz


# ------------------------------------------------------------------ main loop
X, y = load_digits(return_X_y=True); X = X / 16.0
R = {"ann": [], "ann_nnz": [], "snn": {}, "conv": {T: [] for T in CONV_TS}, "quant": {b: [] for b in QBITS}}
for T in TS:
    for lam in LAMS:
        R["snn"][(T, lam)] = []
for sd in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4, max_iter=400, random_state=sd).fit(Xtr, ytr)
    Ws, bs = clf.coefs_, clf.intercepts_
    R["ann"].append(float((clf.predict(Xte) == yte).mean()))
    acts = ann_acts(Ws, bs, Xte)
    R["ann_nnz"].append([64.0, float((acts[1] > 0).sum(1).mean()), float((acts[2] > 0).sum(1).mean())])
    for T in CONV_TS:
        R["conv"][T].append(convert_eval(Ws, bs, Xtr, Xte, yte, T))
    for b in QBITS:
        R["quant"][b].append(quant_eval(Ws, bs, Xtr, Xte, yte, b))
    for T in TS:
        for lam in LAMS:
            P = snn_train(Xtr, ytr, T, lam, rng(100 * sd + 10 * T + int(100 * lam)))
            R["snn"][(T, lam)].append(snn_eval(P, Xte, yte, T))
    print(f"seed {sd} done {time.time()-T0:.1f}s", flush=True)

# ------------------------------------------------------------------ aggregate
out = {"config": dict(seeds=SEEDS, Ts=TS, lams=LAMS, conv_Ts=CONV_TS, qbits=QBITS, beta=BETA, epochs=EPOCHS,
                      arith_pJ=ARITH, ebit_pJ=EBIT, vbits=VBITS, sizes=SIZES)}
ann = np.array(R["ann"]); nnz = np.array(R["ann_nnz"])
out["ann_acc"] = mean_ci(ann)
out["ann_nnz_frac_hidden"] = mean_ci((nnz[:, 1:] / np.array([128, 64])).mean(1))
f_nnz = float((nnz[:, 1] * 64 + nnz[:, 2] * 10).mean() / (128 * 64 + 64 * 10))  # fan-out weighted
out["ann_nnz_frac_fanout_weighted"] = f_nnz
out["breakeven_s"] = {p: {k: {"dense": breakeven(p, e), "sparse": breakeven(p, e, f_nnz)} for k, e in EBIT.items()} for p in ARITH}
out["breakeven_s"]["arith_only"] = {p: {"dense": ARITH[p]["mac"] / ARITH[p]["ac"], "sparse": f_nnz * ARITH[p]["mac"] / ARITH[p]["ac"]} for p in ARITH}

EA = {}  # ANN energies per seed
for p in ARITH:
    for k, e in EBIT.items():
        d = [energy_ann(p, e) for _ in SEEDS]
        s = [energy_ann(p, e, nnz[i]) for i in range(len(SEEDS))]
        EA[(p, k, "dense")] = np.array([a + m for a, m in d]); EA[(p, k, "sparse")] = np.array([a + m for a, m in s])
        EA[(p, k, "dense_arith")] = np.array([a for a, m in d]); EA[(p, k, "sparse_arith")] = np.array([a for a, m in s])
out["ann_energy_pJ"] = {f"{p}|{k}|{v}": mean_ci(EA[(p, k, v)]) for p in ARITH for k in EBIT for v in ["dense", "sparse", "dense_arith", "sparse_arith"]}


def snn_block(accs, s1s, s2s, T):
    d = {"acc": mean_ci(accs), "gap_pp": mean_ci(100 * (ann - np.array(accs))),
         "spikes_per_neuron_l1": mean_ci(s1s), "spikes_per_neuron_l2": mean_ci(s2s)}
    sw = (np.array(s1s) * 128 * 64 + np.array(s2s) * 64 * 10) / (128 * 64 + 64 * 10)
    d["spikes_fanout_weighted"] = mean_ci(sw)
    for p in ARITH:
        for k, e in EBIT.items():
            es = np.array([sum(energy_snn(p, e, a, b, T)) for a, b in zip(s1s, s2s)])
            ea = np.array([energy_snn(p, e, a, b, T)[0] for a, b in zip(s1s, s2s)])
            d[f"E_{p}_{k}"] = mean_ci(es)
            d[f"ratio_vs_dense_{p}_{k}"] = mean_ci(es / EA[(p, k, "dense")])
            d[f"ratio_vs_sparse_{p}_{k}"] = mean_ci(es / EA[(p, k, "sparse")])
            if k == "32KB":
                eh = es - l1_snn(p, e)
                d[f"ratio_hidden_vs_sparse_{p}_{k}"] = mean_ci(eh / (EA[(p, k, "sparse")] - l1_ann(p, e)))
                d[f"ratio_arith_vs_dense_{p}"] = mean_ci(ea / EA[(p, k, "dense_arith")])
                d[f"ratio_arith_vs_sparse_{p}"] = mean_ci(ea / EA[(p, k, "sparse_arith")])
    return d, sw


out["snn_direct"] = {}
crit = []  # (predicted win, actual win) for criterion test
for (T, lam), lst in R["snn"].items():
    a = [x[0] for x in lst]; s1 = [x[1] for x in lst]; s2 = [x[2] for x in lst]
    d, sw = snn_block(a, s1, s2, T)
    out["snn_direct"][f"T{T}_lam{lam}"] = d
    for p in ARITH:
        for k, e in EBIT.items():
            for i in range(len(SEEDS)):
                es = sum(energy_snn(p, e, s1[i], s2[i], T))
                # theory predicts win vs sparse ANN iff spikes < s*, ignoring neuron-state terms
                pred = sw[i] < breakeven(p, e, f_nnz)
                act = es < EA[(p, k, "sparse")][i]
                crit.append((pred, act, p, k))
out["criterion_agreement"] = {
    "all": float(np.mean([a == b for a, b, _, _ in crit])),
    "n": len(crit),
    "by_precision": {p: float(np.mean([a == b for a, b, pp, _ in crit if pp == p])) for p in ARITH},
    "frac_snn_wins_vs_sparse": float(np.mean([b for _, b, _, _ in crit])),
}
out["conversion"] = {}
for T in CONV_TS:
    lst = R["conv"][T]
    d, _ = snn_block([x[0] for x in lst], [x[1] for x in lst], [x[2] for x in lst], T)
    out["conversion"][f"T{T}"] = d
out["quant_ann"] = {f"{b}bit": {"acc": mean_ci([x[0] for x in R["quant"][b]]),
                                "gap_pp": mean_ci(100 * (ann - np.array([x[0] for x in R["quant"][b]]))),
                                "levels": 2 ** b - 1, "equiv_T_binary_spikes": 2 ** b - 1} for b in QBITS}
# paired test: direct T=1 lam0 vs conversion T=1 / T=4
dT1 = np.array([x[0] for x in R["snn"][(1, 0.0)]])
out["tests"] = {
    "directT1_vs_convT1": {"diff_pp": mean_ci(100 * (dT1 - np.array([x[0] for x in R["conv"][1]]))),
                           "p": float(stats.ttest_rel(dT1, [x[0] for x in R["conv"][1]]).pvalue)},
    "directT1_vs_convT4": {"diff_pp": mean_ci(100 * (dT1 - np.array([x[0] for x in R["conv"][4]]))),
                           "p": float(stats.ttest_rel(dT1, [x[0] for x in R["conv"][4]]).pvalue)},
    "directT2_vs_ann": {"diff_pp": mean_ci(100 * (np.array([x[0] for x in R["snn"][(2, 0.0)]]) - ann)),
                        "p": float(stats.ttest_rel([x[0] for x in R["snn"][(2, 0.0)]], ann).pvalue)},
}
# best iso-accuracy (<=1pp, mean) direct config energy advantage vs INT8 sparse ANN at 32KB
iso = {k: v for k, v in out["snn_direct"].items() if v["gap_pp"][0] <= 1.0}
out["iso_1pp_configs"] = {k: {"ratio_vs_sparse_int8_32KB": v["ratio_vs_sparse_int8_32KB"],
                              "ratio_arith_vs_sparse_int8": v["ratio_arith_vs_sparse_int8"],
                              "ratio_vs_sparse_fp32_32KB": v["ratio_vs_sparse_fp32_32KB"]} for k, v in iso.items()}
out["runtime_s"] = time.time() - T0
od = results_dir(__file__)
(od / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

# ------------------------------------------------------------------ figures
P = plt()
fig, ax = P.subplots(1, 2, figsize=(12, 4.3))
for T, mk in zip(TS, "os^"):
    xs = [out["snn_direct"][f"T{T}_lam{l}"]["spikes_fanout_weighted"][0] for l in LAMS]
    ys = [out["snn_direct"][f"T{T}_lam{l}"]["acc"][0] for l in LAMS]
    ax[0].plot(xs, ys, marker=mk, label=f"直接训练 T={T}（λ=0→0.1）")
for T in CONV_TS:
    d = out["conversion"][f"T{T}"]
    ax[0].scatter(d["spikes_fanout_weighted"][0], d["acc"][0], marker="x", color="k")
    ax[0].annotate(f"转换T={T}", (d["spikes_fanout_weighted"][0], d["acc"][0]), fontsize=7)
ax[0].axhline(out["ann_acc"][0], ls="--", color="gray", label="ANN")
for p, ls in [("int8", ":"), ("fp32", "-.")]:
    ax[0].axvline(out["breakeven_s"][p]["32KB"]["sparse"], ls=ls, color="r", label=f"s* vs 稀疏ANN ({p},32KB)")
ax[0].set_xlabel("每神经元每推理脉冲数（扇出加权）"); ax[0].set_ylabel("测试准确率"); ax[0].set_ylim(0.0, 1.0)
ax[0].legend(fontsize=7); ax[0].set_title("精度–稀疏度：直接训练 vs 转换")
labels, arith, full = [], [], []
for T in TS:
    for l in LAMS:
        k = f"T{T}_lam{l}"; labels.append(f"T{T},λ{l}")
        arith.append(out["snn_direct"][k]["ratio_arith_vs_sparse_int8"][0])
        full.append(out["snn_direct"][k]["ratio_vs_sparse_int8_32KB"][0])
x = np.arange(len(labels))
ax[1].bar(x - 0.2, arith, 0.4, label="仅算术")
ax[1].bar(x + 0.2, full, 0.4, label="算术+访存(32KB)")
ax[1].axhline(1, color="k", ls="--"); ax[1].set_yscale("log")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=60, fontsize=7)
ax[1].set_ylabel("SNN 能耗 / INT8 零跳过 ANN"); ax[1].legend(fontsize=8); ax[1].set_title("计入访存前后的能耗比")
fig.tight_layout(); fig.savefig(od / "frontier_accuracy_energy.png"); P.close(fig)

fig, ax = P.subplots(figsize=(6, 4))
ebs = np.linspace(0, 2.0, 41)
for p, c in [("int8", "C0"), ("fp32", "C1")]:
    a = ARITH[p]
    ax.plot(ebs, [f_nnz * (a["mac"] + a["wbits"] * e) / (a["ac"] + a["wbits"] * e) for e in ebs], color=c, label=f"s* vs 稀疏 ANN ({p})")
    ax.plot(ebs, [(a["mac"] + a["wbits"] * e) / (a["ac"] + a["wbits"] * e) for e in ebs], color=c, ls="--", label=f"s* vs 稠密 ANN ({p})")
for k, e in EBIT.items():
    ax.axvline(e, color="gray", ls=":"); ax.annotate(k, (e, 6), fontsize=7)
best = out["snn_direct"]["T1_lam0.1"]["spikes_fanout_weighted"][0]
ax.axhline(best, color="g", ls="-.", label="直接训练 T=1,λ=0.1 实测脉冲数")
ax.set_yscale("log"); ax.set_xlabel("SRAM 每 bit 访问能耗 (pJ)"); ax.set_ylabel("盈亏平衡脉冲数 s*")
ax.legend(fontsize=7); ax.set_title("访存使盈亏平衡点塌缩到 ≈ 激活密度")
fig.tight_layout(); fig.savefig(od / "frontier_breakeven.png"); P.close(fig)
print(f"done {time.time()-T0:.1f}s")
