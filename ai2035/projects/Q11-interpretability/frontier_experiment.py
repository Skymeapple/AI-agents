"""Q11 前沿实验：稀疏自编码器（SAE）能否在叠加玩具模型中恢复真实特征？是否胜过简单基线？

设定（合成激活，真值已知；参照 Elhage et al. 2022 的叠加模型与 SynthSAEBench 思路）：
  * n=96 个真实特征方向 F_i ∈ R^32（随机单位向量，3 倍过完备），每个特征以概率 p 独立激活，
    幅值 ~ U(0,1)；激活 h = Σ_i s_i F_i + 0.01·噪声。期望激活数 k = p·n。
  * 实验 A（密度扫描）：k ∈ {1,2,4,8,16}；SAE（宽 192、L1）与基线：
      原始神经元、PCA、随机冻结解码器 SAE（只训练编码器；Korznikov et al. 2026 式随机基线）、
      学到的字典 + ISTA 推断（去掉"摊销差距"）、真字典 + ISTA（oracle 推断上限）、
      有监督线性探针（每个特征单独训练逻辑回归；上限）。
  * 实验 B（特征吸收）：层级结构——8 个父特征各有 3 个子特征，子特征只在父特征激活时出现；
      扫描 L1 系数 λ，测量父特征 latent 在子特征出现时"缺席"的比例（吸收率）。
  * 实验 C（宽度）：SAE 宽 m ∈ {48, 96, 192}，k=4。
输出：results/frontier.json，results/frontier_*.png
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common.util import SEED, mean_ci, plt, results_dir, rng

T0 = time.time()
D, NF = 32, 96
SEEDS = list(range(5))
KS = [1, 2, 4, 8, 16]
LAM = 0.3          # 主实验 L1 系数：在独立试验种子 99、k=4 上预先选定（使 L0≈2k），见 paper §5
WIDTH = 192
STEPS, BATCH, LR = 1200, 256, 3e-3
N_TEST = 3000
COS_THR = 0.9


def make_dict(g, n=NF, d=D):
    F = g.standard_normal((n, d))
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def sample_flat(g, F, p, n):
    act = g.random((n, F.shape[0])) < p
    S = act * g.random((n, F.shape[0]))
    H = S @ F + 0.01 * g.standard_normal((n, F.shape[1]))
    return H, S


def sample_hier(g, F, n, p_parent=0.06, p_child=0.3, n_par=8, n_ch=3, p_other=0.03):
    """features 0..7 parents, 8..31 children (3 per parent), 32..95 independent controls."""
    S = np.zeros((n, F.shape[0]))
    par = g.random((n, n_par)) < p_parent
    S[:, :n_par] = par * g.uniform(0.3, 1.0, (n, n_par))
    for i in range(n_par):
        ch = (g.random((n, n_ch)) < p_child) & par[:, [i]]
        idx = n_par + i * n_ch + np.arange(n_ch)
        S[:, idx] = ch * g.uniform(0.3, 1.0, (n, n_ch))
    k0 = n_par * (1 + n_ch)
    oth = g.random((n, F.shape[0] - k0)) < p_other
    S[:, k0:] = oth * g.uniform(0.3, 1.0, (n, F.shape[0] - k0))
    H = S @ F + 0.01 * g.standard_normal((n, F.shape[1]))
    return H, S


# ----------------------------------- SAE -----------------------------------
def train_sae(g, sampler, m, lam, frozen_dec=None, steps=STEPS):
    We = g.standard_normal((D, m)) * 0.1
    Wd = We.T.copy() if frozen_dec is None else frozen_dec.copy()
    Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    be, bd = np.zeros(m), np.zeros(D)
    params = [We, be, Wd, bd]
    mom = [np.zeros_like(p) for p in params]
    vel = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        X, _ = sampler(BATCH)
        Xc = X - bd
        pre = Xc @ We + be
        Z = np.maximum(pre, 0)
        Xh = Z @ Wd + bd
        R = Xh - X
        dXh = 2 * R / BATCH
        gWd = Z.T @ dXh
        gbd = dXh.sum(0)
        dZ = dXh @ Wd.T + lam * (Z > 0) / BATCH
        dpre = dZ * (pre > 0)
        gWe = Xc.T @ dpre
        gbe = dpre.sum(0)
        gbd = gbd - (dpre @ We.T).sum(0)
        grads = [gWe, gbe, gWd, gbd]
        if frozen_dec is not None:
            grads[2] = np.zeros_like(Wd)
        for i, (p, gr) in enumerate(zip(params, grads)):
            mom[i] = 0.9 * mom[i] + 0.1 * gr
            vel[i] = 0.999 * vel[i] + 0.001 * gr**2
            p -= LR * (mom[i] / (1 - 0.9**t)) / (np.sqrt(vel[i] / (1 - 0.999**t)) + 1e-8)
        Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    return dict(We=We, be=be, Wd=Wd, bd=bd)


def encode(sae, X):
    return np.maximum((X - sae["bd"]) @ sae["We"] + sae["be"], 0)


def ista(Dict, X, lam=0.02, iters=150):
    """非负 LASSO 推断：min ||x - z Dict||^2/2 + lam*|z|_1, z>=0。"""
    L = np.linalg.norm(Dict, 2) ** 2
    Z = np.zeros((X.shape[0], Dict.shape[0]))
    for _ in range(iters):
        G = (Z @ Dict - X) @ Dict.T
        Z = np.maximum(Z - G / L - lam / L, 0)
    return Z


def recovery(F, Wd):
    C = np.abs(F @ Wd.T)            # NF x m
    mx = C.max(1)
    return float(mx.mean()), float((mx > COS_THR).mean())


def best_latent_auroc(Z, S, max_feats=None):
    """每个真实特征：在所有 latent 中取单个 latent 检测"该特征是否激活"的最大 AUROC，再对特征取平均。"""
    y = S > 0
    out = []
    Zr = np.argsort(np.argsort(Z, axis=0), axis=0).astype(float)  # rank transform for fast AUROC
    n = Z.shape[0]
    for i in range(S.shape[1] if max_feats is None else max_feats):
        pos = y[:, i]
        npos = pos.sum()
        if npos == 0 or npos == n:
            continue
        # Mann–Whitney AUROC for all latents at once (ties ignored; Z has many zeros -> use average ranks)
        auc = (Zr[pos].sum(0) - npos * (npos - 1) / 2) / (npos * (n - npos))
        auc = np.maximum(auc, 1 - auc)   # 允许方向相反（对原始神经元/PCA 公平）
        out.append(auc.max())
    return float(np.mean(out))


def avg_rank(Z):
    return np.apply_along_axis(stats.rankdata, 0, Z)


def best_latent_auroc_ties(Z, S):
    y = S > 0
    Zr = avg_rank(Z) - 1
    n = Z.shape[0]
    out = []
    for i in range(S.shape[1]):
        pos = y[:, i]
        npos = pos.sum()
        if npos == 0 or npos == n:
            continue
        auc = (Zr[pos].sum(0) - npos * (npos - 1) / 2) / (npos * (n - npos))
        out.append(np.maximum(auc, 1 - auc).max())
    return float(np.mean(out))


def probe_auroc(Htr, Str, Hte, Ste, feats):
    out = []
    for i in feats:
        ytr, yte = Str[:, i] > 0, Ste[:, i] > 0
        if ytr.sum() < 5 or yte.sum() < 2:
            continue
        clf = LogisticRegression(C=10.0, max_iter=500).fit(Htr, ytr)
        out.append(roc_auc_score(yte, clf.decision_function(Hte)))
    return float(np.mean(out))


# ------------------------------- Experiment A -------------------------------
recA = []
for seed in SEEDS:
    gF = rng(1000 + seed)
    F = make_dict(gF)
    for k in KS:
        p = k / NF
        g = rng(2000 + 100 * seed + k)
        sampler = lambda b, g=g, F=F, p=p: sample_flat(g, F, p, b)
        Hte, Ste = sample_flat(rng(3000 + 100 * seed + k), F, p, N_TEST)
        sae = train_sae(rng(4000 + 100 * seed + k), sampler, WIDTH, LAM)
        rnd_dec = make_dict(rng(5000 + 100 * seed + k), WIDTH)
        sae_rnd = train_sae(rng(6000 + 100 * seed + k), sampler, WIDTH, LAM, frozen_dec=rnd_dec)
        Z = encode(sae, Hte)
        row = dict(seed=seed, k=k)
        row["mmcs"], row["recovered"] = recovery(F, sae["Wd"])
        row["l0"] = float((Z > 0).sum(1).mean())
        row["fvu"] = float(((encode(sae, Hte) @ sae["Wd"] + sae["bd"] - Hte) ** 2).sum() / ((Hte - Hte.mean(0)) ** 2).sum())
        row["auc_sae"] = best_latent_auroc_ties(Z, Ste)
        row["auc_sae_randdec"] = best_latent_auroc_ties(encode(sae_rnd, Hte), Ste)
        row["auc_raw"] = best_latent_auroc_ties(Hte, Ste)
        Htr_, Str_ = sample_flat(rng(7000 + 100 * seed + k), F, p, N_TEST)
        row["auc_pca"] = best_latent_auroc_ties(PCA(D).fit(Htr_).transform(Hte), Ste)
        row["auc_learned_dict_ista"] = best_latent_auroc_ties(ista(sae["Wd"], Hte - sae["bd"]), Ste)
        row["auc_oracle_ista"] = best_latent_auroc_ties(ista(F, Hte), Ste)
        row["auc_probe"] = probe_auroc(Htr_, Str_, Hte, Ste, range(0, NF, 4))   # 24 个特征的子样本
        recA.append(row)
    print(f"A seed {seed} {time.time()-T0:.0f}s", flush=True)

# ------------------------------- Experiment B -------------------------------
LAMS = [0.1, 0.3, 1.0]
recB = []
for seed in SEEDS:
    F = make_dict(rng(1000 + seed))
    Hte, Ste = sample_hier(rng(8000 + seed), F, N_TEST * 2)
    for lam in LAMS:
        g = rng(9000 + 10 * seed + int(100 * lam))
        sae = train_sae(rng(9500 + 10 * seed + int(100 * lam)), lambda b, g=g, F=F: sample_hier(g, F, b), WIDTH, lam)
        Z = encode(sae, Hte)
        C = np.abs(F @ sae["Wd"].T)
        absorb, absorb_ctrl, cos_par = [], [], []
        for i in range(8):
            j = int(C[i].argmax())                         # parent latent
            cos_par.append(C[i, j])
            par_on = Ste[:, i] > 0
            ch_idx = 8 + 3 * i + np.arange(3)
            ch_on = (Ste[:, ch_idx] > 0).any(1)
            with_child = par_on & ch_on
            without = par_on & ~ch_on
            if with_child.sum() > 5 and without.sum() > 5:
                # 吸收率：父特征激活且有子特征时，父 latent 不激活的比例，减去无子特征时的漏检率
                absorb.append(float((Z[with_child, j] <= 1e-6).mean()))
                absorb_ctrl.append(float((Z[without, j] <= 1e-6).mean()))
        # 子 latent 的解码方向里含有多少父方向（吸收的几何特征）
        par_in_child = []
        for i in range(8):
            for c in range(3):
                fi = 8 + 3 * i + c
                jc = int(C[fi].argmax())
                par_in_child.append(float(F[i] @ sae["Wd"][jc]))
        recB.append(dict(seed=seed, lam=lam, absorb=float(np.mean(absorb)), miss_no_child=float(np.mean(absorb_ctrl)),
                         absorb_excess=float(np.mean(absorb) - np.mean(absorb_ctrl)), parent_cos=float(np.mean(cos_par)),
                         parent_dir_in_child=float(np.mean(par_in_child)), l0=float((Z > 0).sum(1).mean()),
                         fvu=float(((Z @ sae["Wd"] + sae["bd"] - Hte) ** 2).sum() / ((Hte - Hte.mean(0)) ** 2).sum())))
    print(f"B seed {seed} {time.time()-T0:.0f}s", flush=True)

# ------------------------------- Experiment C -------------------------------
recC = []
for seed in SEEDS:
    F = make_dict(rng(1000 + seed))
    p = 4 / NF
    Hte, Ste = sample_flat(rng(3000 + 100 * seed + 4), F, p, N_TEST)
    for m in [48, 96]:
        g = rng(11000 + 10 * seed + m)
        sae = train_sae(rng(12000 + 10 * seed + m), lambda b, g=g, F=F: sample_flat(g, F, p, b), m, LAM)
        mm, rec = recovery(F, sae["Wd"])
        recC.append(dict(seed=seed, m=m, mmcs=mm, recovered=rec, auc_sae=best_latent_auroc_ties(encode(sae, Hte), Ste)))
    r192 = [r for r in recA if r["seed"] == seed and r["k"] == 4][0]
    recC.append(dict(seed=seed, m=192, mmcs=r192["mmcs"], recovered=r192["recovered"], auc_sae=r192["auc_sae"]))
print(f"C done {time.time()-T0:.0f}s", flush=True)


# ------------------------------- theory -------------------------------
def cs_threshold(k, n=NF, d=D):
    """压缩感知启发式：k 稀疏非负信号可由 d 维随机投影 + L1 恢复的近似条件 d ≳ 2k ln(n/k)。返回 d/(2k ln(n/k))。"""
    return d / (2 * k * np.log(n / k))


def ag(rows, key, **flt):
    v = [r[key] for r in rows if all(r[a] == b for a, b in flt.items())]
    m, h = mean_ci(v)
    return {"mean": m, "ci95": h, "n": len(v)}


out = {"config": dict(D=D, n_features=NF, seeds=SEEDS, ks=KS, lam=LAM, width=WIDTH, steps=STEPS, batch=BATCH,
                      lr=LR, n_test=N_TEST, cos_threshold=COS_THR, lams_absorption=LAMS,
                      hier="8 parents x 3 children; p_parent=0.06, p(child|parent)=0.3; 64 independent controls p=0.03")}
keysA = ["mmcs", "recovered", "l0", "fvu", "auc_sae", "auc_sae_randdec", "auc_raw", "auc_pca", "auc_learned_dict_ista",
         "auc_oracle_ista", "auc_probe"]
out["A_density"] = {f"k={k}": {**{kk: ag(recA, kk, k=k) for kk in keysA}, "cs_ratio_d_over_2klnnk": cs_threshold(k)} for k in KS}
T = {}
for k in KS:
    a = np.array([r["auc_sae"] for r in recA if r["k"] == k])
    for b in ["auc_sae_randdec", "auc_raw", "auc_pca", "auc_learned_dict_ista", "auc_oracle_ista", "auc_probe"]:
        x = np.array([r[b] for r in recA if r["k"] == k])
        T[f"k={k}|sae_minus_{b}"] = {"diff": float(np.mean(a - x)), "paired_t_p": float(stats.ttest_rel(a, x).pvalue)}
out["A_tests"] = T
out["B_absorption"] = {f"lam={l}": {kk: ag(recB, kk, lam=l) for kk in ["absorb", "miss_no_child", "absorb_excess", "parent_cos",
                                                                        "parent_dir_in_child", "l0", "fvu"]} for l in LAMS}
ab = [(r["lam"], r["absorb_excess"]) for r in recB]
sp = stats.spearmanr([a for a, _ in ab], [b for _, b in ab])
out["B_tests"] = {"spearman_lam_vs_absorb_excess": {"rho": float(sp.statistic), "p": float(sp.pvalue)},
                  "lam1_minus_lam0.1_absorb_excess": {
                      "diff": float(np.mean([r["absorb_excess"] for r in recB if r["lam"] == 1.0]) - np.mean([r["absorb_excess"] for r in recB if r["lam"] == 0.1])),
                      "paired_t_p": float(stats.ttest_rel([r["absorb_excess"] for r in recB if r["lam"] == 1.0],
                                                          [r["absorb_excess"] for r in recB if r["lam"] == 0.1]).pvalue)}}
for key in ["parent_dir_in_child", "absorb"]:
    sp2 = stats.spearmanr([r["lam"] for r in recB], [r[key] for r in recB])
    out["B_tests"][f"spearman_lam_vs_{key}"] = {"rho": float(sp2.statistic), "p": float(sp2.pvalue)}
out["per_seed"] = {"A": recA, "B": recB, "C": recC}
out["C_width"] = {f"m={m}": {kk: ag(recC, kk, m=m) for kk in ["mmcs", "recovered", "auc_sae"]} for m in [48, 96, 192]}
out["runtime_sec"] = time.time() - T0
out["_meta"] = {"seed": SEED, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
R = results_dir(__file__)
(R / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

# ---------------------------------- figures ----------------------------------
P = plt()
fig, axes = P.subplots(1, 2, figsize=(12, 4.2))
for kk, lab in [("recovered", f"SAE 恢复比例 (cos>{COS_THR})"), ("mmcs", "SAE 平均最大余弦")]:
    axes[0].errorbar(KS, [out["A_density"][f"k={k}"][kk]["mean"] for k in KS],
                     yerr=[out["A_density"][f"k={k}"][kk]["ci95"] for k in KS], marker="o", capsize=3, label=lab)
axes[0].plot(KS, [min(1, cs_threshold(k)) for k in KS], "k:", label="压缩感知比 d/(2k ln(n/k))（截断到 1）")
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("每个样本的期望激活特征数 k")
axes[0].set_title("A：特征恢复 vs 稀疏度（d=32, n=96）")
axes[0].legend(fontsize=8)
for kk, lab in [("auc_probe", "有监督探针（上限）"), ("auc_oracle_ista", "真字典+ISTA"), ("auc_learned_dict_ista", "SAE 字典+ISTA"),
                ("auc_sae", "SAE"), ("auc_sae_randdec", "随机冻结解码器 SAE"), ("auc_pca", "PCA"), ("auc_raw", "原始神经元")]:
    axes[1].errorbar(KS, [out["A_density"][f"k={k}"][kk]["mean"] for k in KS],
                     yerr=[out["A_density"][f"k={k}"][kk]["ci95"] for k in KS], marker="o", capsize=3, label=lab)
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("k")
axes[1].set_ylabel("单 latent 特征检测 AUROC（特征平均）")
axes[1].set_title("A：SAE vs 基线")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / "frontier_density.png")

fig, ax = P.subplots(figsize=(6, 4))
for kk, lab in [("absorb", "父 latent 漏检率（有子特征）"), ("miss_no_child", "父 latent 漏检率（无子特征）"),
                ("parent_dir_in_child", "子 latent 解码向量中父方向分量")]:
    ax.errorbar(LAMS, [out["B_absorption"][f"lam={l}"][kk]["mean"] for l in LAMS],
                yerr=[out["B_absorption"][f"lam={l}"][kk]["ci95"] for l in LAMS], marker="o", capsize=3, label=lab)
ax.set_xscale("log")
ax.set_xlabel("L1 系数 λ")
ax.set_title("B：层级特征的吸收 vs L1 强度")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / "frontier_absorption.png")
print(f"done {time.time()-T0:.0f}s")
