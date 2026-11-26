"""Q15 前沿实验：类 LLM 生成器的语义不确定性 + 共形弃权在分布偏移下的保证.

玩具生成模型（真值已知）：
  * 每个问题属于主题 z∈{0..4}（难度递增）；模型在 6 个"含义"上的分布 π：正确含义质量 c~Beta(均值 c_z)，
    其余质量按 Dirichlet 分给错误含义；以概率 mis 出现"误解"（正确含义与最大错误含义互换质量 → 自信地错）。
  * 每个含义有 L=1+Poisson(v) 种表述（paraphrase），表述分布 Dirichlet(1)；啰嗦度 v 与正确性无关。
  * 每题采样 M=10 次；回答 = 样本中出现最多的含义；幻觉 = 回答含义 ≠ 正确含义。
不确定性分数：语义熵 SE（按含义聚类；可加聚类错误 η：样本以概率 η 被误分成新簇）、
  表层/字符串熵（≈ token 熵）、自洽性 1-众数频率。
理论（§4）：链式法则 H(串) = H(含义) + Σ_j π_j H(串|含义 j)；第二项是与正确性无关的"表述噪声"。
共形弃权：用 SE 阈值 τ 决定回答/弃权，控制 P(回答且幻觉) ≤ α（共形风险控制 CRC, Angelopoulos et al.）。
偏移：S1 协变量偏移（主题分布变难，P(幻觉|SE,z) 不变）→ 加权 CRC；S2 概念偏移（误解率上升）→ 在线自适应（ACI 式）。
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
from sklearn.metrics import roc_auc_score

from common.util import SEED, mean_ci, plt, results_dir, rng

T0 = time.time()
SEEDS = list(range(5))
KM, M = 6, 10
C_MEAN = np.array([0.9, 0.8, 0.65, 0.5, 0.35])
P_SRC = np.array([0.35, 0.3, 0.2, 0.1, 0.05])
P_TGT = P_SRC[::-1].copy()
MIS_SRC, MIS_S2 = 0.05, 0.3
ALPHA = 0.1
N_CAL, N_TEST, REPS = 500, 2000, 100
VERB = {"const2": ("const", 2.0), "unif0-4": ("unif", 4.0), "unif0-10": ("unif", 10.0)}
ETAS = [0.0, 0.1, 0.3]
GAMMA, T_STREAM = 0.05, 4000


def ent(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def gen(g, n, pz, mis, verb=("unif", 4.0), eta=0.0):
    z = g.choice(5, n, p=pz)
    out = {k: np.zeros(n) for k in ["se", "str", "cons", "halluc", "se_exact", "str_exact", "nuis_exact"]}
    out["z"] = z
    for i in range(n):
        cm = C_MEAN[z[i]]
        c = g.beta(5 * cm, 5 * (1 - cm))
        wrong = (1 - c) * g.dirichlet(0.5 * np.ones(KM - 1))
        pi = np.r_[c, wrong]                                  # meaning 0 is correct
        if g.random() < mis:                                  # 误解：自信地错
            j = 1 + int(np.argmax(wrong))
            pi[0], pi[j] = pi[j], pi[0]
            if pi[0] > pi[j]:
                pi[0], pi[j] = pi[j], pi[0]
        v = verb[1] if verb[0] == "const" else g.uniform(0, verb[1])
        L = 1 + g.poisson(v, KM)
        para = [g.dirichlet(np.ones(l)) for l in L]
        mean_s = g.choice(KM, M, p=pi)
        strs = np.array([j * 100 + g.choice(L[j], p=para[j]) for j in mean_s])
        clus = mean_s.copy()
        if eta > 0:                                           # 等价判断失败 → 拆成新簇
            bad = g.random(M) < eta
            clus = np.where(bad, 1000 + np.arange(M), clus)
        cc = np.bincount(clus if eta == 0 else np.unique(clus, return_inverse=True)[1])
        out["se"][i] = ent(cc)
        out["str"][i] = ent(np.unique(strs, return_counts=True)[1].astype(float))
        mc = np.bincount(mean_s, minlength=KM)
        mode = g.choice(np.flatnonzero(mc == mc.max()))
        out["cons"][i] = 1 - mc.max() / M
        out["halluc"][i] = float(mode != 0)
        h_par = np.array([-(q * np.log(q)).sum() for q in para])
        out["se_exact"][i] = -(pi[pi > 0] * np.log(pi[pi > 0])).sum()
        out["nuis_exact"][i] = float(pi @ h_par)
        out["str_exact"][i] = out["se_exact"][i] + out["nuis_exact"][i]
    return out


# =========================== Part A: detection AUROC ===========================
A = []
for seed in SEEDS:
    for vn, verb in VERB.items():
        for eta in ETAS:
            if vn != "unif0-4" and eta > 0:
                continue
            d = gen(rng(1000 * seed + 10 * list(VERB).index(vn) + int(10 * eta)), 3000, P_SRC, MIS_SRC, verb, eta)
            y = d["halluc"]
            row = dict(seed=seed, verb=vn, eta=eta, halluc_rate=float(y.mean()))
            for k in ["se", "str", "cons", "se_exact", "str_exact"]:
                row[f"auc_{k}"] = float(roc_auc_score(y, d[k]))
            row["nuis_var_ratio"] = float(d["nuis_exact"].var() / d["se_exact"].var())
            row["chain_rule_max_abs_err"] = float(np.abs(d["str_exact"] - d["se_exact"] - d["nuis_exact"]).max())
            # 误解题上的检测能力（自信地错）
            A.append(row)
    print(f"A seed {seed} {time.time()-T0:.0f}s", flush=True)


# =========================== Part B: conformal abstention ===========================
def crc_threshold(s, l_wrong, alpha, w=None, w_test=None):
    """最大 τ 使 (Σ w_i·1{s_i≤τ, wrong_i} + w_test·1)/(Σ w_i + w_test) ≤ α（无权重时 w≡1）。"""
    if w is None:
        w = np.ones(len(s))
        w_test = 1.0
    order = np.argsort(s)
    s_sorted, l_sorted, w_sorted = s[order], l_wrong[order], w[order]
    cum = np.cumsum(w_sorted * l_sorted)
    denom = w.sum() + w_test
    ok = (cum + w_test) / denom <= alpha
    # τ 可取到第 i 个分数（含并列）；需要所有不超过该分数的样本都满足
    cand = -np.inf
    uniq = np.unique(s_sorted)
    last_idx = np.searchsorted(s_sorted, uniq, side="right") - 1
    for u, li in zip(uniq, last_idx):
        if ok[li]:
            cand = u
        else:
            break
    return cand


def risk(s, wrong, tau):
    ans = s <= tau
    return float(np.mean(ans & (wrong > 0))), float(1 - ans.mean())


B = []
for seed in SEEDS:
    g = rng(5000 + seed)
    src = gen(g, N_CAL * 4, P_SRC, MIS_SRC)
    tgt = {"S0": gen(g, N_TEST, P_SRC, MIS_SRC), "S1": gen(g, N_TEST, P_TGT, MIS_SRC), "S2": gen(g, N_TEST, P_SRC, MIS_S2)}
    ratio = P_TGT / P_SRC
    for rep in range(REPS):
        idx = g.choice(len(src["se"]), N_CAL, replace=False)
        s_cal, l_cal, z_cal = src["se"][idx], src["halluc"][idx], src["z"][idx]
        tau = crc_threshold(s_cal, l_cal, ALPHA)
        for sh, d in tgt.items():
            r_, ab = risk(d["se"], d["halluc"], tau)
            row = dict(seed=seed, rep=rep, shift=sh, method="static", risk=r_, abstain=ab)
            B.append(row)
        # weighted CRC for S1: per-topic threshold, weights = P_TGT/P_SRC
        w = ratio[z_cal]
        taus_z = [crc_threshold(s_cal, l_cal, ALPHA, w=w, w_test=ratio[zz]) for zz in range(5)]
        for sh in ["S1"]:        # S2 的主题分布与源相同，协变量权重恒为 1，加权 CRC 退化为静态 CRC
            d = tgt[sh]
            tz = np.array(taus_z)[d["z"]]
            ans = d["se"] <= tz
            B.append(dict(seed=seed, rep=rep, shift=sh, method="weighted", risk=float(np.mean(ans & (d["halluc"] > 0))), abstain=float(1 - ans.mean())))
        # oracle: CRC calibrated on target-distribution data (upper reference)
    print(f"B seed {seed} {time.time()-T0:.0f}s", flush=True)

# =========================== Part C: online ACI-style stream ===========================
C = []
for seed in SEEDS:
    g = rng(9000 + seed)
    src = gen(g, N_CAL, P_SRC, MIS_SRC)
    tau0 = crc_threshold(src["se"], src["halluc"], ALPHA)
    tau0 = 0.0 if not np.isfinite(tau0) else tau0
    for sh, (pz, mis) in {"S1": (P_TGT, MIS_SRC), "S2": (P_SRC, MIS_S2)}.items():
        d = gen(g, T_STREAM, pz, mis)
        tau = tau0
        losses, abst = [], []
        for t in range(T_STREAM):
            ans = d["se"][t] <= tau
            l = float(ans and d["halluc"][t] > 0)
            losses.append(l)
            abst.append(float(not ans))
            tau = tau + GAMMA * (ALPHA - l)                     # 在线更新（Gibbs & Candès 2021 的风险版本）
            tau = min(max(tau, -GAMMA), np.log(M) + GAMMA)
        static = [float(d["se"][t] <= tau0 and d["halluc"][t] > 0) for t in range(T_STREAM)]
        C.append(dict(seed=seed, shift=sh, risk_aci=float(np.mean(losses)), abstain_aci=float(np.mean(abst)),
                      risk_static=float(np.mean(static)), abstain_static=float(np.mean(d["se"] > tau0)),
                      risk_aci_last_half=float(np.mean(losses[T_STREAM // 2:]))))
print(f"C {time.time()-T0:.0f}s", flush=True)

ACI_BOUND = (np.log(M) + 2 * GAMMA) / (GAMMA * T_STREAM)


def ag(rows, key, **flt):
    v = [r[key] for r in rows if all(r[a] == b for a, b in flt.items())]
    m_, h = mean_ci(v)
    return {"mean": m_, "ci95": h, "n": len(v)}


out = {"config": dict(seeds=SEEDS, meanings=KM, samples_M=M, c_mean_by_topic=C_MEAN.tolist(), p_src=P_SRC.tolist(),
                      p_tgt=P_TGT.tolist(), mis_src=MIS_SRC, mis_s2=MIS_S2, alpha=ALPHA, n_cal=N_CAL, n_test=N_TEST,
                      reps=REPS, verbosity=list(VERB), etas=ETAS, gamma=GAMMA, t_stream=T_STREAM, aci_bound=ACI_BOUND)}
out["A_detection"] = {}
for vn in VERB:
    for eta in ETAS:
        rows = [r for r in A if r["verb"] == vn and r["eta"] == eta]
        if not rows:
            continue
        out["A_detection"][f"{vn}|eta={eta}"] = {k: ag(rows, k) for k in ["halluc_rate", "auc_se", "auc_str", "auc_cons", "auc_se_exact",
                                                                         "auc_str_exact", "nuis_var_ratio", "chain_rule_max_abs_err"]}
TA = {}
for vn in VERB:
    rows = [r for r in A if r["verb"] == vn and r["eta"] == 0.0]
    a, b = np.array([r["auc_se"] for r in rows]), np.array([r["auc_str"] for r in rows])
    TA[f"{vn}|se_minus_str"] = {"diff": float(np.mean(a - b)), "paired_t_p": float(stats.ttest_rel(a, b).pvalue)}
rows0 = [r for r in A if r["verb"] == "unif0-4"]
for eta in ETAS[1:]:
    a = np.array([r["auc_se"] for r in rows0 if r["eta"] == 0.0])
    b = np.array([r["auc_se"] for r in rows0 if r["eta"] == eta])
    TA[f"unif0-4|se_eta0_minus_eta{eta}"] = {"diff": float(np.mean(a - b)), "paired_t_p": float(stats.ttest_rel(a, b).pvalue)}
out["A_tests"] = TA
out["B_conformal"] = {}
for sh in ["S0", "S1", "S2"]:
    for mth in ["static", "weighted"]:
        rows = [r for r in B if r["shift"] == sh and r["method"] == mth]
        if rows:
            # 每个种子先平均 REPS 次，再跨种子求 CI
            per = [np.mean([r["risk"] for r in rows if r["seed"] == s]) for s in SEEDS]
            pab = [np.mean([r["abstain"] for r in rows if r["seed"] == s]) for s in SEEDS]
            viol = [np.mean([r["risk"] > ALPHA for r in rows if r["seed"] == s]) for s in SEEDS]
            out["B_conformal"][f"{sh}|{mth}"] = {"risk": dict(zip(["mean", "ci95"], mean_ci(per))),
                                                 "abstain": dict(zip(["mean", "ci95"], mean_ci(pab))),
                                                 "frac_reps_risk_above_alpha": dict(zip(["mean", "ci95"], mean_ci(viol)))}
out["C_online"] = {sh: {k: ag(C, k, shift=sh) for k in ["risk_aci", "abstain_aci", "risk_static", "abstain_static", "risk_aci_last_half"]}
                   for sh in ["S1", "S2"]}
out["runtime_sec"] = time.time() - T0
out["_meta"] = {"seed": SEED, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
R = results_dir(__file__)
(R / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

# ---------------------------------- figures ----------------------------------
P = plt()
fig, axes = P.subplots(1, 2, figsize=(12, 4.2))
xs = np.arange(len(VERB))
for i, (k, lab) in enumerate([("auc_se", "语义熵 SE"), ("auc_str", "字符串熵"), ("auc_cons", "自洽性 1-众数频率")]):
    axes[0].bar(xs + (i - 1) * 0.25, [out["A_detection"][f"{vn}|eta=0.0"][k]["mean"] for vn in VERB], 0.25,
                yerr=[out["A_detection"][f"{vn}|eta=0.0"][k]["ci95"] for vn in VERB], label=lab, capsize=3)
axes[0].set_xticks(xs, ["啰嗦度恒定", "啰嗦度 U(0,4)", "啰嗦度 U(0,10)"])
axes[0].set_ylim(0.5, 0.9)
axes[0].set_ylabel("幻觉检测 AUROC")
axes[0].set_title("表述多样性是与正确性无关的噪声")
axes[0].legend(fontsize=8)
labels, risks, errs = [], [], []
for sh in ["S0", "S1", "S2"]:
    for mth in ["static", "weighted"]:
        if f"{sh}|{mth}" in out["B_conformal"]:
            labels.append(f"{sh}\n{'静态' if mth == 'static' else '加权'}")
            risks.append(out["B_conformal"][f"{sh}|{mth}"]["risk"]["mean"])
            errs.append(out["B_conformal"][f"{sh}|{mth}"]["risk"]["ci95"])
for sh in ["S1", "S2"]:
    labels.append(f"{sh}\n在线")
    risks.append(out["C_online"][sh]["risk_aci"]["mean"])
    errs.append(out["C_online"][sh]["risk_aci"]["ci95"])
axes[1].bar(range(len(labels)), risks, yerr=errs, capsize=3, color=["C0", "C1", "C1", "C2", "C1", "C2"][:len(labels)])
axes[1].axhline(ALPHA, color="k", ls="--", label="目标 α")
axes[1].set_xticks(range(len(labels)), labels, fontsize=8)
axes[1].set_ylabel("P(回答且幻觉)")
axes[1].set_title("共形弃权：S0 无偏移 / S1 协变量偏移 / S2 概念偏移")
axes[1].legend()
fig.tight_layout()
fig.savefig(R / "frontier_semantic_conformal.png")
print(f"done {time.time()-T0:.0f}s")
