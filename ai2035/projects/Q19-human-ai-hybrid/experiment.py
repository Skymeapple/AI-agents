"""Q19 人机混合增强智能：人–AI 互补性的仿真研究。

设计（参数在运行前固定）：
- 数据：sklearn digits；每种子分层划分 训练 50% / 验证 25% / 测试 25%。
- AI：PCA(8) + 多项 Logistic 回归（刻意保持中等精度，约 0.9）。
  置信度 c = max softmax；“校准”= 在验证集上拟合温度 T*（最小化 NLL）。
  显示给人的置信度四种：cal（T*）、over（logits×3/T*，过度自信）、under（logits×(1/3)/T*，信心不足）、
  noisy（校准后 logit 加 N(0,1.5²) 噪声，破坏排序）。
- 模拟人类：潜变量 z_h = ρ·z_ai + √(1-ρ²)·ε，z_ai = Φ^{-1}(AI 对真类概率的秩)（AI 越难的样本 z_ai 越小）。
  人在类别 k 上的目标准确率 acc_h ± δ（δ=0.08，偶数类 +δ、奇数类 −δ，表示可学习的专长差异）；
  z_h 高于该类分位阈值则答对，否则在 9 个错误类中均匀随机选。分类准确率截断于 0.995，
  故名义 acc_h=0.95/0.98 的实际准确率较低；报告与作图均使用实际准确率。
- 网格：ρ ∈ {0, 0.3, 0.6, 0.9}，acc_h ∈ {0.70, 0.80, 0.85, 0.90, 0.95, 0.98}。
- 策略：
  AI 单独；人单独；
  建议模式（人看到 AI 答案与显示置信度，二者一致则采纳；分歧时）：
    over  = 以 0.9 概率跟随 AI；under = 以 0.1 概率跟随 AI；
    conf  = 当显示置信度 > 人对自身准确率的信念（= 人在验证集上的真实边际准确率）时跟随
            （依赖“绝对数值”，对校准敏感）；
    opt   = 分歧时跟随 AI 的置信度阈值在验证集上调优（只依赖排序，“理想适度依赖”）。
  延迟模式（AI 决定由谁回答，被延迟的样本人独立作答）：
    defer_conf = 置信度低于阈值则交给人，阈值在验证集上调优；
    defer_l2d  = 两阶段 learning-to-defer：Logistic 回归用 [PCA 特征, AI 预测类 one-hot, 置信度]
                 预测“人是否答对”，若 P̂(人对) > 校准后 AI 置信度则延迟。
- 指标：团队准确率、互补增益 = 团队 − max(AI, 人)；5 个种子 × 每种子 10 次人类抽样。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng, save_results  # noqa: E402

from scipy import stats  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import softmax  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
DRAWS = 10
RHOS = [0.0, 0.3, 0.6, 0.9]
ACC_H = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
DELTA = 0.08
CALS = ["cal", "over", "under", "noisy"]
POLICIES = ["ai", "human", "over", "under", "conf", "opt", "defer_conf", "defer_l2d"]
CAL_POLICIES = ["conf", "opt", "defer_conf", "defer_l2d"]


def ece(conf, correct, bins=10):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def simulate_human(y, z_ai, rho, acc_h, g):
    z = rho * z_ai + np.sqrt(1 - rho ** 2) * g.standard_normal(len(y))
    acc_k = np.where(y % 2 == 0, acc_h + DELTA, acc_h - DELTA).clip(0.02, 0.995)
    # correct iff z > Φ^{-1}(1 - acc_k) (z is marginally N(0,1))
    correct = z > stats.norm.ppf(1 - acc_k)
    wrong = (y + g.integers(1, 10, len(y))) % 10
    return np.where(correct, y, wrong)


def best_threshold(score, gain_if_ai):
    """choose threshold t maximising sum(gain_if_ai[score>t]) (use AI when score > t)."""
    order = np.argsort(-score)
    cs = np.concatenate([[0], np.cumsum(gain_if_ai[order])])
    k = int(np.argmax(cs))
    s_sorted = score[order]
    if k == 0:
        return np.inf
    if k == len(score):
        return -np.inf
    return float((s_sorted[k - 1] + s_sorted[k]) / 2)


X, y = load_digits(return_X_y=True)
res = {"config": dict(seeds=SEEDS, draws=DRAWS, rhos=RHOS, acc_h=ACC_H, delta=DELTA, cals=CALS, policies=POLICIES)}
# acc[pol][cal][rho][acch] -> list over seeds ; for non-cal policies cal index ignored (store under "cal")
store = {p: {c: {r: {a: [] for a in ACC_H} for r in RHOS} for c in CALS} for p in POLICIES}
gain = {p: {c: {r: {a: [] for a in ACC_H} for r in RHOS} for c in CALS} for p in POLICIES}
defer_rate = {p: {c: {r: {a: [] for a in ACC_H} for r in RHOS} for c in CALS} for p in ["defer_conf", "defer_l2d"]}
err_phi = {r: [] for r in RHOS}
ai_accs, ece_store = [], {c: [] for c in CALS}

for sd in SEEDS:
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, stratify=y, random_state=sd)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=sd)
    pca = PCA(8, random_state=sd).fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(pca.transform(Xtr), ytr)
    Zva, Zte = pca.transform(Xva), pca.transform(Xte)
    Lva, Lte = clf.decision_function(Zva), clf.decision_function(Zte)
    Tstar = minimize_scalar(lambda T: -np.log(softmax(Lva / T, 1)[np.arange(len(yva)), yva] + 1e-12).mean(),
                            bounds=(0.05, 20), method="bounded").x
    pred_va, pred_te = Lva.argmax(1), Lte.argmax(1)
    ai_c_va, ai_c_te = pred_va == yva, pred_te == yte
    ai_accs.append(float(ai_c_te.mean()))
    g0 = rng(70 + sd)
    conf = {}
    for split, L in (("va", Lva), ("te", Lte)):
        cal = softmax(L / Tstar, 1).max(1)
        noisy_logit = np.log(cal / (1 - cal + 1e-12)) + 1.5 * g0.standard_normal(len(cal))
        conf[split] = {"cal": cal, "over": softmax(3 * L / Tstar, 1).max(1),
                       "under": softmax(L / (3 * Tstar), 1).max(1), "noisy": 1 / (1 + np.exp(-noisy_logit))}
    for c in CALS:
        ece_store[c].append(ece(conf["te"][c], ai_c_te))
    # AI difficulty latent: rank of AI prob of the true class
    def zai(L, yy):
        p = softmax(L / Tstar, 1)[np.arange(len(yy)), yy]
        return stats.norm.ppf((stats.rankdata(p) - 0.5) / len(p))
    zva, zte = zai(Lva, yva), zai(Lte, yte)
    Fva = np.hstack([Zva / 10, np.eye(10)[pred_va], conf["va"]["cal"][:, None]])
    Fte = np.hstack([Zte / 10, np.eye(10)[pred_te], conf["te"]["cal"][:, None]])
    for rho in RHOS:
        for acch in ACC_H:
            acc_d = {p: {c: [] for c in CALS} for p in POLICIES}
            dr = {p: {c: [] for c in CALS} for p in ["defer_conf", "defer_l2d"]}
            for d in range(DRAWS):
                g = rng(10000 * sd + 100 * d + int(100 * rho) * 7 + int(1000 * acch))
                hva = simulate_human(yva, zva, rho, acch, g)
                hte = simulate_human(yte, zte, rho, acch, g)
                h_c_va, h_c_te = hva == yva, hte == yte
                if acch == 0.85:
                    err_phi[rho].append(float(np.corrcoef(~ai_c_te, ~h_c_te)[0, 1]))
                agree_te = hte == pred_te
                dis_va = hva != pred_va
                acc_d["ai"]["cal"].append(ai_c_te.mean()); acc_d["human"]["cal"].append(h_c_te.mean())
                u = g.random(len(yte))
                for pol, pf in (("over", 0.9), ("under", 0.1)):
                    follow = u < pf
                    team = np.where(agree_te | follow, pred_te == yte, h_c_te)
                    acc_d[pol]["cal"].append(team.mean())
                # human learns their own reliability only through (un)biased belief acc_h
                l2d = LogisticRegression(max_iter=1000).fit(Fva, h_c_va.astype(int)) if 0 < h_c_va.mean() < 1 else None
                ph_te = l2d.predict_proba(Fte)[:, 1] if l2d is not None else np.full(len(yte), h_c_va.mean())
                for c in CALS:
                    cva, cte = conf["va"][c], conf["te"][c]
                    follow = cte > h_c_va.mean()  # human's belief = own (validation) accuracy
                    team = np.where(agree_te | follow, ai_c_te, h_c_te)
                    acc_d["conf"][c].append(team.mean())
                    # opt: threshold on displayed conf for following AI under disagreement, tuned on val
                    t = best_threshold(cva[dis_va], ai_c_va[dis_va].astype(float) - h_c_va[dis_va])
                    team = np.where(agree_te | (cte > t), ai_c_te, h_c_te)
                    acc_d["opt"][c].append(team.mean())
                    # deferral by confidence threshold tuned on val
                    t = best_threshold(cva, ai_c_va.astype(float) - h_c_va)
                    use_ai = cte > t
                    acc_d["defer_conf"][c].append(np.where(use_ai, ai_c_te, h_c_te).mean())
                    dr["defer_conf"][c].append(1 - use_ai.mean())
                    # two-stage L2D: compare P(human correct|x) with displayed AI confidence
                    use_ai = cte >= ph_te
                    acc_d["defer_l2d"][c].append(np.where(use_ai, ai_c_te, h_c_te).mean())
                    dr["defer_l2d"][c].append(1 - use_ai.mean())
            base = max(np.mean(acc_d["ai"]["cal"]), np.mean(acc_d["human"]["cal"]))
            for p in POLICIES:
                for c in (CALS if p in CAL_POLICIES else ["cal"]):
                    store[p][c][rho][acch].append(float(np.mean(acc_d[p][c])))
                    gain[p][c][rho][acch].append(float(np.mean(acc_d[p][c]) - base))
            for p in dr:
                for c in CALS:
                    defer_rate[p][c][rho][acch].append(float(np.mean(dr[p][c])))
    print(f"seed {sd} done {time.time()-T0:.1f}s", flush=True)

res["ai_acc"] = mean_ci(ai_accs)
res["ece"] = {c: mean_ci(v) for c, v in ece_store.items()}
res["error_phi_corr_at_acch_0.85"] = {str(r): mean_ci(v) for r, v in err_phi.items()}
res["human_acc_realized"] = {str(a): mean_ci([np.mean([store["human"]["cal"][r][a][s] for r in RHOS]) for s in range(len(SEEDS))]) for a in ACC_H}
res["acc"], res["gain"], res["gain_sig"] = {}, {}, {}
# reliance extremes vs the corresponding single agent (pp)
res["over_minus_ai_pp"] = {str(r): {str(a): mean_ci(100 * (np.array(store["over"]["cal"][r][a]) - np.array(store["ai"]["cal"][r][a]))) for a in ACC_H} for r in RHOS}
res["under_minus_human_pp"] = {str(r): {str(a): mean_ci(100 * (np.array(store["under"]["cal"][r][a]) - np.array(store["human"]["cal"][r][a]))) for a in ACC_H} for r in RHOS}
for p in POLICIES:
    res["acc"][p], res["gain"][p], res["gain_sig"][p] = {}, {}, {}
    for c in (CALS if p in CAL_POLICIES else ["cal"]):
        res["acc"][p][c] = {str(r): {str(a): mean_ci(store[p][c][r][a]) for a in ACC_H} for r in RHOS}
        res["gain"][p][c] = {str(r): {str(a): mean_ci(gain[p][c][r][a]) for a in ACC_H} for r in RHOS}
        res["gain_sig"][p][c] = {str(r): {str(a): bool(mean_ci(gain[p][c][r][a])[0] - mean_ci(gain[p][c][r][a])[1] > 0) for a in ACC_H} for r in RHOS}
res["defer_rate"] = {p: {c: {str(r): {str(a): mean_ci(defer_rate[p][c][r][a]) for a in ACC_H} for r in RHOS} for c in CALS} for p in defer_rate}

# value of calibration: cal minus miscalibrated, averaged over grid cells (per seed)
voc = {}
for p in CAL_POLICIES:
    voc[p] = {}
    for c in ["over", "under", "noisy"]:
        per_seed = np.array([[store[p]["cal"][r][a][s] - store[p][c][r][a][s] for r in RHOS for a in ACC_H] for s in range(len(SEEDS))])
        cell = per_seed.mean(0)
        voc[p][c] = {"mean_over_grid_pp": mean_ci(100 * per_seed.mean(1)), "max_cell_pp": float(100 * cell.max()),
                     "min_cell_pp": float(100 * cell.min())}
res["value_of_calibration"] = voc
# best policy per cell (calibrated)
best = {}
for r in RHOS:
    for a in ACC_H:
        cand = {p: np.mean(store[p]["cal"][r][a]) for p in POLICIES}
        bp = max(cand, key=cand.get)
        best[f"{r}|{a}"] = {"policy": bp, "gain": mean_ci(gain[bp]["cal"][r][a])}
res["best_policy"] = best
res["runtime_s"] = time.time() - T0

# ---------------------------------------------------------------- figures
P = plt()
od = results_dir(__file__)
show = ["over", "under", "conf", "opt", "defer_conf", "defer_l2d"]
fig, ax = P.subplots(1, len(show), figsize=(3.2 * len(show), 3.4), sharey=True)
gap_labels = [f"{res['human_acc_realized'][str(a)][0]:.2f}" for a in ACC_H]
for i, p in enumerate(show):
    M = np.array([[res["gain"][p]["cal"][str(r)][str(a)][0] for a in ACC_H] for r in RHOS]) * 100
    im = ax[i].imshow(M, cmap="RdBu", vmin=-10, vmax=10, aspect="auto", origin="lower")
    for (ri, ai), v in np.ndenumerate(M):
        sig = res["gain_sig"][p]["cal"][str(RHOS[ri])][str(ACC_H[ai])]
        ax[i].text(ai, ri, f"{v:.1f}{'*' if sig else ''}", ha="center", va="center", fontsize=6)
    ax[i].set_xticks(range(len(ACC_H))); ax[i].set_xticklabels(gap_labels, fontsize=7, rotation=45)
    ax[i].set_yticks(range(len(RHOS))); ax[i].set_yticklabels(RHOS)
    ax[i].set_title(p, fontsize=9); ax[i].set_xlabel("人类实际准确率")
ax[0].set_ylabel("误差相关 ρ")
fig.colorbar(im, ax=ax, shrink=0.8, label="互补增益 (pp)")
fig.suptitle(f"团队 − max(人, AI)（AI≈{res['ai_acc'][0]:.3f}，校准置信度；*=95%CI>0）", fontsize=10)
fig.savefig(od / "complementarity_map.png", bbox_inches="tight"); P.close(fig)

fig, ax = P.subplots(1, 2, figsize=(11, 4))
HX = [res["human_acc_realized"][str(a)][0] for a in ACC_H]
for c, mk in zip(CALS, "osd^"):
    ax[0].plot(HX, [res["acc"]["conf"][c]["0.3"][str(a)][0] for a in ACC_H], marker=mk, label=f"conf 依赖, {c}")
    ax[1].plot(HX, [res["acc"]["defer_conf"][c]["0.3"][str(a)][0] for a in ACC_H], marker=mk, label=f"阈值延迟, {c}")
for k in range(2):
    ax[k].plot(HX, [res["acc"]["ai"]["cal"]["0.3"][str(a)][0] for a in ACC_H], "k--", label="AI 单独")
    ax[k].plot(HX, [res["acc"]["human"]["cal"]["0.3"][str(a)][0] for a in ACC_H], "k:", label="人单独")
    ax[k].set_xlabel("人类实际准确率"); ax[k].set_ylabel("团队准确率 (ρ=0.3)"); ax[k].legend(fontsize=7)
ax[0].set_title("基于绝对置信度的依赖：对校准敏感"); ax[1].set_title("验证集调阈值的延迟：对单调失准不变")
fig.tight_layout(); fig.savefig(od / "calibration.png"); P.close(fig)

save_results(__file__, res)
print(f"done in {time.time()-T0:.1f}s")
