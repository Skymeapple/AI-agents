"""Q19 前沿实验：依赖的动态——人通过反馈学习何时信任 AI；解释的说服偏置、挤出效应与技能退化。

问题（2025–2026 争论）：解释提高信任却挤出人类独有知识 (arXiv:2511.04050)；从信任到“适度依赖”的度量
(2604.23896)；心智模型的发展 (2510.08104)；专家漂移/疲劳下的延迟 (2611.26440, 2604.00904, 2606.08919)；
噪声反馈下的自我评估 (2610.34737)。

设计（参数在运行前固定）：
- AI：digits 上 PCA(8)+Logistic（同第一阶段），温度缩放校准；置信度分 5 档 [0,.6,.75,.9,.97,1]。
- 模拟人：答对 ⇔ ρ·z_ai + √(1-ρ²)·ε > Φ^{-1}(1-p_h)，ρ=0.3，p_h=0.85；答错时随机选错误类。
  每个“人”看到 T=500 个随机抽取的测试样本（流式），先给出独立答案；与 AI 一致则采纳；
  分歧时：当 E[q_c] + b·1[解释看似合理] > E[h_c] 时跟随 AI；q_c、h_c 是“分歧时 AI 对 / 自己对”的
  Beta 后验，按线索单元 c（置信档 × 解释信号）维护；先验均值：q=m0、h=1-m0，强度 4。
- 反馈：full（每次揭示真值）；noisy（真值以 0.2 概率被替换为随机错误类）；
  none（无反馈，“自我确认”启发式：分歧时假定自己对）。
- 解释：none；persuasive（P(合理|AI对)=0.9, P(合理|AI错)=0.8，说服偏置 b=0.15）；
  diagnostic（0.9 / 0.3，b=0）；persuasive+anchor（同 persuasive，且以 γ=0.3 概率人不独立思考而直接
  采用 AI 答案 = 挤出效应）。
- 先验信任 m0 ∈ {0.3 怀疑, 0.8 过度信任}。
- 技能退化：full 反馈、无解释，T=3000；人的准确率 p_h(t) = 0.85 − δ·r̄_t，r̄_t 为“分歧时跟随 AI”
  的指数滑动平均（时间常数 200）；δ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}。理论：不动点 p = 0.85 − δ·F(p)，
  F(p) 为准确率 p 的人在贝叶斯最优依赖下的跟随率（由静态蒙特卡洛估计）。
- 5 个种子（AI 数据划分）× 每条件 200 个模拟人。
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
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import softmax  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
NH, T, RHO, P0 = 200, 500, 0.3, 0.85
BINS = np.array([0.0, 0.6, 0.75, 0.9, 0.97, 1.0001])
FEEDBACK = ["full", "noisy", "none"]
PRIORS = [0.3, 0.8]
EXPL = {"none": None, "persuasive": (0.9, 0.8, 0.15, 0.0), "diagnostic": (0.9, 0.3, 0.0, 0.0),
        "persuasive+anchor": (0.9, 0.8, 0.15, 0.3)}
DELTAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
T_LONG, TAU = 3000, 200.0


def build_ai(sd):
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=sd)
    pca = PCA(8, random_state=sd).fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(pca.transform(Xtr), ytr)
    L = clf.decision_function(pca.transform(Xte))
    Tst = minimize_scalar(lambda t: -np.log(softmax(L / t, 1)[np.arange(len(yte)), yte] + 1e-12).mean(), bounds=(0.05, 20), method="bounded").x
    P = softmax(L / Tst, 1)
    pred = P.argmax(1); conf = P.max(1)
    ptrue = P[np.arange(len(yte)), yte]
    z = stats.norm.ppf((stats.rankdata(ptrue) - 0.5) / len(ptrue))
    return dict(y=yte, pred=pred, conf=conf, cbin=np.digitize(conf, BINS) - 1, z=z, acc=float((pred == yte).mean()))


def human_answer(g, y, z, p_h):
    s = RHO * z + np.sqrt(1 - RHO ** 2) * g.standard_normal(len(y))
    corr = s > stats.norm.ppf(1 - p_h)
    return np.where(corr, y, (y + g.integers(1, 10, len(y))) % 10)


def run(ai, g, fb, m0, expl, Tn=T, delta=None):
    n_items = len(ai["y"])
    ncell = 5 * (2 if expl else 1)
    aq = np.full((NH, ncell), 4 * m0); bq = np.full((NH, ncell), 4 * (1 - m0))
    ah = np.full((NH, ncell), 4 * (1 - m0)); bh = np.full((NH, ncell), 4 * m0)
    hidx = np.arange(NH)
    team, hum, aic, follow_dis, over, under, dis_rate = [], [], [], [], [], [], []
    p_h = np.full(NH, P0); rbar = np.zeros(NH)
    for t in range(Tn):
        it = g.integers(0, n_items, NH)
        y, a, cb, z = ai["y"][it], ai["pred"][it], ai["cbin"][it], ai["z"][it]
        a_ok = a == y
        h = human_answer(g, y, z, p_h)
        h_indep_ok = h == y
        cell = cb.copy(); plaus = np.zeros(NH, bool); b = 0.0
        if expl:
            p_c, p_w, b, gam = expl
            plaus = g.random(NH) < np.where(a_ok, p_c, p_w)
            cell = cb * 2 + plaus
            if gam > 0:
                h = np.where(g.random(NH) < gam, a, h)  # crowding out: adopt AI answer without own judgement
        dis = h != a
        qm = aq[hidx, cell] / (aq[hidx, cell] + bq[hidx, cell]); hm = ah[hidx, cell] / (ah[hidx, cell] + bh[hidx, cell])
        fol = qm + b * plaus > hm
        final = np.where(dis & fol, a, h)
        team.append((final == y).mean()); hum.append(h_indep_ok.mean()); aic.append(a_ok.mean())
        dis_rate.append(dis.mean())
        if dis.any():
            follow_dis.append(fol[dis].mean())
            over.append((dis & fol & ~a_ok & (h == y)).sum() / max(1, (dis & ~a_ok & (h == y)).sum()))
            under.append((dis & ~fol & a_ok).sum() / max(1, (dis & a_ok).sum()))
        # feedback & learning on disagreement trials
        if fb == "full":
            lab = y
        elif fb == "noisy":
            lab = np.where(g.random(NH) < 0.2, (y + g.integers(1, 10, NH)) % 10, y)
        if fb in ("full", "noisy"):
            aok, hok = (a == lab), (h == lab)
        else:  # self-confirmation: assume own answer correct
            aok, hok = np.zeros(NH, bool), np.ones(NH, bool)
        upd = dis
        aq[hidx[upd], cell[upd]] += aok[upd]; bq[hidx[upd], cell[upd]] += ~aok[upd]
        ah[hidx[upd], cell[upd]] += hok[upd]; bh[hidx[upd], cell[upd]] += ~hok[upd]
        if delta is not None:
            rbar = np.where(dis, rbar + (fol.astype(float) - rbar) / TAU, rbar)
            p_h = np.clip(P0 - delta * rbar, 0.05, 0.99)
    return dict(team=np.array(team), hum=np.array(hum), ai=np.array(aic), follow=np.array(follow_dis),
                over=np.array(over), under=np.array(under), dis=np.array(dis_rate), p_h_final=float(p_h.mean()))


def static_follow_rate(ai, g, p_h, n=20000):
    """Bayes-optimal (oracle) reliance for a human of accuracy p_h: follow iff q_c > h_c (estimated by MC)."""
    it = g.integers(0, len(ai["y"]), n)
    y, a, cb, z = ai["y"][it], ai["pred"][it], ai["cbin"][it], ai["z"][it]
    h = human_answer(g, y, z, p_h); dis = h != a
    F, team = 0.0, 0.0
    fol_all = np.zeros(n, bool)
    for c in range(5):
        m = dis & (cb == c)
        if m.sum() == 0:
            continue
        q = (a[m] == y[m]).mean(); hh = (h[m] == y[m]).mean()
        fol_all[m] = q > hh
    F = fol_all[dis].mean() if dis.any() else 0.0
    final = np.where(dis & fol_all, a, h)
    return float(F), float((final == y).mean()), float((h == y).mean())


out = {"config": dict(seeds=SEEDS, n_humans=NH, T=T, rho=RHO, p0=P0, bins=BINS.tolist(), expl=EXPL, deltas=DELTAS, T_long=T_LONG, tau=TAU)}
grid = {}
ai_acc = []
for sd in SEEDS:
    ai = build_ai(sd); ai_acc.append(ai["acc"])
    for fb in FEEDBACK:
        for m0 in PRIORS:
            for en, ex in EXPL.items():
                r = run(ai, rng(1000 * sd + 7 * FEEDBACK.index(fb) + int(10 * m0) + 100 * list(EXPL).index(en)), fb, m0, ex)
                grid.setdefault((fb, m0, en), []).append(r)
    print(f"seed {sd} grid {time.time()-T0:.1f}s", flush=True)
out["ai_acc"] = mean_ci(ai_acc)
summ = {}
for (fb, m0, en), lst in grid.items():
    first = np.array([r["team"][:100].mean() for r in lst]); last = np.array([r["team"][-100:].mean() for r in lst])
    hum = np.array([r["hum"].mean() for r in lst]); aia = np.array([r["ai"].mean() for r in lst])
    base = np.maximum(hum, aia)
    summ[f"{fb}|{m0}|{en}"] = {
        "team_first100": mean_ci(first), "team_last100": mean_ci(last),
        "human_alone": mean_ci(hum), "ai_alone": mean_ci(aia),
        "gain_last100_pp": mean_ci(100 * (last - base)),
        "follow_rate_last": mean_ci([r["follow"][-100:].mean() for r in lst]),
        "overreliance_last": mean_ci([r["over"][-100:].mean() for r in lst]),
        "underreliance_last": mean_ci([r["under"][-100:].mean() for r in lst]),
        "disagree_rate": mean_ci([r["dis"].mean() for r in lst]),
        "curve": np.mean([np.convolve(r["team"], np.ones(25) / 25, mode="valid") for r in lst], 0)[::5].tolist(),
    }
    # time to complementarity: first window (25-trial moving avg, mean over seeds) exceeding base
    curve = np.mean([np.convolve(r["team"], np.ones(50) / 50, mode="valid") for r in lst], 0)
    above = np.nonzero(curve > base.mean())[0]
    summ[f"{fb}|{m0}|{en}"]["t_complementarity"] = int(above[0] + 50) if len(above) else None
out["grid"] = summ


def pdiff(k1, k2, field="team_last100"):
    a = np.array([r["team"][-100:].mean() for r in grid[k1]]); b = np.array([r["team"][-100:].mean() for r in grid[k2]])
    return {"diff_pp": mean_ci(100 * (a - b)), "p": float(stats.ttest_rel(a, b).pvalue)}


out["tests"] = {
    "persuasive_vs_none_full_m0.3": pdiff(("full", 0.3, "persuasive"), ("full", 0.3, "none")),
    "persuasive_vs_none_full_m0.8": pdiff(("full", 0.8, "persuasive"), ("full", 0.8, "none")),
    "diagnostic_vs_none_full_m0.3": pdiff(("full", 0.3, "diagnostic"), ("full", 0.3, "none")),
    "anchor_vs_persuasive_full_m0.3": pdiff(("full", 0.3, "persuasive+anchor"), ("full", 0.3, "persuasive")),
    "full_vs_none_fb_m0.8": pdiff(("full", 0.8, "none"), ("none", 0.8, "none")),
    "noisy_vs_full_m0.3": pdiff(("noisy", 0.3, "none"), ("full", 0.3, "none")),
}
# theory for persuasion loss: using oracle cell statistics (full feedback, explanation cells), predicted loss from bias b
pred_loss = []
for sd in SEEDS:
    ai = build_ai(sd); g = rng(5000 + sd)
    n = 40000; it = g.integers(0, len(ai["y"]), n)
    y, a, cb, z = ai["y"][it], ai["pred"][it], ai["cbin"][it], ai["z"][it]
    h = human_answer(g, y, z, P0); dis = h != a
    pc, pw, b, _ = EXPL["persuasive"]
    plaus = g.random(n) < np.where(a == y, pc, pw)
    cell = cb * 2 + plaus
    loss = 0.0
    for c in range(10):
        m = dis & (cell == c)
        if m.sum() == 0:
            continue
        q = (a[m] == y[m]).mean(); hh = (h[m] == y[m]).mean()
        bias = b if c % 2 == 1 else 0.0
        if q < hh < q + bias:  # biased rule follows AI although human is better
            loss += m.mean() * (hh - q)
    pred_loss.append(100 * loss)
out["theory_persuasion_loss_pp"] = mean_ci(pred_loss)
print(f"theory {time.time()-T0:.1f}s", flush=True)

# ------------------------------------------------------------------ deskilling dynamics
desk = {d: [] for d in DELTAS}
fixed = {d: [] for d in DELTAS}
for sd in SEEDS:
    ai = build_ai(sd)
    ps = np.linspace(0.3, 0.85, 12)
    Fs = [static_follow_rate(ai, rng(6000 + sd), p)[0] for p in ps]
    for d in DELTAS:
        r = run(ai, rng(7000 + 13 * sd + int(100 * d)), "full", 0.5, None, Tn=T_LONG, delta=d)
        desk[d].append(dict(team_last=float(r["team"][-500:].mean()), ai=float(r["ai"].mean()),
                            p_h_final=r["p_h_final"], follow_last=float(r["follow"][-500:].mean())))
        # fixed point of p = P0 - d * F(p) by iteration on interpolated F
        p = P0
        for _ in range(200):
            p = P0 - d * np.interp(p, ps, Fs)
        fixed[d].append(dict(p_star=float(p), F_star=float(np.interp(p, ps, Fs)),
                             team_star=static_follow_rate(ai, rng(8000 + sd), p)[1]))
    print(f"seed {sd} deskill {time.time()-T0:.1f}s", flush=True)
out["deskilling"] = {str(d): {"sim_p_h_final": mean_ci([x["p_h_final"] for x in desk[d]]),
                              "sim_team_last500": mean_ci([x["team_last"] for x in desk[d]]),
                              "sim_follow_last500": mean_ci([x["follow_last"] for x in desk[d]]),
                              "ai_alone": mean_ci([x["ai"] for x in desk[d]]),
                              "theory_p_star": mean_ci([x["p_star"] for x in fixed[d]]),
                              "theory_team_star": mean_ci([x["team_star"] for x in fixed[d]]),
                              "team_minus_ai_pp": mean_ci([100 * (x["team_last"] - x["ai"]) for x in desk[d]])} for d in DELTAS}
out["runtime_s"] = time.time() - T0
od = results_dir(__file__)
(od / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

# ------------------------------------------------------------------ figures
P = plt()
fig, ax = P.subplots(1, 3, figsize=(16, 4.3))
xs = np.arange(len(summ["full|0.3|none"]["curve"])) * 5 + 25
for (fb, m0), c in zip([("full", 0.3), ("full", 0.8), ("noisy", 0.3), ("none", 0.3), ("none", 0.8)], ["C0", "C1", "C2", "C3", "C4"]):
    ax[0].plot(xs, summ[f"{fb}|{m0}|none"]["curve"], color=c, label=f"反馈={fb}, 先验信任={m0}")
ax[0].axhline(summ["full|0.3|none"]["ai_alone"][0], color="k", ls="--", label="AI 单独")
ax[0].axhline(summ["full|0.3|none"]["human_alone"][0], color="k", ls=":", label="人单独")
ax[0].set_xlabel("试次"); ax[0].set_ylabel("团队准确率（25 试次滑动）"); ax[0].legend(fontsize=7); ax[0].set_title("反馈驱动的信任学习")
ens = list(EXPL)
for i, m0 in enumerate(PRIORS):
    ax[1].bar(np.arange(len(ens)) + 0.38 * i, [summ[f"full|{m0}|{e}"]["gain_last100_pp"][0] for e in ens], 0.38,
              yerr=[summ[f"full|{m0}|{e}"]["gain_last100_pp"][1] for e in ens], capsize=3, label=f"先验信任 {m0}")
ax[1].axhline(0, color="k"); ax[1].set_xticks(np.arange(len(ens)) + 0.19); ax[1].set_xticklabels(ens, fontsize=8, rotation=10)
ax[1].set_ylabel("末 100 试次互补增益 (pp)"); ax[1].legend(fontsize=8); ax[1].set_title("解释：说服、诊断与挤出（完整反馈）")
ds = DELTAS
ax[2].errorbar(ds, [out["deskilling"][str(d)]["sim_team_last500"][0] for d in ds], yerr=[out["deskilling"][str(d)]["sim_team_last500"][1] for d in ds], marker="o", capsize=3, label="团队（仿真）")
ax[2].plot(ds, [out["deskilling"][str(d)]["theory_team_star"][0] for d in ds], "s--", label="团队（不动点理论）")
ax[2].errorbar(ds, [out["deskilling"][str(d)]["sim_p_h_final"][0] for d in ds], yerr=[out["deskilling"][str(d)]["sim_p_h_final"][1] for d in ds], marker="^", capsize=3, label="人独立准确率（仿真）")
ax[2].plot(ds, [out["deskilling"][str(d)]["theory_p_star"][0] for d in ds], "v:", label="人准确率（理论 p*）")
ax[2].axhline(np.mean(ai_acc), color="k", ls="--", label="AI 单独")
ax[2].set_xlabel("技能退化强度 δ"); ax[2].set_ylabel("准确率"); ax[2].legend(fontsize=7); ax[2].set_title("依赖–退化反馈环")
fig.tight_layout(); fig.savefig(od / "frontier_reliance_dynamics.png"); P.close(fig)
print(f"done {time.time()-T0:.1f}s")
