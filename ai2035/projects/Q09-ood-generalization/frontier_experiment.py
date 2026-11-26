"""Q09 前沿实验：长度泛化——哪种归纳偏置能把在短序列上学到的算法外推到长序列？

任务：模 p 前缀和（p=2 即奇偶校验；p=3 为三元序列模 3 求和）。训练长度 L∈[1,10]，测试 L 到 40。
模型（均为 numpy/sklearn）：
  A 位置索引 MLP（每个位置 one-hot，补零到 40；类比绝对位置编码）——训练长度 10 与 20 两个版本；
  B 计数 MLP（对 token 计数后输入 MLP；类比“无位置、求和池化”的归纳偏置）；
  C 中间监督的递归转移模型（CoT/scratchpad：学习 (状态, token)→新状态，推理时循环展开），
     贪心解码，以及在中间标签加噪 η 后按预测概率采样解码；
  D 端到端训练的小型 tanh RNN（BPTT，无中间监督）。
理论（paper.md §4）：A 在 L>L_train 时准确率恰为 1/p；B 在计数超出训练范围后退化为常数预测（→1/p）；
  C 的逐步误差 ε 下，准确率 acc(L)=1/p+(1-1/p)(1-εp/(p-1))^L，贪心解码 ε=0 → 任意长度 100%。
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, mean_ci, plt, results_dir

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
OUT = results_dir(__file__)
SEEDS = 5
LMAX = 40
TEST_L = [1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 20, 24, 30, 40]
NTRAIN, NTEST = 4000, 500
ETAS = [0.0, 0.02, 0.05, 0.1]


def gen(nseq, Lmin, Lmax, p, g):
    Ls = g.integers(Lmin, Lmax + 1, nseq)
    seqs = [g.integers(0, p, L) for L in Ls]
    return seqs, np.array([s.sum() % p for s in seqs])


def pos_feats(seqs, p):
    F = np.zeros((len(seqs), LMAX * p))
    for i, s in enumerate(seqs):
        F[i, np.arange(len(s)) * p + s] = 1
    return F


def count_feats(seqs, p):
    return np.array([np.bincount(s, minlength=p) for s in seqs], float)


def mlp(seed, h=(256,)):
    return MLPClassifier(hidden_layer_sizes=h, max_iter=200, alpha=1e-4, random_state=seed)


def transition_data(seqs, p, eta, g):
    Xs, ys = [], []
    for s in seqs:
        st = 0
        for tok in s:
            nxt = (st + tok) % p
            lab = nxt
            if g.uniform() < eta:
                lab = (nxt + g.integers(1, p)) % p          # wrong state, uniformly among the others
            Xs.append(np.r_[np.eye(p)[st], np.eye(p)[tok]]); ys.append(lab)
            st = nxt                                         # teacher forcing on the true state
    return np.array(Xs), np.array(ys)


def trans_table(model, p):
    T = np.zeros((p, p, p))
    for st in range(p):
        for tok in range(p):
            T[st, tok] = model.predict_proba(np.r_[np.eye(p)[st], np.eye(p)[tok]][None])[0]
    return T


def rollout(T, S, p, g, sample):
    """S: (m, L) int array of equal-length sequences; vectorised over sequences."""
    st = np.zeros(len(S), int)
    for t in range(S.shape[1]):
        pr = T[st, S[:, t]]
        if sample:
            u = g.uniform(size=(len(S), 1))
            st = np.minimum((u > np.cumsum(pr, 1)).sum(1), p - 1)
        else:
            st = pr.argmax(1)
    return st


# ---------- numpy tanh RNN (end-to-end, BPTT, Adam) ----------
def train_rnn(p, g, H=16, steps=3000, lr=0.01, B=64):
    P_ = {"W": g.standard_normal((H, H)) * 0.5 / np.sqrt(H), "U": g.standard_normal((H, p)) * 1.0,
          "b": np.zeros(H), "V": g.standard_normal((p, H)) * 0.3, "c": np.zeros(p)}
    m = {k: np.zeros_like(v) for k, v in P_.items()}; v2 = {k: np.zeros_like(v) for k, v in P_.items()}
    for it in range(1, steps + 1):
        L = int(g.integers(1, 11))
        X = g.integers(0, p, (B, L)); y = X.sum(1) % p
        hs = [np.zeros((B, H))]
        for t in range(L):
            hs.append(np.tanh(hs[-1] @ P_["W"].T + P_["U"][:, X[:, t]].T + P_["b"]))
        o = hs[-1] @ P_["V"].T + P_["c"]; o -= o.max(1, keepdims=True)
        pr = np.exp(o); pr /= pr.sum(1, keepdims=True)
        do = pr.copy(); do[np.arange(B), y] -= 1; do /= B
        gr = {k: np.zeros_like(v) for k, v in P_.items()}
        gr["V"] = do.T @ hs[-1]; gr["c"] = do.sum(0)
        dh = do @ P_["V"]
        for t in range(L, 0, -1):
            da = dh * (1 - hs[t] ** 2)
            gr["W"] += da.T @ hs[t - 1]; gr["b"] += da.sum(0)
            np.add.at(gr["U"].T, X[:, t - 1], da)
            dh = da @ P_["W"]
        for k in P_:
            gn = np.clip(gr[k], -1, 1)
            m[k] = 0.9 * m[k] + 0.1 * gn; v2[k] = 0.999 * v2[k] + 0.001 * gn ** 2
            P_[k] -= lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v2[k] / (1 - 0.999 ** it)) + 1e-8)
    return P_


def rnn_predict(P_, S):
    h = np.zeros((len(S), P_["W"].shape[0]))
    for t in range(S.shape[1]):
        h = np.tanh(h @ P_["W"].T + P_["U"][:, S[:, t]].T + P_["b"])
    return (h @ P_["V"].T + P_["c"]).argmax(1)


results = {}
for p in [2, 3]:
    R = {k: {L: [] for L in TEST_L} for k in ["A 位置MLP(训练≤10)", "A' 位置MLP(训练≤20)", "B 计数MLP", "C 递归CoT-贪心", "D 端到端RNN"] +
         [f"C 递归CoT-采样 η={e}" for e in ETAS]}
    eps_meas = {e: [] for e in ETAS}
    const_frac = []
    for s in range(SEEDS):
        g = rng(1000 * p + s)
        seqs, y = gen(NTRAIN, 1, 10, p, g)
        seqs20, y20 = gen(NTRAIN, 1, 20, p, g)
        mA = mlp(s).fit(pos_feats(seqs, p), y)
        mA2 = mlp(s).fit(pos_feats(seqs20, p), y20)
        mB = mlp(s, (64, 64)).fit(count_feats(seqs, p), y)
        trans = {}
        for e in ETAS:
            Xt, yt = transition_data(seqs[:1500], p, e, g)
            trans[e] = MLPClassifier(hidden_layer_sizes=(32,), max_iter=600, random_state=s).fit(Xt, yt)
        rnn = train_rnn(p, g)
        tabs = {e: trans_table(trans[e], p) for e in ETAS}
        for L in TEST_L:
            ts, ty = gen(NTEST, L, L, p, g)
            S = np.array(ts)
            R["A 位置MLP(训练≤10)"][L].append(np.mean(mA.predict(pos_feats(ts, p)) == ty))
            R["A' 位置MLP(训练≤20)"][L].append(np.mean(mA2.predict(pos_feats(ts, p)) == ty))
            pb = mB.predict(count_feats(ts, p))
            R["B 计数MLP"][L].append(np.mean(pb == ty))
            if L == 40:
                const_frac.append(np.max(np.bincount(pb, minlength=p)) / len(pb))
            pg = rollout(tabs[0.0], S, p, g, sample=False)
            R["C 递归CoT-贪心"][L].append(np.mean(pg == ty))
            for e in ETAS:
                ps_ = rollout(tabs[e], S, p, g, sample=True)
                R[f"C 递归CoT-采样 η={e}"][L].append(np.mean(ps_ == ty))
            R["D 端到端RNN"][L].append(np.mean(rnn_predict(rnn, S) == ty))
        for e in ETAS:
            table = tabs[e]
            eps_meas[e].append(1 - np.mean([table[st, tok, (st + tok) % p] for st in range(p) for tok in range(p)]))
        print("p", p, "seed", s, round(time.time() - T0, 1), flush=True)
    summ = {k: {L: dict(zip(["mean", "ci95"], mean_ci(v))) for L, v in d.items()} for k, d in R.items()}
    # theory checks
    chance = 1 / p
    th = {}
    # A: accuracy beyond training length vs chance (one-sample t-test pooled over L>Ltrain)
    for key, Ltr in [("A 位置MLP(训练≤10)", 10), ("A' 位置MLP(训练≤20)", 20)]:
        beyond = [a for L in TEST_L if L > Ltr for a in R[key][L]]
        th[key] = {"beyond_mean": float(np.mean(beyond)), "beyond_ci95": mean_ci(beyond)[1], "chance": chance,
                   "p_vs_chance": float(stats.ttest_1samp(beyond, chance).pvalue),
                   "in_range_mean": float(np.mean([a for L in TEST_L if L <= Ltr for a in R[key][L]]))}
    th["B 计数MLP"] = {"L40_majority_class_fraction": dict(zip(["mean", "ci95"], mean_ci(const_frac))),
                      "in_range_mean": float(np.mean([a for L in TEST_L if L <= 10 for a in R["B 计数MLP"][L]])),
                      "beyond20_mean": float(np.mean([a for L in TEST_L if L >= 20 for a in R["B 计数MLP"][L]]))}
    th["C_sampling"] = {}
    for e in ETAS:
        eps = float(np.mean(eps_meas[e]))
        pred = {L: chance + (1 - chance) * (1 - eps * p / (p - 1)) ** L for L in TEST_L}
        meas = {L: summ[f"C 递归CoT-采样 η={e}"][L]["mean"] for L in TEST_L}
        th["C_sampling"][e] = {"eps_measured": eps, "eps_ci95": mean_ci(eps_meas[e])[1], "pred": pred,
                               "max_abs_dev": float(max(abs(pred[L] - meas[L]) for L in TEST_L))}
    rnn40 = np.array(R["D 端到端RNN"][40]); rnn10 = np.array(R["D 端到端RNN"][10])
    th["D_rnn"] = {"seeds_ge95_at_L40": int((rnn40 >= 0.95).sum()), "seeds_ge95_at_L10": int((rnn10 >= 0.95).sum()),
                   "per_seed_L10": rnn10.tolist(), "per_seed_L40": rnn40.tolist()}
    results[p] = {"accuracy": summ, "theory_checks": th}

# ---------------- figures ----------------
P = plt()
fig, axes = P.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, p in zip(axes, [2, 3]):
    S = results[p]["accuracy"]
    for k, st in [("A 位置MLP(训练≤10)", "o-"), ("A' 位置MLP(训练≤20)", "o--"), ("B 计数MLP", "s-"), ("C 递归CoT-贪心", "^-"),
                  ("C 递归CoT-采样 η=0.05", "v-"), ("C 递归CoT-采样 η=0.1", "v--"), ("D 端到端RNN", "d-")]:
        ax.errorbar(TEST_L, [S[k][L]["mean"] for L in TEST_L], yerr=[S[k][L]["ci95"] for L in TEST_L], fmt=st, ms=3.5, capsize=2, label=k)
    for e, ls in [(0.05, ":"), (0.1, "-.")]:
        pr = results[p]["theory_checks"]["C_sampling"][e]["pred"]
        ax.plot(TEST_L, [pr[L] for L in TEST_L], "k" + ls, lw=1, label=f"理论 acc(L), η={e}")
    ax.axhline(1 / p, color="grey", lw=.8); ax.axvline(10.5, color="grey", ls=":", lw=.8)
    ax.set_xlabel("测试序列长度 L（训练 L≤10，虚线）"); ax.set_ylabel("准确率"); ax.set_title(f"模 {p} 求和（{'奇偶校验' if p == 2 else '三元'}）")
    ax.grid(alpha=.3)
axes[1].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "frontier_length_generalization.png"); P.close(fig)

runtime = time.time() - T0
res = {"config": {"seeds": SEEDS, "train_lengths": [1, 10], "test_lengths": TEST_L, "n_train": NTRAIN, "n_test_per_length": NTEST,
                  "posMLP": "one-hot per position padded to 40, MLP(256)", "countMLP": "token counts, MLP(64,64)",
                  "CoT": "MLP(32) on (state,token) one-hots, 1500 training sequences, intermediate labels corrupted w.p. eta",
                  "RNN": "tanh H=16, BPTT, Adam lr .01, 3000 steps, batch 64, lengths 1-10", "etas": ETAS},
       "results": results, "runtime_sec": runtime}
(OUT / "frontier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print("runtime", runtime)
