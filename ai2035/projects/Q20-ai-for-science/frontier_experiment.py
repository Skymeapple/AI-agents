"""Q20 前沿实验：机器学习原子间势（MLIP）中的对称性、能量守恒与外推——在 LJ 团簇上的受控研究。

问题（2025–2026 争论）：测试误差低 ≠ MD 中能量守恒与物性正确 (arXiv:2502.12147, 2509.20630)；
直接预测力（非保守）是否可接受 (Bigi 等 arXiv:2412.11569)；不变/等变结构与数据效率
(2503.05771, 2610.14239)；外推到训练分布之外 (2608.22933, 2610.14891, 2512.23939)。

设计（参数在运行前固定）：
- 体系：N=6 原子的 Lennard-Jones 团簇（ε=σ=1），参考数据由带 Andersen 恒温器的 MD 在 T=0.25 采样
  （每 25 步取一个构型）；“热”测试集 T=0.45。标签噪声：σ_E=0.02，σ_F=0.05。
- 模型（全部为线性最小二乘，岭 λ=1e-6·n_rows）：
  RAW    : 原始（去质心）笛卡尔坐标上的随机傅里叶特征（D=1000，ℓ=0.5），用能量+力联合拟合（保守，但不具旋转/置换不变性）；
  RAW+AUG: 同上 + 每个构型 8 个随机旋转×置换增广副本（只在 n≤300 时）；
  INV-E  : 不变对势展开 E = Σ_{i<j} Σ_k w_k φ_k(r_ij)，φ_k 为 16 个高斯径向基（μ∈[0.85,2.6]，宽 0.15），
           能量+力联合拟合（保守 + 不变）；
  INV-E+rep: INV-E 加物理形式特征 r^-12、r^-6（“正确的物理先验”，检验外推）；
  DF-pair: 直接拟合力 F_i = Σ_j Σ_k w_k φ_k(r_ij) r̂_ij（等变、中心对力 → 保守）；
  DF-nc  : 直接拟合力 F_i = Σ_j Σ_k Σ_m w_km φ_k(r_ij) ψ_m(ρ_i) r̂_ij，ψ_m(ρ)=ρ^m，m=0,1,2，
           ρ_i = Σ_j exp(-r_ij²)（等变，但依赖于 i 的环境 → 非保守）。
- E1 数据效率：n ∈ {10, 30, 100, 300, 1000} 个构型，测试力/能量 RMSE。
- E2 NVE 分子动力学：每模型 10 条轨迹（n=1000 训练），速度 Verlet dt=0.005，4000 步；
  （试运行后修正为 dt=0.002、10000 步：dt=0.005 时参考 LJ 力自身的积分误差主导漂移；岭正则改为相对迹
  1e-6·tr(A)/p，原设置在小样本下病态；环路边长 h 由 0.02 改为 0.005 以降低离散化基线。）
  记录真实总能量 E_LJ+KE 的漂移（线性斜率与终点偏差）、“失稳”（最小间距<0.75 或最大间距>3.5）。
  非保守度：配置空间小方环路（边长 h=0.02）上的环路功 |∮F·dx|/h²。
- E3 外推：热测试集上的力 RMSE / 分布内力 RMSE。
- 5 个种子。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng  # noqa: E402

from scipy import stats  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
NA = 6
SIG_E, SIG_F = 0.02, 0.05
NS = [10, 30, 100, 300, 1000]
MU = np.linspace(0.85, 2.6, 16); WID = 0.15
D_RFF, ELL = 1000, 0.5
IU = np.triu_indices(NA, 1)


# ------------------------------------------------------------------ LJ reference
def lj(X):
    """X: (B, NA, 3) -> E (B,), F (B, NA, 3)"""
    d = X[:, :, None, :] - X[:, None, :, :]
    r2 = (d ** 2).sum(-1) + np.eye(NA)[None] * 1e6
    ir6 = 1 / r2 ** 3
    E = (4 * (ir6 ** 2 - ir6)).sum((1, 2)) / 2  # each pair counted twice in the double sum
    fmag = 24 * (2 * ir6 ** 2 - ir6) / r2
    F = (fmag[..., None] * d).sum(2)
    return E, F


def sample_md(g, n, T, nchains=20, every=25, burn=400, dt=0.005):
    base = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * 2 ** (1 / 6) / np.sqrt(2)
    X = base[None].repeat(nchains, 0) + 0.02 * g.standard_normal((nchains, NA, 3))
    V = np.sqrt(T) * g.standard_normal((nchains, NA, 3))
    _, F = lj(X); out = []
    step = 0
    while len(out) * nchains < n:
        V += 0.5 * dt * F; X += dt * V; _, F = lj(X); V += 0.5 * dt * F
        resel = g.random((nchains, NA)) < 0.01  # Andersen thermostat
        V[resel] = np.sqrt(T) * g.standard_normal((resel.sum(), 3))
        step += 1
        if step > burn and step % every == 0:
            out.append(X - X.mean(1, keepdims=True))
    X = np.concatenate(out)[:n]
    return X


# ------------------------------------------------------------------ features
def pair_geom(X):
    d = X[:, :, None, :] - X[:, None, :, :]
    r = np.sqrt((d ** 2).sum(-1) + np.eye(NA)[None])
    return d, r


def phi(r):
    return np.exp(-((r[..., None] - MU) ** 2) / (2 * WID ** 2))


def dphi(r):
    return phi(r) * (-(r[..., None] - MU) / WID ** 2)


def design_inv(X, rep=False):
    """rows: energy (B, K) and forces (B*NA*3, K) for INV-E (rep: add r^-12, r^-6 physics features)"""
    d, r = pair_geom(X)
    mask = 1 - np.eye(NA)
    P = phi(r); dP = dphi(r)
    if rep:
        P = np.concatenate([P, (r ** -12)[..., None], (r ** -6)[..., None]], -1)
        dP = np.concatenate([dP, (-12 * r ** -13)[..., None], (-6 * r ** -7)[..., None]], -1)
    P = P * mask[None, :, :, None]
    AE = P.sum((1, 2)) / 2
    dP = dP * mask[None, :, :, None]  # dE/dr_ij
    # F_i = - sum_j dphi(r_ij) * (x_i - x_j)/r_ij
    AF = -(dP[:, :, :, None, :] * (d / r[..., None])[..., None]).sum(2)  # (B, NA, 3, K)
    return AE, AF.reshape(-1, AE.shape[1])


def design_df(X, nc):
    d, r = pair_geom(X)
    mask = 1 - np.eye(NA)
    P = phi(r) * mask[None, :, :, None]
    rhat = d / r[..., None]
    base = (P[:, :, :, None, :] * rhat[..., None]).sum(2)  # (B, NA, 3, K)  sum_j phi(r_ij) rhat_ij
    if not nc:
        return base.reshape(-1, len(MU))
    rho = (np.exp(-r ** 2) * mask[None]).sum(2)  # (B, NA)
    cols = [base * (rho[:, :, None, None] ** m) for m in range(3)]
    return np.concatenate(cols, -1).reshape(-1, 3 * len(MU))


class RawRFF:
    def __init__(self, g):
        self.W = g.standard_normal((NA * 3, D_RFF)) / ELL
        self.b = g.uniform(0, 2 * np.pi, D_RFF)

    def design(self, X):
        x = X.reshape(len(X), -1)
        z = x @ self.W + self.b
        AE = np.sqrt(2 / D_RFF) * np.cos(z)
        # dE/dx = sqrt(2/D) * (-sin z) * W ;  F = -dE/dx
        AF = (np.sqrt(2 / D_RFF) * np.sin(z))[:, None, :] * self.W[None, :, :]  # (B, 18, D)
        return AE, AF.reshape(-1, D_RFF)


def solve(blocks, lam=1e-6):
    A = sum(a.T @ a for a, _ in blocks)
    bvec = sum(a.T @ y for a, y in blocks)
    return np.linalg.solve(A + lam * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), bvec)


def aug(X, g, k=8):
    outs = []
    for _ in range(k):
        R = Rotation.random(len(X), random_state=int(g.integers(1 << 30))).as_matrix()
        Xr = np.einsum("bij,baj->bai", R, X)
        perm = np.array([g.permutation(NA) for _ in range(len(X))])
        outs.append((np.take_along_axis(Xr, perm[:, :, None], 1), R, perm))
    return outs


class Model:
    def __init__(self, kind, g):
        self.kind = kind
        if kind.startswith("RAW"):
            self.rff = RawRFF(g)

    def fit(self, X, E, F, g=None):
        if self.kind.startswith("RAW"):
            Xs, Es, Fs = [X], [E], [F]
            if self.kind == "RAW+AUG":
                for Xa, R, perm in aug(X, g):
                    Fa = np.take_along_axis(np.einsum("bij,baj->bai", R, F), perm[:, :, None], 1)
                    Xs.append(Xa); Es.append(E); Fs.append(Fa)
            blocks = []
            for Xi, Ei, Fi in zip(Xs, Es, Fs):
                AE, AF = self.rff.design(Xi)
                blocks += [(AE / SIG_E, (Ei - self.e0(Ei)) / SIG_E), (AF / SIG_F, Fi.reshape(-1) / SIG_F)]
            self.Emean = E.mean()
            self.w = solve(blocks)
        elif self.kind.startswith("INV-E"):
            AE, AF = design_inv(X, self.kind == "INV-E+rep")
            A1 = np.concatenate([AE, np.ones((len(AE), 1))], 1); A2 = np.concatenate([AF, np.zeros((len(AF), 1))], 1)
            self.w = solve([(A1 / SIG_E, E / SIG_E), (A2 / SIG_F, F.reshape(-1) / SIG_F)])
        else:
            A = design_df(X, self.kind == "DF-nc")
            self.w = solve([(A / SIG_F, F.reshape(-1) / SIG_F)])
        return self

    def e0(self, E):
        return E.mean()

    def forces(self, X):
        if self.kind.startswith("RAW"):
            return (self.rff.design(X)[1] @ self.w).reshape(len(X), NA, 3)
        if self.kind.startswith("INV-E"):
            return (design_inv(X, self.kind == "INV-E+rep")[1] @ self.w[:-1]).reshape(len(X), NA, 3)
        return (design_df(X, self.kind == "DF-nc") @ self.w).reshape(len(X), NA, 3)

    def energy(self, X):
        if self.kind.startswith("RAW"):
            return self.rff.design(X)[0] @ self.w + self.Emean
        if self.kind.startswith("INV-E"):
            return design_inv(X, self.kind == "INV-E+rep")[0] @ self.w[:-1] + self.w[-1]
        return None


def loop_work(model, X, g, h=0.005):
    B = len(X)
    u = g.standard_normal((B, NA, 3)); u /= np.linalg.norm(u.reshape(B, -1), axis=1)[:, None, None]
    v = g.standard_normal((B, NA, 3)); v -= (u * v).sum((1, 2))[:, None, None] * u
    v /= np.linalg.norm(v.reshape(B, -1), axis=1)[:, None, None]
    W = np.zeros(B)
    corners = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    for (a0, b0), (a1, b1) in zip(corners[:-1], corners[1:]):
        mid = X + h * ((a0 + a1) / 2 * u + (b0 + b1) / 2 * v)
        dx = h * ((a1 - a0) * u + (b1 - b0) * v)
        W += (model.forces(mid) * dx).sum((1, 2))
    return float(np.mean(np.abs(W)) / h ** 2)


def nve(force_fn, X0, V0, steps=10000, dt=0.002):
    X, V = X0.copy(), V0.copy()
    F = force_fn(X); E0 = lj(X)[0] + 0.5 * (V ** 2).sum((1, 2))
    Et = []; alive = np.ones(len(X), bool)
    for s in range(steps):
        V += 0.5 * dt * F; X += dt * V
        F = np.nan_to_num(force_fn(X), nan=0.0, posinf=0.0, neginf=0.0); F = np.clip(F, -1e3, 1e3)
        V += 0.5 * dt * F
        if s % 50 == 0:
            _, r = pair_geom(X)
            rr = r + np.eye(NA)[None] * 10
            bad = (rr.min((1, 2)) < 0.75) | (r.max((1, 2)) > 3.5) | ~np.isfinite(X).all((1, 2))
            alive &= ~bad
            Et.append(lj(X)[0] + 0.5 * (V ** 2).sum((1, 2)))
    Et = np.array(Et) - E0[None]  # (t, B)
    tt = np.arange(len(Et)) * 50 * dt
    slopes = np.array([np.polyfit(tt, Et[:, b], 1)[0] for b in range(Et.shape[1])])
    return dict(abs_slope=np.abs(slopes), final_dev=np.abs(Et[-1]), max_dev=np.abs(Et).max(0), alive=alive)


KINDS = ["RAW", "RAW+AUG", "INV-E", "INV-E+rep", "DF-pair", "DF-nc"]
E1 = {k: {n: {"F": [], "E": [], "F_hot": []} for n in NS} for k in KINDS}
ZERO = []
MD = {k: {"abs_slope": [], "final_dev": [], "max_dev": [], "alive": [], "loop": []} for k in KINDS + ["TRUE"]}
for sd in SEEDS:
    g = rng(100 + sd)
    Xtr = sample_md(g, max(NS), 0.25)
    Xte = sample_md(rng(200 + sd), 400, 0.25); Xhot = sample_md(rng(300 + sd), 400, 0.45)
    Etr, Ftr = lj(Xtr); Ete, Fte = lj(Xte); Ehot, Fhot = lj(Xhot)
    ZERO.append(float(np.sqrt((Fte ** 2).mean())))
    Etr_n = Etr + SIG_E * g.standard_normal(Etr.shape); Ftr_n = Ftr + SIG_F * g.standard_normal(Ftr.shape)
    fitted = {}
    for n in NS:
        for k in KINDS:
            if k == "RAW+AUG" and n > 300:
                continue
            m = Model(k, rng(400 + sd)).fit(Xtr[:n], Etr_n[:n], Ftr_n[:n], g=rng(500 + sd + n))
            Fp = m.forces(Xte)
            E1[k][n]["F"].append(float(np.sqrt(((Fp - Fte) ** 2).mean())))
            E1[k][n]["F_hot"].append(float(np.sqrt(((m.forces(Xhot) - Fhot) ** 2).mean())))
            Ep = m.energy(Xte)
            if Ep is not None:
                E1[k][n]["E"].append(float(np.sqrt(((Ep - Ete) ** 2).mean())))
            if n == 1000 or (k == "RAW+AUG" and n == 300):
                fitted[k] = m
    # MD
    gm = rng(600 + sd)
    X0 = Xte[gm.choice(len(Xte), 10, replace=False)].copy()
    V0 = np.sqrt(0.25) * gm.standard_normal((10, NA, 3)); V0 -= V0.mean(1, keepdims=True)
    for k in KINDS + ["TRUE"]:
        fn = (lambda X: lj(X)[1]) if k == "TRUE" else fitted[k].forces
        r = nve(fn, X0, V0)
        for key in ("abs_slope", "final_dev", "max_dev"):
            MD[k][key].append(float(np.median(r[key])))
        MD[k]["alive"].append(float(r["alive"].mean()))
        if k != "TRUE":
            MD[k]["loop"].append(loop_work(fitted[k], Xte[:50], rng(700 + sd)))
        else:
            class _T:
                forces = staticmethod(lambda X: lj(X)[1])
            MD[k]["loop"].append(loop_work(_T, Xte[:50], rng(700 + sd)))
    print(f"seed {sd} {time.time()-T0:.1f}s", flush=True)

out = {"config": dict(seeds=SEEDS, n_atoms=NA, sig_E=SIG_E, sig_F=SIG_F, ns=NS, mu=MU.tolist(), width=WID, D_rff=D_RFF, ell=ELL,
                      md=dict(dt=0.002, steps=10000, n_traj=10, T=0.25), loop_h=0.005, kinds=KINDS)}
out["zero_predictor_force_rmse"] = mean_ci(ZERO)
out["E1"] = {k: {str(n): {m: mean_ci(v) for m, v in E1[k][n].items() if v} for n in NS if E1[k][n]["F"]} for k in KINDS}
out["E2_md"] = {k: {key: mean_ci(v) for key, v in MD[k].items()} for k in MD}
out["E3_extrap_ratio"] = {k: mean_ci(np.array(E1[k][300]["F_hot"]) / np.array(E1[k][300]["F"])) for k in KINDS}
# data-efficiency: smallest n at which INV-E force error <= RAW at n=1000
raw1000 = np.mean(E1["RAW"][1000]["F"])
out["inv_n_to_match_raw1000"] = next((n for n in NS if np.mean(E1["INV-E"][n]["F"]) <= raw1000), None)
out["tests"] = {
    "drift_DFnc_vs_INV": {"ratio_geo": float(np.exp(np.mean(np.log(np.array(MD["DF-nc"]["abs_slope"]) / np.array(MD["INV-E"]["abs_slope"]))))),
                          "p": float(stats.ttest_rel(np.log(MD["DF-nc"]["abs_slope"]), np.log(MD["INV-E"]["abs_slope"])).pvalue)},
    "drift_DFnc_vs_DFpair": {"ratio_geo": float(np.exp(np.mean(np.log(np.array(MD["DF-nc"]["abs_slope"]) / np.array(MD["DF-pair"]["abs_slope"]))))),
                             "p": float(stats.ttest_rel(np.log(MD["DF-nc"]["abs_slope"]), np.log(MD["DF-pair"]["abs_slope"])).pvalue)},
    "forceRMSE_DFnc_vs_INV_n1000": {"diff": mean_ci(np.array(E1["DF-nc"][1000]["F"]) - np.array(E1["INV-E"][1000]["F"])),
                                    "p": float(stats.ttest_rel(E1["DF-nc"][1000]["F"], E1["INV-E"][1000]["F"]).pvalue)},
    "forceRMSE_INV30_vs_RAW1000": {"diff": mean_ci(np.array(E1["INV-E"][30]["F"]) - np.array(E1["RAW"][1000]["F"])),
                                   "p": float(stats.ttest_rel(E1["INV-E"][30]["F"], E1["RAW"][1000]["F"]).pvalue)},
}
allk = [k for k in KINDS]
stable = [k for k in KINDS if not k.startswith("RAW")]
out["corr_loop_vs_drift_spearman_stable_models"] = float(stats.spearmanr([v for k in stable for v in MD[k]["loop"]], [v for k in stable for v in MD[k]["abs_slope"]]).statistic)
out["corr_loop_vs_drift_spearman"] = float(stats.spearmanr([v for k in allk for v in MD[k]["loop"]], [v for k in allk for v in MD[k]["abs_slope"]]).statistic)
out["runtime_s"] = time.time() - T0
od = results_dir(__file__)
(od / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

# ------------------------------------------------------------------ figures
P = plt()
fig, ax = P.subplots(1, 3, figsize=(16, 4.3))
for k, c in zip(KINDS, ["C0", "C1", "C2", "C5", "C3", "C4"]):
    ns = [n for n in NS if str(n) in out["E1"][k]]
    ax[0].errorbar(ns, [out["E1"][k][str(n)]["F"][0] for n in ns], yerr=[out["E1"][k][str(n)]["F"][1] for n in ns], marker="o", capsize=2, color=c, label=k)
ax[0].axhline(SIG_F, color="gray", ls=":", label="标签噪声 σ_F")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("训练构型数 n"); ax[0].set_ylabel("测试力 RMSE"); ax[0].legend(fontsize=7); ax[0].set_title("对称性与数据效率")
ks = KINDS + ["TRUE"]
ax[1].bar(range(len(ks)), [out["E2_md"][k]["abs_slope"][0] for k in ks], yerr=[out["E2_md"][k]["abs_slope"][1] for k in ks], capsize=3)
ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels(ks, fontsize=8, rotation=15); ax[1].set_yscale("log")
ax[1].set_ylabel("真实总能量漂移斜率 |dE/dt|（中位数）"); ax[1].set_title("NVE 能量守恒")
ax[2].scatter([v for k in KINDS for v in MD[k]["loop"]], [v for k in KINDS for v in MD[k]["abs_slope"]],
              c=[KINDS.index(k) for k in KINDS for _ in MD[k]["loop"]], cmap="tab10", s=25)
for i, k in enumerate(KINDS):
    ax[2].annotate(k, (np.mean(MD[k]["loop"]), np.mean(MD[k]["abs_slope"])), fontsize=7)
ax[2].set_xscale("symlog", linthresh=1e-4); ax[2].set_yscale("log"); ax[2].set_xlabel("环路功 |∮F·dx|/h²（非保守度）"); ax[2].set_ylabel("能量漂移斜率")
ax[2].set_title(f"非保守度与漂移（稳定模型 Spearman {out['corr_loop_vs_drift_spearman_stable_models']:.2f}）")
fig.tight_layout(); fig.savefig(od / "frontier_mlip.png"); P.close(fig)
print(f"done {time.time()-T0:.1f}s")
