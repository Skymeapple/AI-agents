"""R2 core experiment: the first-order speed limit of the stability gap (Theorems 5-7 of the revised paper).

d=100 realizable linear model (same tasks/e0 as linear_theory.py, seeds 0-9).  Every *linear first-order*
method on the replay objective produces e_k = p_k(H) e0 with p_k(0)=1, so we simulate all of them exactly in
the eigenbasis of H (scalar recurrences per eigenvalue) and additionally on a dense grid of [0, Lambda] to get
C_k = max_{[0,Lambda]} |p_k|.

Methods: GD (eta=1/L), GD large step (1.9/L), GD with linear warm-up (200 steps) , heavy ball beta=0.9 (eta=1/L),
heavy ball tuned to [a,L], Nesterov tuned, Chebyshev semi-iteration on [a,L], conjugate gradients.
(a = lambda_min^+(H), L = lambda_max(H)).

Outputs ../results/linear_speedlimit.json:
  A. bound check (Thm 5): sqrt(2 L1_k) >= RHS_k at every step, every method, seeds 0-9, rho in RHO_SMALL
  B. recovery scaling: T_eps = last step with L1_k >= eps*F_inf or L2_k >= eps*L2_0 (eps=0.1), rho in RHO_SCAL
  C. depth at matched new-task progress: max_{j<=k_q} L1_j / F_inf where k_q first step with L2 <= q L2_0
"""
import numpy as np, json, os, time, sys
from scipy.linalg import eigh
here = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(here, "..", "results")
sys.path.insert(0, here)
from linear_theory_lib import make_tasks, null_proj
d = 100; EPS = 0.1
RHO_SMALL = [1e-4, 3e-4, 1e-3, 3e-3]
RHO_SCAL = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
QS = [0.5, 0.1, 1e-2, 1e-3]


def setup(seed):
    S1, S2 = make_tasks(False, seed)
    e0 = null_proj(S1) @ np.random.default_rng(200 + seed).standard_normal(d); e0 /= np.linalg.norm(e0)
    return S1, S2, e0


def method_coeffs(name, L, a):
    """returns a step function (p_prev, p, k, lam) -> p_next for a fixed-coefficient method"""
    if name == "GD":
        eta = 1 / L; return lambda pp, p, k, lam: (1 - eta * lam) * p
    if name == "GD_large":
        eta = 1.9 / L; return lambda pp, p, k, lam: (1 - eta * lam) * p
    if name == "GD_warmup":
        return lambda pp, p, k, lam: (1 - min(1.0, (k + 1) / 200) / L * lam) * p
    if name == "HB_0.9":
        eta, b = 1 / L, 0.9; return lambda pp, p, k, lam: (1 + b - eta * lam) * p - b * pp
    if name == "HB_tuned":
        eta = 4 / (np.sqrt(L) + np.sqrt(a)) ** 2; b = ((np.sqrt(L) - np.sqrt(a)) / (np.sqrt(L) + np.sqrt(a))) ** 2
        return lambda pp, p, k, lam: (1 + b - eta * lam) * p - b * pp
    if name == "Nesterov":
        eta = 1 / L; q = np.sqrt(a / L); b = (1 - q) / (1 + q)
        return lambda pp, p, k, lam: (1 - eta * lam) * ((1 + b) * p - b * pp)
    raise ValueError(name)


def cheb_vals(k, lam, L, a):
    """p_k(lam) = T_k(z)/T_k(s0), z=(L+a-2lam)/(L-a), s0=(L+a)/(L-a), overflow-safe"""
    z = (L + a - 2 * lam) / (L - a); s0 = (L + a) / (L - a); A0 = np.arccosh(s0)
    out = np.empty_like(lam)
    m = np.abs(z) <= 1
    out[m] = np.cos(k * np.arccos(z[m])) * 2 * np.exp(-k * A0) / (1 + np.exp(-2 * k * A0))
    zz = z[~m]; sg = np.sign(zz) ** k; Az = np.arccosh(np.abs(zz))
    out[~m] = sg * np.exp(k * (Az - A0)) * (1 + np.exp(-2 * k * Az)) / (1 + np.exp(-2 * k * A0))
    return out


def run_poly(name, lam, c0, A1, A2, L, a, K, grid=None, slow=None):
    """simulate p_k on spectrum lam (and on grid); returns L1, L2 arrays (len K+1), p on spectrum at all k (subsampled), C_k"""
    lam_all = lam if grid is None else np.concatenate([lam, grid]); n = len(lam)
    L1 = np.empty(K + 1); L2 = np.empty(K + 1); C = np.empty(K + 1); Ps = []
    if name == "CG":
        # run CG in the eigenbasis on the actual problem, record alpha/beta, replay polynomial on lam_all
        x = c0.copy(); g = lam * x; dvec = -g.copy(); pa = np.ones_like(lam_all); qa = -lam_all * pa
        for k in range(K + 1):
            v = pa[:n] * c0; L1[k] = 0.5 * v @ A1 @ v; L2[k] = 0.5 * v @ A2 @ v
            C[k] = np.abs(pa[n:]).max() if grid is not None else np.nan; Ps.append((np.abs(1 - pa[slow]).max(), np.abs(pa[len(slow):n]).max()) if slow is not None else (0.0, 0.0))
            if k == K: break
            gg = g @ g
            if gg < 1e-30:      # converged: freeze
                continue
            alpha = gg / (dvec @ (lam * dvec)); x = x + alpha * dvec; gn = g + alpha * lam * dvec
            beta = (gn @ gn) / gg; dvec = -gn + beta * dvec; g = gn
            pa = pa + alpha * qa; qa = -lam_all * pa + beta * qa
        return L1, L2, np.array(Ps, dtype=float).reshape(-1, 2), C
    if name == "Cheb":
        for k in range(K + 1):
            pa = cheb_vals(k, lam_all, L, a); v = pa[:n] * c0
            L1[k] = 0.5 * v @ A1 @ v; L2[k] = 0.5 * v @ A2 @ v
            C[k] = np.abs(pa[n:]).max() if grid is not None else np.nan; Ps.append((np.abs(1 - pa[slow]).max(), np.abs(pa[len(slow):n]).max()) if slow is not None else (0.0, 0.0))
        return L1, L2, np.array(Ps, dtype=float).reshape(-1, 2), C
    step = method_coeffs(name, L, a); pp = np.ones_like(lam_all); p = np.ones_like(lam_all)
    for k in range(K + 1):
        v = p[:n] * c0; L1[k] = 0.5 * v @ A1 @ v; L2[k] = 0.5 * v @ A2 @ v
        C[k] = np.abs(p[n:]).max() if grid is not None else np.nan; Ps.append((np.abs(1 - p[slow]).max(), np.abs(p[len(slow):n]).max()) if slow is not None else (0.0, 0.0))
        pn = step(pp, p, k, lam_all); pp, p = p, pn
    return L1, L2, np.array(Ps, dtype=float).reshape(-1, 2), C


def last_violation(L1, L2, F, L20):
    bad = np.where((L1 >= EPS * F) | (L2 >= EPS * L20))[0]
    return int(bad[-1]) + 1 if len(bad) else 0


def gd_closed_T(lam, c0, A1, A2, eta, F, L20, kmax):
    """GD has p_k = (1-eta lam)^k; find last violation by scanning a geometric grid then bisection"""
    def LL(k):
        v = (1 - eta * lam) ** k * c0; return 0.5 * v @ A1 @ v, 0.5 * v @ A2 @ v
    viol = lambda k: (lambda l: l[0] >= EPS * F or l[1] >= EPS * L20)(LL(k))
    ks = np.unique(np.geomspace(1, kmax, 3000).astype(np.int64)); vs = np.array([viol(k) for k in ks])
    if not vs.any(): return 0
    j = np.where(vs)[0][-1]
    if j == len(ks) - 1: return np.inf
    lo, hi = ks[j], ks[j + 1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if viol(mid): lo = mid
        else: hi = mid
    return int(hi)



def companion(name, L, a, lam):
    if name == "HB_0.9":
        eta, b = 1 / L, 0.9; t = 1 + b - eta * lam; u = -b * np.ones_like(lam)
    elif name == "HB_tuned":
        eta = 4 / (np.sqrt(L) + np.sqrt(a)) ** 2; b = ((np.sqrt(L) - np.sqrt(a)) / (np.sqrt(L) + np.sqrt(a))) ** 2
        t = 1 + b - eta * lam; u = -b * np.ones_like(lam)
    elif name == "HB_rho":
        eta, b = 1 / L, 1 - np.sqrt(RHO_CUR[0]); t = 1 + b - eta * lam; u = -b * np.ones_like(lam)
    elif name == "Nesterov":
        eta = 1 / L; q = np.sqrt(a / L); b = (1 - q) / (1 + q); t = (1 - eta * lam) * (1 + b); u = -(1 - eta * lam) * b
    M = np.zeros((len(lam), 2, 2)); M[:, 0, 0] = t; M[:, 0, 1] = u; M[:, 1, 0] = 1
    return M


def pk_closed(name, k, lam, L, a):
    """p_k(lam) for fixed-coefficient methods, any integer k (state (p_k, p_{k-1}), p_0 = p_{-1} = 1)"""
    if name == "GD": return (1 - lam / L) ** k
    if name == "Cheb": return cheb_vals(k, lam, L, a)
    M = companion(name, L, a, lam); Mk = np.linalg.matrix_power(M, int(k))
    return Mk[:, 0, 0] + Mk[:, 0, 1]


def T_closed(name, lam, c0, A1, A2, L, a, F, L20, kmax):
    def viol(k):
        v = pk_closed(name, k, lam, L, a) * c0
        return (0.5 * v @ A1 @ v >= EPS * F) or (0.5 * v @ A2 @ v >= EPS * L20)
    ks = np.unique(np.concatenate([np.arange(0, 2000), np.geomspace(2000, kmax, 20000).astype(np.int64)]))
    vs = np.array([viol(k) for k in ks])
    if not vs.any(): return 0
    j = np.where(vs)[0][-1]
    if j == len(ks) - 1: return np.inf
    lo, hi = int(ks[j]), int(ks[j + 1])
    for k in range(hi - 1, lo, -1):          # dense refinement between the two grid points
        if viol(k): return k + 1
    return lo + 1


METHODS = ["GD", "GD_large", "GD_warmup", "HB_0.9", "HB_tuned", "Nesterov", "Cheb", "CG"]
B_METHODS = os.environ.get("B_METHODS", "GD,HB_0.9,HB_tuned,Nesterov,Cheb,CG").split(","); RHO_CUR = [None]
B_ONLY = os.environ.get("B_ONLY") == "1"
RHO_AC = RHO_SMALL + [1e-2, 1e-1]
out = dict(A=[], B=[], C=[], meta=dict(eps=EPS, methods=METHODS)); t0 = time.time()
NSEED_B = int(os.environ.get("NSEED_B", 10)); NSEED = int(os.environ.get("NSEED", 10))
for seed in [int(x) for x in os.environ.get('SEEDS', ','.join(map(str, range(NSEED)))).split(',')]:
    S1, S2, e0 = setup(seed)
    l1max = eigh(S1, eigvals_only=True)[-1]
    lam2 = eigh(S2, eigvals_only=True); mu = lam2[lam2 > 1e-9 * lam2[-1]].min()
    PN = null_proj(S2); z = PN @ e0; F = 0.5 * z @ S1 @ z; dimN = int(round(np.trace(PN)))
    lamS_ = eigh(S1 + S2, eigvals_only=True); nu = lamS_[lamS_ > 1e-9 * lamS_[-1]].min()
    for rho in sorted(set(RHO_AC + RHO_SCAL)):
        H = (1 - rho) * S2 + rho * S1; lam, Q = eigh(H); lam = np.clip(lam, 0, None)
        L = lam[-1]; pos = lam > 1e-10 * L; a = lam[pos].min(); lam[~pos] = 0.0
        c0 = Q.T @ e0; A1 = Q.T @ S1 @ Q; A2 = Q.T @ S2 @ Q; L20 = 0.5 * e0 @ S2 @ e0
        slow = np.arange(dimN); lamS = lam[slow].max(); gapok = rho * l1max < (1 - rho) * mu
        PS = Q[:, slow] @ Q[:, slow].T
        g_rho = np.sqrt(e0 @ PS @ S1 @ PS @ e0); nPS = np.linalg.norm(PS @ e0)
        QF = Q[:, dimN:]
        KF = np.sqrt(eigh(QF.T @ S1 @ QF, QF.T @ S2 @ QF, eigvals_only=True)[-1]) if gapok else np.nan
        RHO_CUR[0] = rho
        if rho in RHO_AC and not B_ONLY:
            acc_bound = gapok; grid = np.linspace(0, L, 2001)
            for name in METHODS:
                K = 400 if name == "CG" else 20000
                L1, L2, sig_phi, C = run_poly(name, lam, c0, A1, A2, L, a, K, grid=grid if acc_bound else None,
                                          slow=slow if acc_bound else None)
                rowC = dict(seed=seed, rho=rho, method=name, F=float(F), G_over_F_window=float(L1.max() / F),
                            k_peak=int(L1.argmax()), K=K)
                for q in QS:
                    kq = np.where(L2 <= q * L20)[0]
                    rowC[f"Gq{q}"] = float(L1[:kq[0] + 1].max() / F) if len(kq) else None
                    rowC[f"kq{q}"] = int(kq[0]) if len(kq) else None
                out["C"].append(rowC)
                if acc_bound:
                    sig = sig_phi[:, 0]; phi = sig_phi[:, 1]; L2c = np.maximum(L2, 0)
                    nPF = np.linalg.norm(e0 - PS @ e0)
                    rhs = g_rho - np.sqrt(l1max) * sig * nPS - KF * (np.sqrt(2 * L2c) + (1 + sig) * np.sqrt(lamS / (1 - rho)) * nPS)
                    rhs2 = g_rho - np.sqrt(l1max) * (sig * nPS + phi * nPF)
                    lhs = np.sqrt(2 * np.maximum(L1, 0)); ks = np.arange(len(L1)); mk = ks > 0
                    markov = 2 * C[mk] * ks[mk] ** 2 * lamS / L
                    kq = np.where(L2 <= 1e-3 * L20)[0]; kL = int(kq[0]) if len(kq) else None
                    pick = lambda arr: (float(arr[kL]) if kL is not None else None)
                    out["A"].append(dict(seed=seed, rho=rho, method=name, steps=len(L1),
                                         viol_thmA=int((lhs < rhs - 1e-12).sum()), viol_thmA2=int((lhs < rhs2 - 1e-12).sum()),
                                         g_rho=float(g_rho), sqrt2F=float(np.sqrt(2 * F)), KF=float(KF), nPS=float(nPS), nPF=float(nPF),
                                         sqrt_l1max=float(np.sqrt(l1max)), lamS=float(lamS), L=float(L),
                                         markov_ok=bool(np.all(sig[mk] <= markov + 1e-12)), Cmax=float(np.nanmax(C)),
                                         kL=kL, L1_at_kL_over_F=pick(L1 / F), depth_to_kL_over_F=(float(L1[:kL + 1].max() / F) if kL is not None else None),
                                         sigma_at_kL=pick(sig), phi_at_kL=pick(phi), C_at_kL=pick(C),
                                         bound2_at_kL_over_F=(float(max(rhs2[kL], 0) ** 2 / 2 / F) if kL is not None else None),
                                         markov_sigma_at_kL=(float(2 * C[kL] * kL ** 2 * lamS / L) if kL else None),
                                         g2_over_2F=float(g_rho ** 2 / 2 / F)))
        if rho in RHO_SCAL and seed < NSEED_B:
            for name in B_METHODS:
                if name == "CG":
                    L1, L2, _, _ = run_poly("CG", lam, c0, A1, A2, L, a, 400); T = last_violation(L1, L2, F, L20)
                    T = T if T <= 400 else np.inf
                else:
                    T = T_closed(name, lam, c0, A1, A2, L, a, F, L20, int(1e10))
                out["B"].append(dict(seed=seed, rho=rho, method=name, T_eps=float(T), L_over_a=float(L / a), a=float(a), L=float(L),
                                     lamS=float(lamS), l1max=float(l1max), nu=float(nu), F=float(F), gapok=bool(gapok)))
        print(seed, rho, f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open(os.path.join(R, os.environ.get("OUT", "linear_speedlimit.json")), "w"), default=float)
print("done", f"{time.time()-t0:.0f}s")
