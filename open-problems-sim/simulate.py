"""Numerical experiments for ten famous open problems.

Each function returns a dict of data. None of this proves anything: the
experiments reproduce the conjectured behaviour in small, finite ranges.
Run:  python simulate.py   (writes results.json and figures.png, ~5-10 min)
"""
import json, math, random, time
import numpy as np

RNG = np.random.default_rng(0)
random.seed(0)


def sieve(N):
    s = np.ones(N + 1, dtype=bool); s[:2] = False
    for i in range(2, int(N ** .5) + 1):
        if s[i]: s[i * i::i] = False
    return s


# 1 ------------------------------------------------------------------ Riemann
def riemann(nzeros=150, X=10 ** 7):
    from mpmath import mp, zetazero, zeta, gamma, pi, log, taylor
    mp.dps = 25
    g = [float(zetazero(n).imag) for n in range(1, nzeros + 1)]
    g = np.array(g)
    # spacings normalised by the local mean density log(t/2pi)/2pi
    sp = np.diff(g) * np.log(g[:-1] / (2 * np.pi)) / (2 * np.pi)
    # Chebyshev psi(x) - x against sqrt(x)
    s = sieve(X); lam = np.zeros(X + 1)
    for p in np.nonzero(s)[0]:
        q = int(p)
        while q <= X:
            lam[q] = math.log(p); q *= int(p)
    psi = np.cumsum(lam); x = np.arange(X + 1)
    err = np.abs(psi[2:] - x[2:]) / np.sqrt(x[2:])
    samp = np.unique(np.logspace(1, math.log10(X), 400).astype(int))
    # Li coefficients (RH <=> all positive)
    mp.dps = 80
    xi = lambda w: (w - 1) * pi ** (-w / 2) * gamma(w / 2 + 1) * zeta(w)
    c = taylor(lambda z: log(2 * xi(-z / (1 - z))), 0, 16)
    li = [float(n * c[n]) for n in range(1, 17)]
    return dict(zeros=g.tolist(), spacings=sp.tolist(), mean_spacing=float(sp.mean()),
                min_spacing=float(sp.min()), psi_x=samp.tolist(),
                psi_err=[float((psi[v] - v) / math.sqrt(v)) for v in samp],
                max_psi_err_over_sqrt=float(err.max()),
                rh_bound_at_X=float(math.log(X) ** 2 / (8 * math.pi)), li=li)


# 2 ------------------------------------------------------------------ P vs NP
def _dpll(clauses, assign, stats):
    # unit propagation
    while True:
        unit = None; new = []
        for c in clauses:
            sat = False; rem = []
            for l in c:
                v = assign.get(abs(l))
                if v is None: rem.append(l)
                elif v == (l > 0): sat = True; break
            if sat: continue
            if not rem: return False
            if len(rem) == 1 and unit is None: unit = rem[0]
            new.append(rem)
        clauses = new
        if unit is None: break
        assign[abs(unit)] = unit > 0
    if not clauses: return True
    stats[0] += 1
    l = min(clauses, key=len)[0]
    for val in (l > 0, l < 0):
        a = dict(assign); a[abs(l)] = val
        if _dpll(clauses, a, stats): return True
    return False


def p_vs_np(n=50, trials=40):
    out = []
    for r in [3.0, 3.5, 3.8, 4.0, 4.26, 4.5, 4.8, 5.0, 5.5, 6.0, 7.0]:
        sat = 0; br = []
        for _ in range(trials):
            cl = [[v * random.choice((1, -1)) for v in random.sample(range(1, n + 1), 3)]
                  for _ in range(int(r * n))]
            st = [0]; sat += _dpll(cl, {}, st); br.append(st[0])
        out.append(dict(ratio=r, sat_fraction=sat / trials, median_branches=float(np.median(br)),
                        max_branches=int(max(br))))
    # exact De Morgan formula size of every 4-variable function
    n4 = 4; M = 1 << 16; FULL = M - 1
    xs = [sum(1 << r for r in range(16) if (r >> i) & 1) for i in range(n4)]
    size = np.zeros(M, dtype=np.int8); size[[0, FULL]] = -1
    levels = {1: np.array(sorted(set(xs + [FULL ^ x for x in xs])), dtype=np.uint32)}
    size[levels[1]] = 1; found = 10; s = 1
    while found < M:
        s += 1; new = np.zeros(M, dtype=bool)
        for a in range(1, s // 2 + 1):
            for g in levels[a]: new[g & levels[s - a]] = True
        idx = np.nonzero(new)[0]; idx = idx[size[idx] == 0]
        idx = np.unique(np.concatenate([idx, FULL ^ idx])); idx = idx[size[idx] == 0]
        size[idx] = s; levels[s] = idx.astype(np.uint32); found += len(idx)
    par = xs[0] ^ xs[1] ^ xs[2] ^ xs[3]
    return dict(sat_n=n, sat=out, formula_size_hist={int(k): int(len(v)) for k, v in levels.items()},
                parity4_size=int(size[par]))


# 3 ------------------------------------------------------------ Navier-Stokes
def navier_stokes(N=64, nus=(0.01, 0.005, 0.0025), T=10.0, dt=0.01):
    import scipy.fft as sf
    k1 = np.fft.fftfreq(N, 1 / N); kz = np.fft.rfftfreq(N, 1 / N)
    KX, KY, KZ = np.meshgrid(k1, k1, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2; K2i = np.where(K2 == 0, 1, K2)
    mask = (np.abs(KX) < N / 3) & (np.abs(KY) < N / 3) & (np.abs(KZ) < N / 3)
    K = (KX, KY, KZ)
    f = lambda a: sf.rfftn(a, workers=4); b = lambda a: sf.irfftn(a, s=(N, N, N), workers=4)
    x = np.arange(N) * 2 * np.pi / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    def curl(uh):
        return (1j * (KY * uh[2] - KZ * uh[1]), 1j * (KZ * uh[0] - KX * uh[2]), 1j * (KX * uh[1] - KY * uh[0]))

    def rhs(uh, nu):
        u = [b(c) for c in uh]; w = [b(c) for c in curl(uh)]
        n = [f(u[1] * w[2] - u[2] * w[1]) * mask, f(u[2] * w[0] - u[0] * w[2]) * mask,
             f(u[0] * w[1] - u[1] * w[0]) * mask]
        div = (KX * n[0] + KY * n[1] + KZ * n[2]) / K2i
        return [n[i] - K[i] * div - nu * K2 * uh[i] for i in range(3)], w

    runs = []
    for nu in nus:
        uh = [f(np.sin(X) * np.cos(Y) * np.cos(Z)), f(-np.cos(X) * np.sin(Y) * np.cos(Z)), f(np.zeros_like(X))]
        ts, E, Om, wmax = [], [], [], []
        for step in range(int(T / dt) + 1):
            k1_, w = rhs(uh, nu)
            if step % 10 == 0:
                u = [b(c) for c in uh]
                ts.append(step * dt); E.append(float(0.5 * np.mean(sum(c * c for c in u))))
                Om.append(float(0.5 * np.mean(sum(c * c for c in w))))
                wmax.append(float(np.sqrt(sum(c * c for c in w)).max()))
            mid = [uh[i] + dt * k1_[i] for i in range(3)]
            k2_, _ = rhs(mid, nu)
            uh = [uh[i] + 0.5 * dt * (k1_[i] + k2_[i]) for i in range(3)]
        E_, Om_ = np.array(E), np.array(Om)
        # energy balance dE/dt = -2 nu * enstrophy
        resid = np.max(np.abs(np.gradient(E_, ts) + 2 * nu * Om_)[2:-2]) / np.max(2 * nu * Om_)
        runs.append(dict(nu=nu, Re=1 / nu, t=ts, energy=E, enstrophy=Om, max_vorticity=wmax,
                         peak_enstrophy=float(Om_.max()), t_peak=float(ts[int(Om_.argmax())]),
                         peak_max_vorticity=float(max(wmax)), energy_balance_rel_error=float(resid)))
    return dict(N=N, runs=runs)


# 4 --------------------------------------------------------------- Yang-Mills
def _qmul(a, b):
    r = np.empty(np.broadcast(a, b).shape)
    r[..., 0] = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    r[..., 1:] = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] - np.cross(a[..., 1:], b[..., 1:])
    return r


def _dag(a):
    r = a.copy(); r[..., 1:] *= -1; return r


def yang_mills(L=8, betas=(0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 3.0), therm=80, meas=120):
    sh = lambda A, mu, n=1: np.roll(A, -n, axis=mu)
    par = np.indices((L,) * 4).sum(0) % 2

    def staple(U, mu):
        A = np.zeros((L,) * 4 + (4,))
        for nu in range(4):
            if nu == mu: continue
            A += _qmul(_qmul(sh(U[nu], mu), _dag(sh(U[mu], nu))), _dag(U[nu]))
            Un = sh(U[nu], nu, -1)
            A += _qmul(_qmul(_dag(sh(Un, mu)), _dag(sh(U[mu], nu, -1))), Un)
        return A

    def line(U, mu, R):
        P = U[mu]
        for i in range(1, R): P = _qmul(P, sh(U[mu], mu, i))
        return P

    def wloop(U, R, T):
        tot = 0; cnt = 0
        for mu in range(4):
            for nu in range(4):
                if mu == nu or (R == T and mu > nu): continue
                a, c = line(U, mu, R), line(U, nu, T)
                W = _qmul(_qmul(a, sh(c, mu, R)), _qmul(_dag(sh(a, nu, T)), _dag(c)))
                tot += W[..., 0].mean(); cnt += 1
        return tot / cnt

    out = []
    for beta in betas:
        U = np.zeros((4,) + (L,) * 4 + (4,)); U[..., 0] = 1
        eps = min(0.9, 1.2 / math.sqrt(beta)); acc = []
        W = {k: [] for k in [(1, 1), (1, 2), (2, 2)]}
        for sweep in range(therm + meas):
            for mu in range(4):
                for p in (0, 1):
                    A = staple(U, mu); m = par == p
                    for hit in range(6):
                        r = np.concatenate([np.ones((L,) * 4 + (1,)), eps * RNG.normal(size=(L,) * 4 + (3,))], -1)
                        r /= np.linalg.norm(r, axis=-1, keepdims=True)
                        new = _qmul(r, U[mu])
                        dS = beta * (_qmul(new, A)[..., 0] - _qmul(U[mu], A)[..., 0])
                        ok = m & (RNG.random((L,) * 4) < np.exp(np.minimum(dS, 0)))
                        U[mu][ok] = new[ok]; acc.append(ok.sum() / m.sum())
            if sweep >= therm:
                for k in W: W[k].append(wloop(U, *k))
        w = {k: float(np.mean(v)) for k, v in W.items()}
        chi = -math.log(w[(2, 2)] * w[(1, 1)] / w[(1, 2)] ** 2) if w[(2, 2)] > 0 else None
        out.append(dict(beta=beta, plaquette=w[(1, 1)], plaquette_err=float(np.std(W[(1, 1)]) / math.sqrt(meas / 10)),
                        W12=w[(1, 2)], W22=w[(2, 2)], creutz_ratio_22=chi,
                        strong_coupling=beta / 4, weak_coupling=1 - 3 / (4 * beta), acceptance=float(np.mean(acc))))
    return dict(L=L, group="SU(2)", data=out)


# 5 ---------------------------------------------------------------------- BSD
def bsd(P=50000):
    primes = np.nonzero(sieve(P))[0]
    curves = {"11a3 (rank 0)": (-1, 0, 0), "37a (rank 1)": (0, -1, 0),
              "389a (rank 2)": (1, -2, 0), "5077a (rank 3)": (0, -7, 6)}
    res = {}
    for name, (a2, a4, a6) in curves.items():
        pr = 1.0; ps, vals = [], []
        for p in primes:
            p = int(p)
            if p == 2:
                Np = 1 + sum((y * y + y - (x ** 3 + a2 * x * x + a4 * x + a6)) % 2 == 0 for x in range(2) for y in range(2))
            else:
                x = np.arange(p, dtype=np.int64)
                d = (1 + 4 * ((x * x % p * x + a2 * x * x + a4 * x + a6) % p)) % p
                sq = np.zeros(p, dtype=np.int8); sq[(x * x) % p] = 1
                Np = 1 + p + int(np.where(d == 0, 0, np.where(sq[d] == 1, 1, -1)).sum())
            pr *= Np / p; ps.append(p); vals.append(pr)
        ps, vals = np.array(ps), np.array(vals)
        sel = ps > 100
        slope = np.polyfit(np.log(np.log(ps[sel])), np.log(vals[sel]), 1)[0]
        keep = np.unique(np.logspace(0, math.log10(len(ps) - 1), 200).astype(int))
        res[name] = dict(p=ps[keep].tolist(), product=vals[keep].tolist(), fitted_exponent=float(slope))
    return res


# 6 -------------------------------------------------------------------- Hodge
def hodge():
    # Hodge numbers of a smooth degree-d hypersurface in P^(n+1) via the Jacobian ring
    def hyper(n, d):
        poly = np.array([1.0])
        for _ in range(n + 2): poly = np.convolve(poly, np.ones(d - 1))
        row = []
        for p in range(n + 1):
            deg = (n - p + 1) * d - (n + 2)
            prim = int(round(poly[deg])) if 0 <= deg < len(poly) else 0
            row.append(prim + (1 if 2 * p == n else 0))
        return row
    table = {f"dim {n}, degree {d}": hyper(n, d) for n, d in
             [(2, 3), (2, 4), (2, 5), (3, 5), (4, 3), (4, 4), (4, 6), (6, 3)]}
    # Explicit instance: the 48 lines on the Fermat quartic x0^4+x1^4+x2^4+x3^4=0
    z8 = [np.exp(1j * np.pi * (2 * k + 1) / 4) for k in range(4)]
    lines = []
    for (i, j), (k, l) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        for a in z8:
            for c in z8:
                f1 = np.zeros(4, complex); f1[i] = 1; f1[j] = -a
                f2 = np.zeros(4, complex); f2[k] = 1; f2[l] = -c
                lines.append(np.array([f1, f2]))
    G = np.zeros((48, 48))
    for i in range(48):
        for j in range(48):
            G[i, j] = -2 if i == j else float(abs(np.linalg.det(np.vstack([lines[i], lines[j]]))) < 1e-9)
    sv = np.linalg.svd(G, compute_uv=False)
    return dict(hypersurface_hodge_numbers=table, fermat_quartic_lines=48,
                line_gram_rank=int((sv > 1e-8).sum()), h11_of_K3=20,
                lines_met_by_each_line=int((G[0] == 1).sum()))


# 7, 8 ---------------------------------------------------- Goldbach and twins
def goldbach_twins(N=10 ** 7):
    s = sieve(N); sf = s.astype(np.float64); primes = np.nonzero(s)[0]
    C2 = float(np.prod(1 - 1.0 / (primes[1:80000] - 1.0) ** 2))

    def sing(n):
        r = 2 * C2; m = n; q = 3
        while m % 2 == 0: m //= 2
        while q * q <= m:
            if m % q == 0:
                r *= (q - 1) / (q - 2)
                while m % q == 0: m //= q
            q += 2
        if m > 2: r *= (m - 1) / (m - 2)
        return r
    reps = lambda n: int(np.dot(sf[2:n - 1], sf[n - 2:1:-1]))
    comet_n = list(range(4, 20001, 2)); comet = [reps(n) for n in comet_n]
    big = [2 * random.randrange(10 ** 5, N // 2) for _ in range(400)]
    ratios = []
    for n in big:
        k = np.arange(2, n - 1, dtype=np.float64)
        ratios.append(reps(n) / (sing(n) * np.sum(1 / (np.log(k) * np.log(n - k)))))
    tw = []
    for X in [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]:
        cnt = int(np.sum(s[3:X - 1] & s[5:X + 1]))
        t = np.linspace(2, X, 2000001); pred = 2 * C2 * np.trapezoid(1 / np.log(t) ** 2, t)
        tw.append(dict(X=X, twin_pairs=cnt, hardy_littlewood=float(pred), ratio=float(cnt / pred)))
    gaps = np.diff(primes); gv, gc = np.unique(gaps, return_counts=True)
    return dict(C2=C2, comet_n=comet_n, comet=comet, min_reps_4_to_20000=int(min(comet[4:])),
                goldbach_ratio_min=float(min(ratios)), goldbach_ratio_max=float(max(ratios)),
                goldbach_ratio_mean=float(np.mean(ratios)), twins=tw,
                gap_values=gv[:30].tolist(), gap_counts=gc[:30].tolist(), max_gap_below_N=int(gaps.max()))


# 9 ------------------------------------------------------------------ Collatz
def collatz(M=10 ** 6, samples=20000, bits=60):
    steps = np.zeros(M, dtype=np.int32); best = (0, 0); peak = (0, 0)
    for n in range(2, M):
        x = n; c = 0; mx = n
        while x >= n:
            if x & 1: x = 3 * x + 1; mx = max(mx, x)
            else: x >>= 1
            c += 1
            if x == 1: break
        steps[n] = c + (steps[x] if x > 1 else 0)
        if steps[n] > best[0]: best = (int(steps[n]), n)
        if mx / n > peak[0]: peak = (mx / n, n)
    odd = half = 0; tot = []
    for _ in range(samples):
        x = random.getrandbits(bits) | (1 << (bits - 1)) | 1; n0 = x; c = 0
        while x != 1:
            if x & 1: x = 3 * x + 1; odd += 1
            else: x >>= 1; half += 1
            c += 1
        tot.append(c / math.log(n0))
    return dict(verified_below=M, longest_orbit_steps=best[0], longest_orbit_start=best[1],
                highest_peak_ratio=float(peak[0]), highest_peak_start=peak[1],
                steps_first_10000=steps[:10000].tolist(),
                halvings_per_tripling_random_60bit=half / odd, break_even=math.log2(3),
                steps_over_log_n_mean=float(np.mean(tot)),
                steps_over_log_n_heuristic=3 / math.log(4 / 3),
                steps_over_log_n_hist=np.histogram(tot, bins=40)[0].tolist(),
                steps_over_log_n_edges=np.histogram(tot, bins=40)[1].tolist())


# 10 --------------------------------------------------------------------- abc
def abc(C=10 ** 5):
    rad = np.ones(C + 1, dtype=np.int64)
    for p in np.nonzero(sieve(C))[0]: rad[p::p] *= p
    hits = []
    for a in range(1, C // 2 + 1):
        b = np.arange(a + 1, C - a + 1)
        if len(b) == 0: break
        c = a + b
        ok = rad[a] * rad[b] < c          # necessary, cheap
        b, c = b[ok], c[ok]
        ok = rad[a] * rad[b] * rad[c] < c
        b, c = b[ok], c[ok]
        for bb, cc in zip(b, c):
            if math.gcd(a, int(bb)) == 1:
                q = math.log(cc) / math.log(int(rad[a] * rad[bb] * rad[cc]))
                hits.append((float(q), a, int(bb), int(cc)))
    hits.sort(reverse=True)
    cs = np.array([h[3] for h in hits])
    return dict(C=C, counts={str(10 ** k): int((cs < 10 ** k).sum()) for k in range(2, int(math.log10(C)) + 1)},
                known_counts={"100": 6, "1000": 31, "10000": 120, "100000": 418},
                top10=hits[:10], qualities=[h[0] for h in hits], cs=cs.tolist(),
                n_quality_above_1_4=int(sum(h[0] > 1.4 for h in hits)))


TASKS = [("riemann", riemann), ("p_vs_np", p_vs_np), ("navier_stokes", navier_stokes),
         ("yang_mills", yang_mills), ("bsd", bsd), ("hodge", hodge),
         ("goldbach_twins", goldbach_twins), ("collatz", collatz), ("abc", abc)]

if __name__ == "__main__":
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name, fn in TASKS:
        if sys.argv[1:] and name not in sys.argv[1:]: continue
        t = time.time(); res[name] = fn(); print(f"{name}: {time.time() - t:.1f}s", flush=True)
        json.dump(res, open(path, "w"))
