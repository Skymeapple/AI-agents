"""E5: a non-synthetic detectability distribution from mutation analysis.

Errors are first-order AST mutants of 8 small reference functions.  A cheap
verifier call runs k tests sampled from a test pool; a mutant's per-call
detectability is d = 1 - C(n-kills, k)/C(n, k) where kills is the number of pool
tests it fails.  We measure (i) the blind-spot mass rho (d=0), (ii) the tail of
d near 0 and the resulting re-verification curve mu_m, (iii) how a second,
differently-constructed test pool (diverse verifier) shrinks the joint blind
spot, and (iv) we check Thm 1 end-to-end by running the long-horizon simulator
with errors drawn from this empirical mutant pool.
"""
import ast, copy, json, random, sys
import numpy as np
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import theory as th
from simulator import DetSampler, run_many

RES = "../results/"
SRC = {
"gcd": """
def f(a, b):
    while b != 0:
        a, b = b, a % b
    return abs(a)
""",
"bsearch": """
def f(xs, t):
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == t:
            return mid
        elif xs[mid] < t:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
"rle": """
def f(s):
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append((s[i], j - i))
        i = j
    return out
""",
"isprime": """
def f(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
""",
"merge": """
def f(a, b):
    i, j, out = 0, 0, []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + a[i:] + b[j:]
""",
"maxsub": """
def f(xs):
    best, cur = 0, 0
    for x in xs:
        cur = max(0, cur + x)
        if cur > best:
            best = cur
    return best
""",
"clamp_sum": """
def f(xs, lo, hi):
    s = 0
    for x in xs:
        if x < lo:
            s += lo
        elif x > hi:
            s += hi
        else:
            s += x
    return s
""",
"leap": """
def f(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    return y % 4 == 0
""",
}

rng = random.Random(0)


def gen_inputs(name, pool):
    """pool 'A': typical random inputs; pool 'B': edge-case-oriented inputs."""
    R = rng
    out = []
    for _ in range(200):
        if name == "gcd":
            out.append((R.randint(1, 500), R.randint(1, 500)) if pool == "A"
                       else (R.choice([0, 1, -7, 12, 2**10, R.randint(-50, 50)]), R.choice([0, 1, 5, -3, R.randint(-50, 50)])))
        elif name == "bsearch":
            xs = sorted(R.sample(range(1000), R.randint(5, 30))) if pool == "A" else sorted(R.sample(range(10), R.randint(0, 3)))
            t = R.choice(xs) if (xs and R.random() < 0.5) else R.randint(-5, 1005 if pool == "A" else 12)
            out.append((xs, t))
        elif name == "rle":
            s = "".join(R.choice("ab") for _ in range(R.randint(5, 20))) if pool == "A" else "".join(R.choice("a") for _ in range(R.randint(0, 2)))
            out.append((s,))
        elif name == "isprime":
            out.append((R.randint(2, 400),) if pool == "A" else (R.choice([-3, 0, 1, 2, 3, 4, 9, 25, 49]),))
        elif name == "merge":
            if pool == "A":
                a = sorted(R.choices(range(50), k=R.randint(3, 10))); b = sorted(R.choices(range(50), k=R.randint(3, 10)))
            else:
                a = sorted(R.choices(range(3), k=R.randint(0, 2))); b = sorted(R.choices(range(3), k=R.randint(0, 2)))
            out.append((a, b))
        elif name == "maxsub":
            out.append(([R.randint(-10, 10) for _ in range(R.randint(5, 15))],) if pool == "A"
                       else ([R.choice([-1, 0, 1]) for _ in range(R.randint(0, 3))],))
        elif name == "clamp_sum":
            if pool == "A":
                out.append(([R.randint(-20, 20) for _ in range(10)], -5, 5))
            else:
                lo = R.randint(-2, 2)
                out.append(([R.choice([lo - 1, lo, lo + 1]) for _ in range(R.randint(0, 3))], lo, lo + R.randint(0, 2)))
        elif name == "leap":
            out.append((R.randint(1, 3000),) if pool == "A" else (R.choice([1900, 2000, 2004, 2100, 2400, 1, 4, 100, 400]),))
    return out


SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
        ast.Mod: ast.FloorDiv, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.And: ast.Or, ast.Or: ast.And}


def mutants(src):
    tree = ast.parse(src)
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in SWAP:
            sites.append(("op", node))
        elif isinstance(node, ast.Compare):
            for i, o in enumerate(node.ops):
                if type(o) in SWAP:
                    sites.append(("cmp%d" % i, node))
        elif isinstance(node, ast.BoolOp) and type(node.op) in SWAP:
            sites.append(("op", node))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            sites.append(("c+1", node)); sites.append(("c-1", node))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
            sites.append(("negb", node))
    out = []
    for idx, (kind, _) in enumerate(sites):
        t2 = ast.parse(src)
        s2 = []
        for node in ast.walk(t2):  # same traversal order -> same site list
            if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in SWAP:
                s2.append(("op", node))
            elif isinstance(node, ast.Compare):
                for i, o in enumerate(node.ops):
                    if type(o) in SWAP:
                        s2.append(("cmp%d" % i, node))
            elif isinstance(node, ast.BoolOp) and type(node.op) in SWAP:
                s2.append(("op", node))
            elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                s2.append(("c+1", node)); s2.append(("c-1", node))
            elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                s2.append(("negb", node))
        k, n = s2[idx]
        if k == "op":
            n.op = SWAP[type(n.op)]()
        elif k.startswith("cmp"):
            i = int(k[3:]); n.ops[i] = SWAP[type(n.ops[i])]()
        elif k == "c+1":
            n.value = n.value + 1
        elif k == "c-1":
            n.value = n.value - 1
        elif k == "negb":
            n.value.value = not n.value.value
        out.append(ast.unparse(t2))
    return out


class Budget(Exception):
    pass


def run_limited(fn, args, limit=20000):
    cnt = [0]

    def tracer(frame, event, arg):
        if event == "line":
            cnt[0] += 1
            if cnt[0] > limit:
                raise Budget()
        return tracer
    sys.settrace(tracer)
    try:
        return ("ok", fn(*copy.deepcopy(args)))
    except Budget:
        return ("timeout", None)
    except Exception as e:
        return ("exc", type(e).__name__)
    finally:
        sys.settrace(None)


def compile_fn(src):
    ns = {}
    exec(compile(src, "<m>", "exec"), ns)
    return ns["f"]


records = []
for name, src in SRC.items():
    ref = compile_fn(src)
    pools = {P: gen_inputs(name, P) for P in "AB"}
    expected = {P: [run_limited(ref, a)[1] for a in pools[P]] for P in "AB"}
    for msrc in mutants(src):
        mf = compile_fn(msrc)
        kills = {}
        for P in "AB":
            kv = []
            for a, e in zip(pools[P], expected[P]):
                st, v = run_limited(mf, a)
                kv.append(not (st == "ok" and v == e))
            kills[P] = kv
        records.append(dict(func=name, src=msrc, kA=kills["A"], kB=kills["B"]))
print("mutants:", len(records))

n, k = 200, 2


def det(nk):
    return 1.0 - comb(n - nk, k) / comb(n, k)


# Mutants killed by no test in either pool were inspected manually (see paper,
# Sec. 6.5): all are semantically equivalent on integer inputs, hence not errors.
# They are excluded from the error population.
equiv = [r for r in records if not any(r["kA"]) and not any(r["kB"])]
EQUIV_SRC = [r["func"] + ": " + r["src"].replace("\n", " | ") for r in equiv]
records = [r for r in records if any(r["kA"]) or any(r["kB"])]
dA = np.array([det(sum(r["kA"])) for r in records])
dB = np.array([det(sum(r["kB"])) for r in records])
# union verifier: each call runs k tests from A and k tests from B
dAB = 1 - (1 - dA) * (1 - dB)
res = dict(n_mutants=len(records), n_pool=n, k_per_call=k,
           per_function={f: int(sum(r["func"] == f for r in records)) for f in SRC})
for lab, d in [("A", dA), ("B", dB), ("A+B", dAB)]:
    res[lab] = dict(rho=float(np.mean(d == 0)), mean_d=float(d.mean()),
                    frac_d_lt_0p05=float(np.mean((d > 0) & (d < 0.05))),
                    frac_d_lt_0p2=float(np.mean((d > 0) & (d < 0.2))))
# joint blind spot vs independence prediction
res["joint_blind_AB"] = res["A+B"]["rho"]
res["indep_pred_joint_blind"] = res["A"]["rho"] * res["B"]["rho"]
res["excluded_equivalent_mutants"] = EQUIV_SRC
res["n_mutants_generated"] = len(records) + len(EQUIV_SRC)

ms = np.unique(np.round(np.logspace(0, 3, 30)).astype(int))
curves = {lab: th.mu_empirical(d, ms).tolist() for lab, d in [("A", dA), ("B", dB), ("A+B", dAB)]}
# homogeneous comparator with same single-call miss rate as A
rA = 1 - curves["A"][0]
curves["homogeneous, same mu_1 as A"] = ((1 - rA) ** ms).tolist()
# fit atom + Beta to A's d distribution by matching mu_m on m in [1, 1000] (log loss)
from scipy.optimize import minimize
yA = np.array(curves["A"])


def loss(x):
    a, b, rho = np.exp(x[0]), np.exp(x[1]), 1 / (1 + np.exp(-x[2]))
    return np.sum((np.log(th.mu_beta(a, b, ms, rho)) - np.log(yA)) ** 2)


fit = minimize(loss, [np.log(0.5), 0.0, -3.0], method="Nelder-Mead", options=dict(maxiter=4000))
a_, b_, rho_ = np.exp(fit.x[0]), np.exp(fit.x[1]), 1 / (1 + np.exp(-fit.x[2]))
res["beta_atom_fit_A"] = dict(a=float(a_), b=float(b_), rho=float(rho_), loss=float(fit.fun))
curves["fit atom+Beta (A)"] = th.mu_beta(a_, b_, ms, rho_).tolist()
res["m"] = ms.tolist(); res["curves"] = curves
# intermediate-regime slope for A excluding atom
nz = dA[dA > 0]
mu_nz = th.mu_empirical(nz, ms)
sel = (ms >= 3) & (ms <= 100)
res["A_nonblind_loglog_slope_m3_100"] = float(np.polyfit(np.log(ms[sel]), np.log(mu_nz[sel]), 1)[0])

# end-to-end check of Thm 1 using the empirical mutant pool as error distribution
endto = {}
seed = 500
for lab, d in [("A", dA), ("A+B", dAB)]:
    for m in [1, 8]:
        seed += 1
        _, c = run_many(2000, seed, N=1000, T=20, p0=0.002, beta=0.0,
                        det=DetSampler("empirical", values=d), m=m, f=0.0, V=1, C=0.5, R=1)
        endto[f"{lab}, m={m}"] = dict(sim=float(c.mean()),
                                       se=float(np.sqrt(c.mean() * (1 - c.mean()) / 2000)),
                                       theory=float(th.p_clean_const(0.002, th.mu_empirical(d, [m])[0], 1000)),
                                       floor=float(th.floor_const(0.002, float(np.mean(d == 0)), 1000)))
res["end_to_end"] = endto
print(json.dumps({k: v for k, v in res.items() if k not in ("curves", "m")}, indent=1))
json.dump(res, open(RES + "e5_mutation.json", "w"), indent=1)
np.savez(RES + "e5_detectabilities.npz", dA=dA, dB=dB, dAB=dAB)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.linspace(0, 1, 41)
ax[0].hist([dA, dB], bins=bins, label=["pool A (typical inputs)", "pool B (edge cases)"], color=["#1f77b4", "#ff7f0e"])
ax[0].set_xlabel("per-call detectability d (k=2 sampled tests)"); ax[0].set_ylabel("# mutants")
ax[0].set_title(f"(a) {len(records)} mutants: blind spots at d=0"); ax[0].legend(fontsize=8)
for lab, sty in [("A", "-o"), ("B", "-s"), ("A+B", "-^"), ("homogeneous, same mu_1 as A", "--"), ("fit atom+Beta (A)", ":")]:
    ax[1].loglog(ms, curves[lab], sty, ms=3, label=lab)
ax[1].set_ylim(1e-4, 1.2); ax[1].set_xlabel("verifier calls m"); ax[1].set_ylabel(r"miss rate $\mu_m$")
ax[1].set_title("(b) re-verification curves from real mutants"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(RES + "fig_mutation.png", dpi=150)
