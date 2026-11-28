"""Task 1.3: measure each verifier's false-pass and false-reject rates.

Known-correct and known-wrong candidates are generated for each verifier; the
false-pass rate is the share of wrong candidates a verifier accepts.
"""
from __future__ import annotations

import json
import random

import numpy as np

from .verifiers import default_service, wilson

STRASSEN = {
    "n": 2,
    # vec index: X11=0, X12=1, X21=2, X22=3
    "U": [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
          [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]],
    "V": [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
          [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
    # rows: C11, C12, C21, C22 ; columns: M1..M7
    "W": [[1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
          [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]],
}


def naive_matmul(n):
    U, V, W = [], [], [[0] * (n ** 3) for _ in range(n * n)]
    r = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                u = [0] * (n * n); u[i * n + j] = 1
                v = [0] * (n * n); v[j * n + k] = 1
                U.append(u); V.append(v); W[i * n + k][r] = 1
                r += 1
    return {"n": n, "U": U, "V": V, "W": W}


def perturb_matmul(c, rng):
    c = json.loads(json.dumps(c))
    key = rng.choice(["U", "V", "W"])
    i = rng.randrange(len(c[key])); j = rng.randrange(len(c[key][i]))
    c[key][i][j] += rng.choice([-1, 1])
    return c


IS_PRIME_OK = """
def is_prime(n):
    if n < 2: return False
    i = 2
    while i * i <= n:
        if n % i == 0: return False
        i += 1
    return True
"""
IS_PRIME_MUTANTS = [
    IS_PRIME_OK.replace("n < 2", "n < 1"),
    IS_PRIME_OK.replace("i * i <= n", "i * i < n"),
    IS_PRIME_OK.replace("i += 1", "i += 2"),
    IS_PRIME_OK.replace("i = 2", "i = 3"),
    IS_PRIME_OK.replace("n % i == 0", "n % i == 1"),
    IS_PRIME_OK.replace("return True", "return n % 2 == 1"),
    IS_PRIME_OK.replace("n < 2", "n < 3"),
    IS_PRIME_OK.replace("while i * i <= n", "while i * i <= n // 2"),
]
WEAK_TESTS = "assert is_prime(7)\nassert not is_prime(8)\nassert is_prime(13)\n"
STRONG_TESTS = """
def _ref(n):
    return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))
for _n in range(0, 600):
    assert is_prime(_n) == _ref(_n), _n
"""

IDENTITIES = [
    ("sin(x)**2 + cos(x)**2", "1"),
    ("(x + y)**2", "x**2 + 2*x*y + y**2"),
    ("(x**3 - y**3)", "(x - y)*(x**2 + x*y + y**2)"),
    ("exp(x)*exp(y)", "exp(x + y)"),
    ("cos(2*x)", "1 - 2*sin(x)**2"),
    ("binomial(n, 2)", "n*(n - 1)/2"),
    ("log(x*y)", "log(x) + log(y)"),  # false for general complex x,y: sympy must reject
    ("tan(x)", "sin(x)/cos(x)"),
]


def perturb_identity(rhs, rng):
    return f"({rhs}) + {rng.choice(['1', 'x', '2*y', 'x**2/3'])}"


def rastrigin(x):
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def needle(x):
    # broad bowl around 2, plus a narrow deep well at -3.7 that random probes rarely hit
    x = np.asarray(x)
    return float(np.sum((x - 2) ** 2) - 50 * np.exp(-np.sum((x + 3.7) ** 2) / 0.001))


def run(seed=0, n_wrong=40):
    rng = random.Random(seed)
    svc = default_service()
    rows = []

    def rec(verifier, label, truth, cand, variant=""):
        v = svc.check(verifier, cand)
        rows.append({"verifier": verifier + variant, "truth": truth, "passed": v.passed})

    # matrix multiplication algorithms
    rec("matmul_tensor", "strassen", True, STRASSEN)
    for n in (2, 3):
        rec("matmul_tensor", f"naive{n}", True, naive_matmul(n))
    for _ in range(n_wrong):
        rec("matmul_tensor", "perturbed", False, perturb_matmul(STRASSEN, rng))

    # code against weak and strong test suites
    for tests, tag in ((WEAK_TESTS, "[weak tests]"), (STRONG_TESTS, "[strong tests]")):
        rec("python_tests", "ok", True, {"source": IS_PRIME_OK, "tests": tests}, tag)
        for m in IS_PRIME_MUTANTS:
            rec("python_tests", "mutant", False, {"source": m, "tests": tests}, tag)

    # symbolic identities
    for lhs, rhs in IDENTITIES:
        truth = lhs != "log(x*y)"
        rec("symbolic_identity", "id", truth, {"lhs": lhs, "rhs": rhs})
        for _ in range(4):
            rec("symbolic_identity", "pert", False, {"lhs": lhs, "rhs": perturb_identity(rhs, rng)})

    # optimum claims: smooth landscape vs deceptive needle landscape
    for f, xtrue, xwrong, tag in (
        (rastrigin, [0, 0], [[0.995, 0.0], [1.99, 0.99], [0.0, -2.98]], "[rastrigin]"),
        (needle, [-3.7, -3.7], [[2, 2]], "[needle]"),
    ):
        base = {"f": f, "lo": [-5.12, -5.12], "hi": [5.12, 5.12], "tol": 1e-3}
        rec("optimum_claim", "true", True, {**base, "x": xtrue}, tag)
        for xw in xwrong:
            rec("optimum_claim", "wrong", False, {**base, "x": xw}, tag)

    report = {}
    for name in sorted({r["verifier"] for r in rows}):
        rs = [r for r in rows if r["verifier"] == name]
        wrong = [r for r in rs if not r["truth"]]
        right = [r for r in rs if r["truth"]]
        fp = sum(r["passed"] for r in wrong)
        fr = sum(not r["passed"] for r in right)
        report[name] = {
            "n_wrong": len(wrong), "false_pass": fp,
            "false_pass_rate": fp / len(wrong) if wrong else None,
            "false_pass_95ci": wilson(fp, len(wrong)),
            "n_right": len(right), "false_reject": fr,
            "false_reject_rate": fr / len(right) if right else None,
        }
    return report, svc.stats()


if __name__ == "__main__":
    rep, stats = run()
    print(json.dumps({"calibration": rep, "service_stats": stats}, indent=2))
