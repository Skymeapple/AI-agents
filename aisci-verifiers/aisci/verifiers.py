"""I1: a uniform verifier service.

Every verifier takes a candidate and returns a Verdict with pass/fail, a reason,
and the latency and (estimated) cost of the check. Agents call verifiers only
through `VerifierService.check`, which logs every call (task 1.2).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
import sympy as sp


@dataclass
class Verdict:
    verifier: str
    passed: bool
    reason: str
    latency_s: float = 0.0
    cost_usd: float = 0.0
    score: float | None = None  # optional confidence in [0, 1] for soft judges


class Verifier:
    name = "base"
    cost_per_call_usd = 0.0

    def _check(self, candidate: Any) -> tuple[bool, str, float | None]:
        raise NotImplementedError

    def check(self, candidate: Any) -> Verdict:
        t0 = time.perf_counter()
        try:
            ok, reason, score = self._check(candidate)
        except Exception as e:  # a crashing check never passes
            ok, reason, score = False, f"verifier error: {e!r}", 0.0
        return Verdict(self.name, bool(ok), reason, time.perf_counter() - t0,
                       self.cost_per_call_usd, score)


class PythonTestsVerifier(Verifier):
    """Runs candidate source code against assert-style tests in a subprocess."""
    name = "python_tests"
    cost_per_call_usd = 0.0001

    def __init__(self, timeout_s: float = 10.0):
        self.timeout_s = timeout_s

    def _check(self, candidate):
        src, tests = candidate["source"], candidate["tests"]
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src + "\n\n" + tests + "\nprint('ALL_TESTS_PASSED')\n")
            path = f.name
        try:
            r = subprocess.run([sys.executable, "-I", path], capture_output=True,
                               text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return False, "timeout", 0.0
        ok = r.returncode == 0 and "ALL_TESTS_PASSED" in r.stdout
        return ok, "tests passed" if ok else (r.stderr.strip().splitlines() or ["failed"])[-1], None


class SymbolicIdentityVerifier(Verifier):
    """Checks a claimed identity lhs == rhs exactly with sympy."""
    name = "symbolic_identity"
    cost_per_call_usd = 0.00001

    def _check(self, candidate):
        lhs, rhs = sp.sympify(candidate["lhs"]), sp.sympify(candidate["rhs"])
        diff = sp.simplify(lhs - rhs)
        if diff == 0:
            return True, "identity holds symbolically", None
        return False, f"difference simplifies to {diff}", None


class MatrixAlgorithmVerifier(Verifier):
    """Checks a bilinear matrix-multiplication algorithm (the kind AlphaEvolve finds).

    Candidate: {"n": int, "U": [r][n*n], "V": [r][n*n], "W": [n*n][r]} meaning
    m_k = (U_k . vec(A)) * (V_k . vec(B));  vec(C) = W @ m.
    Correct iff the induced tensor equals the matrix-multiplication tensor exactly.
    """
    name = "matmul_tensor"
    cost_per_call_usd = 0.00001

    def _check(self, c):
        n = int(c["n"])
        U, V, W = (np.array(c[k], dtype=float) for k in ("U", "V", "W"))
        r = U.shape[0]
        T = np.einsum("ri,rj,kr->ijk", U, V, W)
        target = np.zeros((n * n, n * n, n * n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # C[i,k] += A[i,j] * B[j,k]
                    target[i * n + j, j * n + k, i * n + k] = 1
        err = float(np.max(np.abs(T - target)))
        ok = err < 1e-9
        return ok, f"rank {r}, max tensor error {err:.2e}", None


class OptimumClaimVerifier(Verifier):
    """Checks a claim 'x* minimises f on a box' by local search + random probes."""
    name = "optimum_claim"
    cost_per_call_usd = 0.0005

    def __init__(self, probes: int = 2000, seed: int = 0):
        self.probes, self.seed = probes, seed

    def _check(self, c):
        f: Callable = c["f"]
        lo, hi = np.array(c["lo"], float), np.array(c["hi"], float)
        x = np.array(c["x"], float)
        fx = f(x)
        rng = np.random.default_rng(self.seed)
        pts = rng.uniform(lo, hi, size=(self.probes, len(lo)))
        best = min(f(p) for p in pts)
        tol = c.get("tol", 1e-6)
        ok = fx <= best + tol
        return ok, f"f(x*)={fx:.6g}, best probe={best:.6g}", None


class SoftJudge(Verifier):
    """A noisy judge (a simulation, a literature check, a learned critic).

    Wraps a scoring function returning a probability that the candidate is true.
    Used by RP2 / task 1.4 ensembles where no strong verifier exists.
    """

    def __init__(self, name: str, score_fn: Callable[[Any], float], threshold=0.5,
                 cost=0.0):
        self.name, self.score_fn, self.threshold = name, score_fn, threshold
        self.cost_per_call_usd = cost

    def _check(self, c):
        s = float(self.score_fn(c))
        return s >= self.threshold, f"score {s:.3f}", s


class EnsembleVerifier(Verifier):
    """Task 1.4: combine soft judges with a logistic model fitted on labelled cases."""
    name = "ensemble"

    def __init__(self, judges: list[SoftJudge]):
        self.judges = judges
        self.w = np.zeros(len(judges) + 1)
        self.cost_per_call_usd = sum(j.cost_per_call_usd for j in judges)

    def _features(self, c):
        s = np.array([min(max(j.score_fn(c), 1e-4), 1 - 1e-4) for j in self.judges])
        return np.concatenate([[1.0], np.log(s / (1 - s))])

    def fit(self, cases, labels, l2=1e-2, iters=2000, lr=0.1):
        X = np.array([self._features(c) for c in cases])
        y = np.array(labels, float)
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ w))
            w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        self.w = w
        return self

    def prob(self, c):
        return float(1 / (1 + math.exp(-self._features(c) @ self.w)))

    def _check(self, c):
        p = self.prob(c)
        return p >= 0.5, f"ensemble p={p:.3f}", p


@dataclass
class VerifierService:
    """Task 1.2: one entry point, with a call log for latency and cost."""
    verifiers: dict[str, Verifier] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)

    def register(self, v: Verifier):
        self.verifiers[v.name] = v
        return self

    def check(self, verifier: str, candidate: Any) -> Verdict:
        v = self.verifiers[verifier].check(candidate)
        self.log.append(asdict(v))
        return v

    def stats(self):
        out = {}
        for row in self.log:
            s = out.setdefault(row["verifier"], {"calls": 0, "latency_s": 0.0, "cost_usd": 0.0})
            s["calls"] += 1
            s["latency_s"] += row["latency_s"]
            s["cost_usd"] += row["cost_usd"]
        return out


def default_service() -> VerifierService:
    s = VerifierService()
    for v in (PythonTestsVerifier(), SymbolicIdentityVerifier(),
              MatrixAlgorithmVerifier(), OptimumClaimVerifier()):
        s.register(v)
    return s


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


class DimensionalVerifier(Verifier):
    """Checks a claimed physical formula has the target dimensions.

    Candidate: {"expr": "G*M*m/r**2", "symbols": {"G": "length**3/(mass*time**2)", ...},
                "target": "mass*length/time**2"}
    A cheap screen: it rejects dimensionally impossible laws but cannot confirm a law.
    """
    name = "dimensions"
    cost_per_call_usd = 0.00001

    def _check(self, c):
        from sympy.physics.units.systems.si import dimsys_SI
        from sympy.physics.units import Dimension
        base = {k: sp.Symbol(k) for k in ("length", "mass", "time", "current", "temperature", "amount_of_substance", "luminous_intensity")}

        def dim(s):
            return Dimension(sp.sympify(s, locals=base))
        subs = {sp.Symbol(k): dim(v).name for k, v in c["symbols"].items()}
        expr_dim = Dimension(sp.sympify(c["expr"]).subs(subs))
        a = dimsys_SI.get_dimensional_dependencies(expr_dim)
        b = dimsys_SI.get_dimensional_dependencies(dim(c["target"]))
        ok = a == b
        return ok, f"got {a}, want {b}", None


_default_service_v1 = default_service


def default_service() -> VerifierService:  # noqa: F811  (v1.1 adds the 5th checker)
    return _default_service_v1().register(DimensionalVerifier())
