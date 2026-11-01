"""Tasks 2.2, 2.4, 2.5 and RP3: an executable research agent with a world model,
run for 200 cycles against a world whose ground truth is known.

The world is a random linear-Gaussian causal model (20 variables). The agent
does real statistics on freshly sampled data each cycle; only the world is
synthetic, so every claim can be scored exactly.

Roles (deterministic code standing in for LLM roles; no LLM API is available here):
  analyst    - samples data, tests an association, writes a data claim (code evidence)
  reader     - reads the 'literature': causal statements that are wrong 17.9% of the
               time (Kosmos's literature error rate), writes a literature claim (citation)
  synthesist - chains two active claims into a new causal claim (needs 2 supports)
  critic     - (RP3 arm) only allows a synthesis from two causal supports and tests its
               implied association on fresh data before it is written
  auditor    - (2.5 arm) re-tests old claims on fresh data and retracts failures;
               retraction marks derived claims suspect, which are re-tested first
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np

from .worldmodel import Claim, Evidence, WorldModel

LIT_ERROR = 0.179


@dataclass
class World:
    p: int = 20
    edge_p: float = 0.15
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        W = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(i + 1, self.p):
                if rng.random() < self.edge_p:
                    W[i, j] = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
        self.W = W
        A = np.linalg.inv(np.eye(self.p) - W.T)
        self.A = A
        S = A @ A.T
        d = np.sqrt(np.diag(S))
        self.corr = S / np.outer(d, d)
        R = (W != 0).astype(int)
        reach = R.copy()
        for _ in range(self.p):
            reach = ((reach + reach @ R) > 0).astype(int)
        self.anc = reach  # anc[i, j] = 1 iff directed path i -> ... -> j
        self.rng = np.random.default_rng(self.seed + 1)
        self.samples_drawn = 0

    def sample(self, n):
        self.samples_drawn += n
        e = self.rng.normal(size=(n, self.p))
        return e @ self.A.T

    def intervene(self, i, n):
        """do(X_i := fresh noise): cut X_i's incoming edges, so X_i only varies on its own."""
        self.samples_drawn += n
        W = self.W.copy()
        W[:, i] = 0
        A = np.linalg.inv(np.eye(self.p) - W.T)
        e = self.rng.normal(size=(n, self.p))
        return e @ A.T

    def associated(self, i, j, tol=0.05):
        return abs(self.corr[i, j]) > tol

    def causes(self, i, j):
        return bool(self.anc[i, j])


def assoc_test(world, i, j, n=200, alpha=0.01):
    X = world.sample(n)
    r = np.corrcoef(X[:, i], X[:, j])[0, 1]
    z = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(n - 3)
    p = math.erfc(abs(z) / math.sqrt(2))
    return p < alpha, float(r), float(p)


def interv_test(world, i, j, n=200, alpha=0.01):
    """Experimental test of 'X_i causally affects X_j': randomise X_i, look for a response in X_j."""
    X = world.intervene(i, n)
    r = np.corrcoef(X[:, i], X[:, j])[0, 1]
    z = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(n - 3)
    p = math.erfc(abs(z) / math.sqrt(2))
    return p < alpha, float(r), float(p)


def run(seed=0, cycles=200, critic=False, audit=False, audit_every=5, audit_size=10, interventional=False):
    rng = random.Random(seed)
    world = World(seed=seed)
    wm = WorldModel()
    meta: dict[str, dict] = {}  # claim id -> {"kind": "assoc"|"cause", "i", "j"}
    n = 0
    checks = 0
    curve = {"data": [], "literature": [], "synthesis": [], "all": []}

    def truth(cid):
        m = meta[cid]
        return world.associated(m["i"], m["j"]) if m["kind"] == "assoc" else world.causes(m["i"], m["j"])

    def add(ctype, kind, i, j, ev, deps=()):
        nonlocal n
        n += 1
        cid = f"{ctype[0]}{n}"
        verb = "is associated with" if kind == "assoc" else "causally affects"
        wm.add(Claim(cid, f"X{i} {verb} X{j}", ctype, 0.7, [ev], list(deps), t))
        meta[cid] = {"kind": kind, "i": i, "j": j}

    def active(kind=None):
        return [k for k, c in wm.claims.items() if c.status == "active" and (kind is None or meta[k]["kind"] == kind)]

    for t in range(1, cycles + 1):
        # analyst: 2 association tests
        for _ in range(2):
            i, j = rng.sample(range(world.p), 2)
            sig, r, p = assoc_test(world, i, j)
            checks += 1
            if sig:
                add("data", "assoc", min(i, j), max(i, j), Evidence("code", f"assoc_test(X{i},X{j}) r={r:.2f} p={p:.1e}"))
        # reader: 1 literature causal statement (true edge-path, or an error)
        i, j = sorted(rng.sample(range(world.p), 2))
        true_pairs = [(a, b) for a in range(world.p) for b in range(world.p) if world.causes(a, b)]
        if rng.random() < LIT_ERROR:
            a, b = rng.choice([(a, b) for a in range(world.p) for b in range(world.p) if a != b and not world.causes(a, b)])
        else:
            a, b = rng.choice(true_pairs)
        add("literature", "cause", a, b, Evidence("citation", f"paper-{rng.randrange(10**6)}"))
        # synthesist: 2 attempts per cycle
        for _ in range(2):
            pool = active("cause") if critic else active()
            pairs = [(x, y) for x in pool for y in pool if x != y and meta[x]["j"] == meta[y]["i"] and meta[x]["i"] != meta[y]["j"]]
            if not pairs:
                # also allow association chains in the naive arm: X~Y, Y~Z
                if not critic:
                    pairs = [(x, y) for x in pool for y in pool if x != y and meta[x]["kind"] == "assoc" and meta[y]["kind"] == "assoc"
                             and len({meta[x]["i"], meta[x]["j"]} & {meta[y]["i"], meta[y]["j"]}) == 1]
                if not pairs:
                    continue
            x, y = rng.choice(pairs)
            if meta[x]["j"] == meta[y]["i"]:
                a, b = meta[x]["i"], meta[y]["j"]
            else:
                shared = ({meta[x]["i"], meta[x]["j"]} & {meta[y]["i"], meta[y]["j"]}).pop()
                a = ({meta[x]["i"], meta[x]["j"]} - {shared}).pop()
                b = ({meta[y]["i"], meta[y]["j"]} - {shared}).pop()
            if a == b or any(meta[k]["i"] == a and meta[k]["j"] == b and wm.claims[k].type == "synthesis" for k in active()):
                continue
            if critic:
                sig, r, p = (interv_test if interventional else assoc_test)(world, a, b)
                checks += 1
                if not sig:
                    continue
            add("synthesis", "cause", a, b, Evidence("code", f"chain({x},{y})"), deps=(x, y))
        # auditor
        if audit and t % audit_every == 0:
            sus = [k for k, c in wm.claims.items() if c.status == "suspect"]
            pool = active()
            rng.shuffle(pool)
            for cid in (sus + pool)[:audit_size]:
                m = meta[cid]
                test = interv_test if (interventional and m["kind"] == "cause") else assoc_test
                sig, _, _ = test(world, m["i"], m["j"])  # association is necessary for both kinds; intervention decides causation
                checks += 1
                if not sig:
                    wm.retract(cid, "failed re-test", t)
                elif wm.claims[cid].status == "suspect":
                    wm.claims[cid].status = "active"
        for typ in ("data", "literature", "synthesis"):
            ks = [k for k in active() if wm.claims[k].type == typ]
            curve[typ].append(sum(truth(k) for k in ks) / len(ks) if ks else float("nan"))
        ks = active()
        curve["all"].append(sum(truth(k) for k in ks) / len(ks) if ks else float("nan"))

    return {
        "cycles": cycles, "claims_written": n, "active": len(active()),
        "acc": {typ: {c: round(curve[typ][c - 1], 4) for c in (10, 50, 200) if c <= cycles} for typ in curve},
        "statistical_tests": checks, "samples_drawn": world.samples_drawn,
        "traceability": wm.traceability(),
    }


def arms(seeds=range(10)):
    out = {}
    for name, kw in {"naive": {}, "naive+audit": {"audit": True},
                     "rp3_chain+critic": {"critic": True}, "rp3+audit": {"critic": True, "audit": True},
                     "rp3_interventional+audit": {"critic": True, "audit": True, "interventional": True}}.items():
        rs = [run(seed=s, **kw) for s in seeds]
        agg = {typ: {c: float(np.nanmean([r["acc"][typ][c] for r in rs])) for c in (10, 50, 200)} for typ in ("data", "literature", "synthesis", "all")}
        out[name] = {"accuracy": agg, "mean_tests": float(np.mean([r["statistical_tests"] for r in rs])),
                     "mean_active_claims": float(np.mean([r["active"] for r in rs])),
                     "traceability": min(r["traceability"] for r in rs)}
    return out


if __name__ == "__main__":
    print(json.dumps(arms(), indent=2))
