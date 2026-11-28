"""Build the synthetic claim bank that ``nexus loop`` replays offline.

A claim is something a paper asserts that a machine can check: an identity, an
algorithm, a piece of code. Turning real abstracts into such claims needs a
model, which this environment cannot reach, so the bank stands in for that
step the way ``data/agent_stream.json`` stands in for arXiv in autoscout.

What is real and what is planted:

* The candidates are real. Every one is a payload for a checker in
  ``aisci-verifiers``, generated true or false by construction, and the loop
  runs the actual checker on it.
* The attachment to papers is planted. Each claim is pinned to a work in the
  demo corpus, and each planted research direction is given a hidden yield --
  the share of its claims that are true. Yields are drawn independently of the
  direction's vocabulary, so nothing sciscout can score predicts them.
* Two directions are traps: low yield, with most claims checked only by the
  weak test suite that ``aisci.calibration`` measured passing six of eight
  wrong programs. A loop that believes every pass piles onto them.

The ``truth`` and ``planted_yield`` fields exist so the experiment can score
allocation policies. No decision in ``nexus.loop`` reads them.

Run: ``python scripts/make_claim_bank.py``
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aisci-verifiers"))

from aisci.calibration import (  # noqa: E402
    IS_PRIME_MUTANTS,
    IS_PRIME_OK,
    STRASSEN,
    STRONG_TESTS,
    WEAK_TESTS,
    naive_matmul,
    perturb_matmul,
)

CORPUS = ROOT / "data" / "demo_corpus.json"
OUT = ROOT / "data" / "nexus" / "claim_bank.json"
SEED = 20261128
CLAIMS_PER_DIRECTION = 48
YIELDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
TRAP_YIELDS = (0.10, 0.15)
TRAP_WEAK_SHARE = 0.8


def identity(true: bool, rng: random.Random) -> tuple[dict, str]:
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    lhs, rhs = rng.choice([
        (f"({a}*x + {b}*y)**2", f"{a * a}*x**2 + {2 * a * b}*x*y + {b * b}*y**2"),
        (f"(x + {a})*(x + {b})", f"x**2 + {a + b}*x + {a * b}"),
        (f"{a}*sin(x)**2 + {a}*cos(x)**2", f"{a}"),
        (f"(x**{a} - 1)", f"(x - 1)*({' + '.join(f'x**{k}' for k in range(a))})"),
    ])
    if not true:
        rhs = f"({rhs}) + {rng.choice(['1', 'x', '2*y', 'x**2/3'])}"
    return {"lhs": lhs, "rhs": rhs}, f"{lhs} == {rhs}"


def matmul(true: bool, rng: random.Random) -> tuple[dict, str]:
    base = rng.choice([STRASSEN, naive_matmul(2), naive_matmul(3)])
    c = json.loads(json.dumps(base))
    # Reordering the products leaves the algorithm correct and makes the
    # payload distinct.
    order = list(range(len(c["U"])))
    rng.shuffle(order)
    c["U"] = [c["U"][i] for i in order]
    c["V"] = [c["V"][i] for i in order]
    c["W"] = [[row[i] for i in order] for row in c["W"]]
    if not true:
        c = perturb_matmul(c, rng)
    return c, f"a rank-{len(c['U'])} bilinear algorithm multiplies {c['n']}x{c['n']} matrices"


def program(true: bool, rng: random.Random, tests: str) -> tuple[dict, str]:
    source = IS_PRIME_OK if true else rng.choice(IS_PRIME_MUTANTS)
    return {"source": source, "tests": tests}, "this implementation of is_prime is correct"


def make_claim(true: bool, kind: str, rng: random.Random) -> dict:
    if kind == "symbolic_identity":
        candidate, statement = identity(true, rng)
        return {"verifier": kind, "variant": "", "candidate": candidate, "statement": statement}
    if kind == "matmul_tensor":
        candidate, statement = matmul(true, rng)
        return {"verifier": kind, "variant": "", "candidate": candidate, "statement": statement}
    tests, variant = ((WEAK_TESTS, "[weak tests]") if kind == "weak"
                      else (STRONG_TESTS, "[strong tests]"))
    candidate, statement = program(true, rng, tests)
    return {"verifier": "python_tests", "variant": variant, "candidate": candidate,
            "statement": statement}


def build() -> dict:
    rng = random.Random(SEED)
    works_by_track: dict[str, list[str]] = defaultdict(list)
    for raw in json.loads(CORPUS.read_text()):
        works_by_track[raw["extra"]["track"]].append(raw["id"])

    tracks = sorted(works_by_track)
    yields = YIELDS[:]
    rng.shuffle(yields)
    planted = dict(zip(tracks, yields))

    claims = []
    for track in tracks:
        trap = planted[track] in TRAP_YIELDS
        for index in range(CLAIMS_PER_DIRECTION):
            true = rng.random() < planted[track]
            if trap and rng.random() < TRAP_WEAK_SHARE:
                kind = "weak"
            else:
                kind = rng.choices(["symbolic_identity", "matmul_tensor", "strong"],
                                   weights=[5, 4, 1])[0]
            claim = make_claim(true, kind, rng)
            claim.update(id=f"claim:{track}:{index:03d}",
                         work=rng.choice(works_by_track[track]), truth=true)
            claims.append(claim)
    rng.shuffle(claims)
    return {
        "note": "Synthetic. Candidates are real verifier payloads; their attachment "
                "to works and each direction's yield are planted. See "
                "scripts/make_claim_bank.py.",
        "seed": SEED,
        "planted_yield": planted,
        "traps": sorted(t for t in tracks if planted[t] in TRAP_YIELDS),
        "claims": claims,
    }


if __name__ == "__main__":
    bank = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bank, separators=(",", ":")) + "\n")
    print(f"wrote {len(bank['claims'])} claims for {len(bank['planted_yield'])} "
          f"directions to {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
