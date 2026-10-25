"""Generate a SYNTHETIC stress corpus for benchmarking discovery at scale.

The shipped demo corpus is too easy to benchmark against: every abstract is three
of six fixed phrases, so each direction has only 20 distinct abstracts, and every
title names its direction. Any method looks perfect on it, and scaling it up just
adds duplicates. This generator keeps the same 16 planted directions but makes
each record messier, closer to how real abstracts behave:

- 25% of the words in each phrase are dropped at random, so near-duplicates vanish
- 30% of works borrow a phrase from another direction (cross-topic bridges)
- every work gets 6 words from a shared pool of 400 generic pseudo-words, so
  there is a common background vocabulary pulling directions together
- only half of titles name the direction

Like the demo corpus it is machine-generated and says nothing about any real
field. Labels are kept in ``extra.track`` so discovery can be scored.

Run:
    python bench/make_stress_corpus.py --works 12000 --out data/stress_12000.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts" / "make_demo_corpus.py")
_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_demo)
TRACKS = _demo.TRACKS

SEED = 20261025

# An extra direction that can be planted in small numbers, to check that a
# direction too rare to appear in a sample is still found.
RARE = {
    "track": "magnetocaloric-cooling",
    "name": "Magnetocaloric refrigeration",
    "disciplines": ["physics"],
    "yearly": {2026: 1},
    "phrases": [
        "Gadolinium alloys show a giant magnetocaloric entropy change near room temperature",
        "Active magnetic regenerator cycles reach a record temperature span",
        "Permanent magnet assemblies supply the field without superconducting coils",
        "Hysteresis losses in first-order transition materials limit cycle frequency",
        "Lanthanum iron silicon hydrides are shaped into microchannel regenerator beds",
        "Compressor-free cooling avoids refrigerant gases with high warming potential",
    ],
}
SYLLABLES = "ka lo mi ne ru sa ti vo ze ba de fi gu ho ja ki".split()


def generic_pool(rng: random.Random, size: int = 400) -> list[str]:
    pool: set[str] = set()
    while len(pool) < size:
        pool.add("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(pool)


def degrade(phrase: str, rng: random.Random, drop: float) -> str:
    words = [w for w in phrase.split() if rng.random() >= drop]
    return " ".join(words) if words else phrase


def build(n_works: int, seed: int = SEED, rare: int = 0) -> list[dict]:
    rng = random.Random(seed)
    pool = generic_pool(rng)
    weights = [sum(t["yearly"].values()) for t in TRACKS]
    years = [y for y in range(2021, 2027)]
    records = []
    for k in range(n_works + rare):
        spec = RARE if k >= n_works else rng.choices(TRACKS, weights=weights)[0]
        other = rng.choice([t for t in TRACKS if t is not spec])
        parts = [degrade(p, rng, 0.25) for p in rng.sample(spec["phrases"], k=3)]
        if rng.random() < 0.30:
            parts.append(degrade(rng.choice(other["phrases"]), rng, 0.25))
        parts.append(" ".join(rng.choice(pool) for _ in range(6)))
        rng.shuffle(parts)
        title = (f"{spec['name']}: study {k}" if rng.random() < 0.5
                 else "On " + " ".join(rng.choice(pool) for _ in range(3)))
        year = rng.choice(years)
        records.append({
            "id": f"synthetic-stress:{k:06d}",
            "title": title,
            "abstract": ". ".join(parts),
            "authors": [{"name": f"Author {k}-0", "affiliation": "Institution A",
                         "is_industry": False}],
            "published": dt.date(year, rng.randint(1, 9), rng.randint(1, 28)).isoformat(),
            "venue": "Synthetic Journal of Benchmark Data",
            "maturity": "preprint",
            "disciplines": rng.sample(spec["disciplines"], k=1),
            "citations": rng.randint(0, 50),
            "url": None,
            "source": "synthetic",
            "extra": {"track": spec["track"], "SYNTHETIC": True},
        })
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--works", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rare", type=int, default=0,
                        help="also plant this many works of a 17th, rare direction")
    args = parser.parse_args()
    records = build(args.works, args.seed, args.rare)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records))
    print(f"wrote {len(records)} SYNTHETIC stress records to {args.out}")
