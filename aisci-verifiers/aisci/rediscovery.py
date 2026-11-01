"""I5: executable rediscovery tasks, a baseline law-finder, and a novelty metric.

kepler3 uses real planetary orbit values. planck uses spectra generated from
Planck's law (the law is the ground truth being rediscovered). The baseline is
a power-law / exponential search; it should solve Kepler and fail Planck,
which shows the benchmark separates easy from hard rediscoveries.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# semimajor axis (AU), sidereal period (years)
PLANETS = {"Mercury": (0.387, 0.241), "Venus": (0.723, 0.615), "Earth": (1.000, 1.000),
           "Mars": (1.524, 1.881), "Jupiter": (5.203, 11.86), "Saturn": (9.537, 29.46)}


def kepler_data():
    a, T = np.array(list(PLANETS.values())).T
    return a, T


def planck_data(T=5000.0, n=60):
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    nu = np.linspace(1e13, 2.5e15, n)
    B = 2 * h * nu ** 3 / c ** 2 / (np.exp(h * nu / (k * T)) - 1)
    return nu, B


def fit_laws(x, y):
    """Baseline law-finder: fits y = c x^p and y = c exp(p x); returns the best by R^2 in log space."""
    ly = np.log(y)
    out = []
    p, c = np.polyfit(np.log(x), ly, 1)
    pred = c + p * np.log(x)
    out.append(("power", p, c, 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)))
    p, c = np.polyfit(x, ly, 1)
    pred = c + p * x
    out.append(("exp", p, c, 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)))
    return max(out, key=lambda r: r[3])


def score_kepler(form, p, r2):
    return form == "power" and abs(p - 1.5) < 0.01 and r2 > 0.999


def score_planck(form, p, r2):
    # Planck's law is neither a pure power law nor a pure exponential: a correct
    # rediscovery must fit across the peak (R^2 > 0.999 in log space).
    return r2 > 0.999


# ---------- task 5.2: novelty against a literature cutoff ----------

def _tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]


def novelty(claim: str, corpus: list[tuple[int, str]], cutoff_year: int) -> float:
    """1 - max TF-IDF cosine similarity to any document published before the cutoff."""
    docs = [d for y, d in corpus if y < cutoff_year]
    if not docs:
        return 1.0
    toks = [_tok(d) for d in docs] + [_tok(claim)]
    df = Counter(w for t in toks for w in set(t))
    N = len(toks)

    def vec(t):
        tf = Counter(t)
        return {w: (1 + math.log(n)) * (math.log((1 + N) / (1 + df[w])) + 1) for w, n in tf.items()}

    def cos(a, b):
        num = sum(a[w] * b.get(w, 0) for w in a)
        den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return num / den if den else 0.0

    q = vec(toks[-1])
    return 1 - max(cos(q, vec(t)) for t in toks[:-1])


def run():
    a, T = kepler_data()
    k = fit_laws(a, T)
    nu, B = planck_data()
    pl = fit_laws(nu, B)
    return {"kepler3": {"form": k[0], "exponent": round(float(k[1]), 4), "r2": round(float(k[3]), 6),
                        "solved": bool(score_kepler(k[0], k[1], k[3]))},
            "planck": {"form": pl[0], "param": float(pl[1]), "r2": round(float(pl[3]), 4),
                       "solved": bool(score_planck(pl[0], pl[1], pl[3]))}}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
