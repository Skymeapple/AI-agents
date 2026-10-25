"""Comparison harness for PROTOCOL.md: baselines B0-B2 vs AI arms A1-A3.

    python experiments/discovery_real/harness.py --arm B1 --dataset BIO --data-dir DIR --out out/

Each invocation tunes one arm on the tuning sample, runs it on the five replicates,
and writes a JSON record. Per-work error flags are saved in confidence order, so
cost under other time constants (sensitivity analysis) can be recomputed without
re-running anything.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sciscout.discovery import _agglomerate, _build_tfidf, discover  # noqa: E402
from sciscout.models import Work  # noqa: E402

# Frozen constants (PROTOCOL.md sections 3 and 5).
T_C, T_S, T_R = 120.0, 4.0, 10.0
ANALYST_PER_S = 60.0 / 3600
COMPUTE_PER_S = 4 * 0.10 / 3600
REPLICATE_N = 6000
SEEDS = [0, 1, 2, 3, 4]
GRIDS = {
    "B0": [0.16],
    "B1": [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25],
    "B2": [5, 7, 10, 15, 20, 25, 30, 40],
    "A1": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    "A2": [5, 7, 10, 15, 20, 25, 30, 40],
    "A3": [5, 7, 10, 15, 20, 25, 30, 40],
}


# ---------------------------------------------------------------- data

def load_dataset(name: str, data_dir: Path) -> tuple[list[str], np.ndarray]:
    if name == "BIO":
        texts = (data_dir / "stc2/dataset/Biomedical.txt").read_text().splitlines()
        labels = (data_dir / "stc2/dataset/Biomedical_gnd.txt").read_text().split()
    elif name == "MAG":
        rows = [json.loads(l) for l in open(data_dir / "mag_test.txt")]
        texts, labels = [r["text"] for r in rows], [r["label"] for r in rows]
    elif name == "DEMO":
        rows = json.load(open(Path(__file__).resolve().parents[2] / "data/demo_corpus.json"))
        rows = rows["works"] if isinstance(rows, dict) else rows
        texts = [(r.get("title", "") + " " + r.get("abstract", "")).strip() for r in rows]
        labels = [str(r.get("track") or r.get("extra", {}).get("track")) for r in rows]
    else:
        raise ValueError(name)
    _, y = np.unique(np.array(labels), return_inverse=True)
    return texts, y


def splits(y: np.ndarray, dataset: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Disjoint stratified tuning sample, then stratified replicates from the rest."""
    classes = np.unique(y)
    per_tune = {"BIO": 100, "MAG": 300}.get(dataset)
    rng = np.random.default_rng(100)
    if per_tune is None:  # DEMO: whole corpus, no tuning
        return np.arange(len(y)), [np.arange(len(y))]
    tune = np.concatenate([rng.choice(np.where(y == c)[0], per_tune, replace=False) for c in classes])
    rest = np.setdiff1d(np.arange(len(y)), tune)
    reps = []
    for s in SEEDS:
        r = np.random.default_rng(s)
        per = REPLICATE_N // len(classes)
        idx = np.concatenate([r.choice(rest[y[rest] == c], per, replace=False) for c in classes])
        extra = REPLICATE_N - len(idx)
        if extra:
            idx = np.concatenate([idx, r.choice(np.setdiff1d(rest, idx), extra, replace=False)])
        reps.append(np.sort(idx))
    return tune, reps


# ---------------------------------------------------------------- representations

_WL = None


def embed(texts: list[str]) -> np.ndarray:
    global _WL
    if _WL is None:
        from wordllama import WordLlama
        cache = Path(__file__).resolve().parent / ".wlcache"
        _WL = WordLlama.load(cache_dir=str(cache), disable_download=True)
    return np.asarray(_WL.embed(texts, norm=True), dtype=np.float64)


def as_works(texts: list[str]) -> list[Work]:
    return [Work(id=str(i), title=t) for i, t in enumerate(texts)]


def lsa(works: list[Work], seed: int) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    m, vocab = _build_tfidf(works, 2, 0.5)
    z = TruncatedSVD(100, random_state=seed).fit_transform(m)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def centroid_confidence(x: np.ndarray, assign: np.ndarray) -> np.ndarray:
    conf = np.full(len(assign), -np.inf)
    for c in np.unique(assign[assign >= 0]):
        idx = assign == c
        mu = x[idx].mean(0)
        mu /= max(np.linalg.norm(mu), 1e-12)
        conf[idx] = x[idx] @ mu
    return conf


def agglomerate(x: np.ndarray, threshold: float, min_size: int = 3) -> np.ndarray:
    clusters = _agglomerate(x @ x.T, threshold)
    assign = np.full(len(x), -1)
    for cid, members in enumerate(c for c in clusters if len(c) >= min_size):
        assign[members] = cid
    return assign


def kmeans(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    return KMeans(k, n_init=10, random_state=seed).fit_predict(x)


def run_arm(arm: str, texts: list[str], param, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    works = as_works(texts)
    if arm in ("B0", "B1"):
        result = discover(works, threshold=param)  # shipped code path, untouched
        assign = np.full(len(works), -1)
        for cid, track in enumerate(result.tracks):
            assign[[int(w.id) for w in track.works]] = cid
        m, _ = _build_tfidf(works, 2, 0.5)
        conf = centroid_confidence(m, assign)
    elif arm == "B2":
        x = lsa(works, seed)
        assign = kmeans(x, param, seed)
        conf = centroid_confidence(x, assign)
    elif arm == "A1":
        x = embed(texts)
        assign = agglomerate(x, param)
        conf = centroid_confidence(x, assign)
    elif arm == "A2":
        x = embed(texts)
        assign = kmeans(x, param, seed)
        conf = centroid_confidence(x, assign)
    elif arm == "A3":
        x = np.hstack([lsa(works, seed), embed(texts)]) / np.sqrt(2)
        assign = kmeans(x, param, seed)
        conf = centroid_confidence(x, assign)
    else:
        raise ValueError(arm)
    return assign, conf, time.perf_counter() - t0


# ---------------------------------------------------------------- scoring

def error_flags(y: np.ndarray, assign: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Per-work wrong flags in screening order (least confident first)."""
    wrong = assign < 0
    for c in np.unique(assign[assign >= 0]):
        idx = assign == c
        majority = np.bincount(y[idx]).argmax()
        wrong[idx] = y[idx] != majority
    return wrong[np.argsort(conf, kind="stable")]


def cost(flags: np.ndarray, n_clusters: int, wall: float, q: float = 0.90,
         t_c: float = T_C, t_s: float = T_S, t_r: float = T_R) -> dict:
    n = len(flags)
    allowed = int(np.floor((1 - q) * n))
    total_wrong = int(flags.sum())
    cum = np.concatenate([[0], np.cumsum(flags)])
    need = total_wrong - allowed
    screened = 0 if need <= 0 else int(np.searchsorted(cum, need))
    fixed = int(cum[screened])
    human_s = n_clusters * t_c + screened * t_s + fixed * t_r
    parts = {
        "machine": wall * COMPUTE_PER_S,
        "review": n_clusters * t_c * ANALYST_PER_S,
        "screen": screened * t_s * ANALYST_PER_S,
        "fix": fixed * t_r * ANALYST_PER_S,
    }
    return {"total_usd": sum(parts.values()), "parts_usd": parts, "human_hours": human_s / 3600,
            "screened": screened, "fixed": fixed, "raw_acc": 1 - total_wrong / n}


def metrics(y: np.ndarray, assign: np.ndarray) -> dict:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    a = np.where(assign < 0, assign.max() + 1 + np.arange(len(assign)), assign)  # unassigned = own singleton
    clusters = np.unique(assign[assign >= 0])
    table = np.array([[np.sum((assign == c) & (y == k)) for k in np.unique(y)] for c in clusters])
    r, c = linear_sum_assignment(-table) if len(clusters) else ([], [])
    purities, recovered = [], set()
    for row in table:
        purities.append(row.max() / row.sum())
        recovered.add(int(row.argmax()))
    return {
        "nmi": float(normalized_mutual_info_score(y, a)),
        "ari": float(adjusted_rand_score(y, a)),
        "hungarian_acc": float(table[r, c].sum() / len(y)) if len(clusters) else 0.0,
        "coverage": float((assign >= 0).mean()),
        "clusters": int(len(clusters)),
        "recovery_rate": len(recovered) / len(np.unique(y)),
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
    }


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--param", default=None, help="skip tuning and use this value (DEMO)")
    args = ap.parse_args()

    texts, y = load_dataset(args.dataset, args.data_dir)
    tune, reps = splits(y, args.dataset)
    record = {"arm": args.arm, "dataset": args.dataset, "tuning": [], "replicates": []}

    if args.param is not None:
        best = type(GRIDS[args.arm][0])(args.param)
    else:
        for p in GRIDS[args.arm]:
            assign, conf, wall = run_arm(args.arm, [texts[i] for i in tune], p, seed=100)
            flags = error_flags(y[tune], assign, conf)
            c = cost(flags, int(len(np.unique(assign[assign >= 0]))), wall)
            record["tuning"].append({"param": p, "total_usd": c["total_usd"], **metrics(y[tune], assign)})
        best = min(record["tuning"], key=lambda r: r["total_usd"])["param"]
    record["param"] = best

    for seed, idx in zip(SEEDS, reps):
        assign, conf, wall = run_arm(args.arm, [texts[i] for i in idx], best, seed=seed)
        flags = error_flags(y[idx], assign, conf)
        n_clusters = int(len(np.unique(assign[assign >= 0])))
        record["replicates"].append({
            "seed": seed, "n": len(idx), "wall_s": wall, "n_clusters": n_clusters,
            "q90": cost(flags, n_clusters, wall, 0.90), "q95": cost(flags, n_clusters, wall, 0.95),
            "metrics": metrics(y[idx], assign),
            "flags": np.packbits(flags).tobytes().hex(),
        })
    record["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.dataset}_{args.arm}.json").write_text(json.dumps(record, indent=1))
    r = record["replicates"]
    print(f"{args.dataset} {args.arm} param={best} "
          f"q90=${np.mean([x['q90']['total_usd'] for x in r]):.2f} "
          f"nmi={np.mean([x['metrics']['nmi'] for x in r]):.3f} "
          f"clusters={np.mean([x['n_clusters'] for x in r]):.1f} "
          f"wall={np.mean([x['wall_s'] for x in r]):.1f}s rss={record['peak_rss_mb']:.0f}MB")


if __name__ == "__main__":
    main()
