"""Cross-study cell-type annotation on the Human Lung Cell Atlas, CPU only.

Experiment for HFM's "Cross-Study Cell-Type Annotation with UCE on the Human Lung
Cell Atlas" track. Data: the atlas core (584,944 cells, 11 studies, 107 donors;
Sikkema et al. 2023) from CELLxGENE, CC-BY 4.0. Only the cell labels and the
30-dimension scANVI embedding stored in the file are read, over HTTP, so the 5.9 GB
file is never downloaded.

Protocol: leave one study out. A classifier is trained on cells from the other
studies and must label every cell of the held-out study at the atlas's finest
annotation level. Score: macro-F1 over the cell types present in the held-out
study, averaged over studies (HFM's "study_macro_f1" construction; HFM uses the
full atlas with 35 studies and UCE embeddings, so numbers are NOT comparable).

Important caveat: the scANVI embedding was itself trained with labels from every
study, including the held-out one. Absolute scores are therefore optimistic. The
comparison between methods on the same embedding is still fair.

Methods
  flat                 logistic regression on the finest label
  class-balanced       the same with classes weighted equally, since rare types
                       dominate macro-F1
  hierarchy-aware      class-balanced classifiers at annotation levels 1, 2 and 3
                       and at the finest level; a leaf's score adds the log
                       probability of each of its ancestors, the "decode over the
                       hierarchy" idea of research/human-frontier-mission.md 4.2
  nearest neighbours   15 nearest training cells, distance-weighted vote

Usage: python cross_study_annotation.py [h5ad-url]
"""
from __future__ import annotations

import sys

import fsspec
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

URL = "https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"
LEVELS = ["ann_level_1", "ann_level_2", "ann_level_3"]
PER_STUDY_TRAIN, PER_STUDY_TEST = 25000, 40000


def cat(h, col):
    g = h["obs"][col]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]])
    return g["codes"][:].astype(np.int64), cats


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    with fsspec.open(url, "rb", block_size=8 * 2**20) as fh, h5py.File(fh, "r") as h:
        X = h["obsm"]["X_scanvi_emb"][:].astype(np.float32)
        study, study_names = cat(h, "study")
        fine, fine_names = cat(h, "ann_finest_level")
        levels = [cat(h, c)[0] for c in LEVELS]
    print(f"{len(X)} cells, {len(study_names)} studies, {len(fine_names)} finest-level cell types", flush=True)
    # each finest type's ancestor at each level: the most common ancestor among its cells
    anc = np.zeros((len(LEVELS), len(fine_names)), dtype=np.int64)
    for li, lv in enumerate(levels):
        for c in range(len(fine_names)):
            m = fine == c
            anc[li, c] = np.bincount(lv[m]).argmax() if m.any() else 0

    rng = np.random.default_rng(0)
    scores = {k: [] for k in ("flat", "class-balanced", "hierarchy-aware", "nearest neighbours")}
    rare = {k: [] for k in scores}
    for s in range(len(study_names)):
        tr = np.concatenate([rng.permutation(np.flatnonzero(study == t))[:PER_STUDY_TRAIN] for t in range(len(study_names)) if t != s])
        te = rng.permutation(np.flatnonzero(study == s))[:PER_STUDY_TEST]
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        y, yt = fine[tr], fine[te]
        present = np.unique(yt)
        counts = np.bincount(yt, minlength=len(fine_names))
        rare_types = [c for c in present if counts[c] < 0.005 * len(yt)]

        def record(name, pred):
            scores[name].append(f1_score(yt, pred, labels=present, average="macro", zero_division=0))
            rare[name].append(f1_score(yt, pred, labels=rare_types, average="macro", zero_division=0) if rare_types else np.nan)

        flat = LogisticRegression(max_iter=300, C=1.0).fit(A, y)
        record("flat", flat.predict(B))
        bal = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(A, y)
        logp = np.full((len(B), len(fine_names)), -30.0)
        logp[:, bal.classes_] = bal.predict_log_proba(B)
        record("class-balanced", logp.argmax(axis=1))
        total = logp.copy()
        for li, lv in enumerate(levels):
            m = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(A, lv[tr])
            lp = np.full((len(B), lv.max() + 1), -30.0)
            lp[:, m.classes_] = m.predict_log_proba(B)
            total += lp[:, anc[li]]
        record("hierarchy-aware", total.argmax(axis=1))
        knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=4).fit(A, y)
        record("nearest neighbours", knn.predict(B))
        print(f"  held out {study_names[s][:38]:<38} {len(te):>6} cells, {len(present):>2} types: "
              + ", ".join(f"{k} {v[-1]:.3f}" for k, v in scores.items()), flush=True)

    base = np.array(scores["flat"])
    print(f"\nstudy macro-F1, leave-one-study-out over {len(base)} studies")
    print(f"  {'method':<20} {'macro-F1':>9} {'vs flat':>9} {'studies better':>15} {'rare-type F1':>13}")
    for k, v in scores.items():
        v = np.array(v)
        print(f"  {k:<20} {v.mean():>9.4f} {v.mean() - base.mean():>+9.4f} {int((v > base + 1e-9).sum()):>11}/{len(v)} {np.nanmean(rare[k]):>13.4f}")


if __name__ == "__main__":
    main()
