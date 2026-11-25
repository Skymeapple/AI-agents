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

Four further methods use the held-out study's own cells without their labels,
which the task allows (HFM's agents use pseudo-labels):
  smoothed             average the flat classifier's probabilities over each cell's
                       15 nearest cells in the same study
  centred study        shift the study's embedding so its mean matches the training
                       mean
  self-training        add confidently labelled held-out cells and refit once
  self-training +      both of the above
  smoothing

Usage: python cross_study_annotation.py [h5ad-url | full]
  "full" runs on the full atlas (all studies, Cell Ontology labels), which is HFM's setting.
  The hierarchy-aware method needs the core's annotation levels and is skipped there.
"""
from __future__ import annotations

import sys

import fsspec
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

URL = "https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad"
LEVELS = ["ann_level_1", "ann_level_2", "ann_level_3"]
PER_STUDY_TRAIN, PER_STUDY_TEST = 25000, 40000


def cat(h, col):
    g = h["obs"][col]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]])
    return g["codes"][:].astype(np.int64), cats


def full_atlas_url():
    """Address of the full atlas file (2.3 million cells, all studies), looked up from CELLxGENE."""
    import json
    import urllib.request
    api = "https://api.cellxgene.cziscience.com/curation/v1/collections/6f6d381a-7701-4781-935c-db10d30de293"
    c = json.load(urllib.request.urlopen(api, timeout=120))
    for ds in c["datasets"]:
        if "(full)" in (ds.get("title") or ""):
            return next(a["url"] for a in ds["assets"] if a["filetype"] == "H5AD")
    raise RuntimeError("full atlas not found")


def main():
    global PER_STUDY_TRAIN, PER_STUDY_TEST
    full = len(sys.argv) > 1 and sys.argv[1] == "full"
    url = full_atlas_url() if full else (sys.argv[1] if len(sys.argv) > 1 else URL)
    label = "cell_type" if full else "ann_finest_level"     # the full atlas is labelled with Cell Ontology terms
    if full:
        PER_STUDY_TRAIN, PER_STUDY_TEST = 6000, 30000
    with fsspec.open(url, "rb", block_size=8 * 2**20) as fh, h5py.File(fh, "r") as h:
        print("obsm:", list(h["obsm"].keys()), flush=True)
        X = h["obsm"]["X_scanvi_emb"][:].astype(np.float32)
        study, study_names = cat(h, "study")
        fine, fine_names = cat(h, label)
        levels = [cat(h, c)[0] for c in LEVELS] if not full else []
    keep = fine >= 0
    if full:                                 # drop labels too rare to score and cells without a label
        counts = np.bincount(fine[keep], minlength=len(fine_names))
        keep &= counts[np.maximum(fine, 0)] >= 50
    X, study, fine = X[keep], study[keep], fine[keep]
    levels = [lv[keep] for lv in levels]
    print(f"{len(X)} cells, {len(study_names)} studies, {len(fine_names)} finest-level cell types", flush=True)
    # each finest type's ancestor at each level: the most common ancestor among its cells
    anc = np.zeros((len(LEVELS), len(fine_names)), dtype=np.int64)
    for li, lv in enumerate(levels):
        for c in range(len(fine_names)):
            m = fine == c
            anc[li, c] = np.bincount(lv[m]).argmax() if m.any() else 0

    rng = np.random.default_rng(0)
    scores = {k: [] for k in ("flat", "class-balanced", "hierarchy-aware", "nearest neighbours", "smoothed over neighbours", "centred study", "self-training", "self-training + smoothing")}
    rare = {k: [] for k in scores}
    for s in range(len(study_names)):
        if (study == s).sum() < 2000:
            continue
        tr = np.concatenate([rng.permutation(np.flatnonzero(study == t))[:PER_STUDY_TRAIN] for t in range(len(study_names)) if t != s and (study == t).any()])
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
        record("hierarchy-aware", total.argmax(axis=1) if levels else logp.argmax(axis=1))
        knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=4).fit(A, y)
        record("nearest neighbours", knn.predict(B))
        # --- methods that use the held-out study's own unlabelled cells ---
        pf = np.zeros((len(B), len(fine_names)))
        pf[:, flat.classes_] = flat.predict_proba(B)
        # 1. smooth each cell's prediction over its 15 nearest cells in the same study
        nn = NearestNeighbors(n_neighbors=15, n_jobs=4).fit(B)
        idx = nn.kneighbors(B, return_distance=False)
        record("smoothed over neighbours", pf[idx].mean(axis=1).argmax(axis=1))
        # 2. shift the study's embedding so its mean matches the training mean
        Bc = B - B.mean(axis=0) + A.mean(axis=0)
        record("centred study", flat.predict(Bc))
        # 3. self-training: add confidently labelled held-out cells and refit once
        conf = pf.max(axis=1) > 0.9
        st = LogisticRegression(max_iter=300, C=1.0).fit(np.vstack([A, B[conf]]), np.concatenate([y, pf[conf].argmax(axis=1)]))
        ps = np.zeros_like(pf)
        ps[:, st.classes_] = st.predict_proba(B)
        record("self-training", ps.argmax(axis=1))
        record("self-training + smoothing", ps[idx].mean(axis=1).argmax(axis=1))
        print(f"  held out {study_names[s][:30]:<30} {len(te):>6} cells: " + ", ".join(f"{v[-1]:.3f}" for v in scores.values()), flush=True)

    base = np.array(scores["flat"])
    print(f"\nstudy macro-F1, leave-one-study-out over {len(base)} studies")
    print(f"  {'method':<26} {'macro-F1':>9} {'vs flat':>9} {'studies better':>15} {'rare-type F1':>13}")
    for k, v in scores.items():
        v = np.array(v)
        print(f"  {k:<26} {v.mean():>9.4f} {v.mean() - base.mean():>+9.4f} {int((v > base + 1e-9).sum()):>11}/{len(v)} {np.nanmean(rare[k]):>13.4f}")


if __name__ == "__main__":
    main()
