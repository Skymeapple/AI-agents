"""The two clustering methods, held to the same answer.

The sparse method exists to run where the dense one cannot, which is exactly
where nobody can check it against the dense one. So it is checked here, on
corpora small enough for both, and required to agree.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from sciscout.discovery import (
    AUTO_SPARSE_ABOVE,
    _DenseIndex,
    _index_class,
    _SparseIndex,
    discover,
)
from sciscout.linkage import agglomerate_dense, agglomerate_sparse
from sciscout.sources.local import LocalCorpus


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def both(matrix: np.ndarray, threshold: float):
    dense = sorted(sorted(c) for c in agglomerate_dense(matrix @ matrix.T, threshold))
    return dense, agglomerate_sparse(sp.csr_matrix(matrix), threshold)


def random_corpus(rng: np.random.Generator) -> np.ndarray:
    """Noisy points around a few sparse centres, sometimes with an empty work."""
    n, terms, topics = int(rng.integers(2, 120)), int(rng.integers(5, 200)), int(rng.integers(1, 8))
    centres = rng.random((topics, terms)) * (rng.random((topics, terms)) < rng.uniform(0.05, 0.5))
    noise = rng.random((n, terms)) * (rng.random((n, terms)) < rng.uniform(0.02, 0.4))
    matrix = centres[rng.integers(0, topics, n)] + rng.uniform(0.0, 1.5) * noise
    if rng.random() < 0.2:
        matrix[rng.integers(0, n)] = 0.0
    return unit_rows(matrix)


def test_sparse_and_dense_agree_on_random_corpora():
    rng = np.random.default_rng(7)
    cluster_counts = set()
    for _ in range(40):
        matrix = random_corpus(rng)
        for threshold in (0.0, 0.16, 0.5):
            dense, sparse = both(matrix, threshold)
            assert sparse == dense
            cluster_counts.add(len(dense))
    # The comparison is only worth something if it saw real variety.
    assert len(cluster_counts) > 10


def test_identical_works_merge_and_unrelated_ones_do_not():
    matrix = unit_rows(np.array([[1.0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]))
    dense, sparse = both(matrix, 0.5)
    assert sparse == dense == [[0, 1], [2, 3], [4]]


def test_a_threshold_nothing_reaches_leaves_every_work_alone():
    matrix = unit_rows(np.random.default_rng(1).random((12, 6)))
    dense, sparse = both(matrix, 1.5)
    assert sparse == dense == [[i] for i in range(12)]


def test_a_bridging_work_does_not_fuse_two_topics():
    """The reason for average linkage: one in-between paper must not chain topics."""
    a, b = [1.0, 0.0], [0.0, 1.0]
    matrix = unit_rows(np.array([a] * 5 + [b] * 5 + [[1.0, 1.0]]))
    dense, sparse = both(matrix, 0.6)
    assert sparse == dense
    assert len(sparse) == 2


def test_discovery_gives_the_same_directions_by_either_method():
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    dense = discover(works, method="dense")
    sparse = discover(works, method="sparse")

    def by_members(result):
        return {frozenset(w.id for w in t.works): t for t in result.tracks}

    dense_tracks, sparse_tracks = by_members(dense), by_members(sparse)
    assert sparse_tracks.keys() == dense_tracks.keys()
    for members, track in dense_tracks.items():
        other = sparse_tracks[members]
        assert other.terms == track.terms
        assert other.label == track.label
        assert other.cohesion == pytest.approx(track.cohesion, abs=1e-12)
    assert {w.id for w in sparse.unclustered} == {w.id for w in dense.unclustered}
    assert sparse.vocabulary_size == dense.vocabulary_size


def test_auto_switches_to_sparse_only_for_a_large_corpus():
    assert _index_class("auto", AUTO_SPARSE_ABOVE) is _DenseIndex
    assert _index_class("auto", AUTO_SPARSE_ABOVE + 1) is _SparseIndex
    assert _index_class("sparse", 10) is _SparseIndex
    assert _index_class("dense", 10) is _DenseIndex


def test_an_unknown_method_is_rejected():
    works = LocalCorpus("data/demo_corpus.json").harvest().works[:20]
    with pytest.raises(ValueError, match="unknown method"):
        discover(works, method="approximate")
