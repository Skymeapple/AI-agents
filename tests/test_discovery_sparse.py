"""The sparse clustering path must reproduce the dense reference exactly.

The dense functions are kept as the reference implementation: they are simple
enough to check by eye, and quadratic memory does not matter on test corpora.
"""

import random

import numpy as np

from sciscout.discovery import (
    _agglomerate,
    _agglomerate_sparse,
    _build_tfidf,
    _build_tfidf_sparse,
    _candidate_pairs,
)
from sciscout.sources.local import LocalCorpus

from tests.test_discovery import make_works


def noisy_corpus(n_topics=6, per_topic=25, seed=3):
    """Topics sharing a methods vocabulary, with some works bridging two topics."""
    rng = random.Random(seed)
    methods = [f"method{i}" for i in range(40)]
    topics = [[f"topic{t}term{i}" for i in range(12)] for t in range(n_topics)]
    specs = []
    for t, vocab in enumerate(topics):
        for i in range(per_topic):
            words = rng.choices(vocab, k=12) + rng.choices(methods, k=18)
            if rng.random() < 0.1:
                words += rng.choices(topics[(t + 1) % n_topics], k=8)
            specs.append((f"{t}-{i}", " ".join(words), ["field"], f"topic{t}"))
    rng.shuffle(specs)
    return make_works(specs)


def dense_reference(works, threshold):
    matrix, _ = _build_tfidf(works, 2, 0.5)
    return _agglomerate(matrix @ matrix.T, threshold)


def as_sets(clusters):
    """Clusters in order, members as sets.

    Member order inside a cluster is merge order, and where two averages are
    exactly equal -- the demo corpus repeats abstracts verbatim -- which merges
    first depends on the last bit of rounding, which differs between a dense
    matrix product and summed vectors. Which works end up together, and the
    order of clusters, do not.
    """
    return [set(members) for members in clusters]


def test_sparse_tfidf_matches_dense_bit_for_bit():
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    dense, vocabulary = _build_tfidf(works, 2, 0.5)
    sparse = _build_tfidf_sparse(works, 2, 0.5)
    assert sparse.vocabulary == vocabulary
    rebuilt = np.zeros_like(dense)
    rows = np.repeat(np.arange(len(works)), np.diff(sparse.indptr))
    rebuilt[rows, sparse.indices] = sparse.data
    assert np.array_equal(rebuilt, dense)


def test_candidate_pairs_include_every_pair_above_the_floor():
    works = noisy_corpus()
    dense, _ = _build_tfidf(works, 2, 0.5)
    similarity = dense @ dense.T
    a, b, sims = _candidate_pairs(_build_tfidf_sparse(works, 2, 0.5), 0.16)
    found = set(zip(a.tolist(), b.tolist()))
    expected = {
        (i, j)
        for i, j in zip(*np.nonzero(similarity >= 0.16))
        if i < j
    }
    assert expected <= found
    assert np.allclose(sims, similarity[a, b])


def test_sparse_clustering_matches_dense_on_the_demo_corpus():
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    clusters, _ = _agglomerate_sparse(_build_tfidf_sparse(works, 2, 0.5), 0.16)
    assert as_sets(clusters) == as_sets(dense_reference(works, 0.16))


def test_sparse_clustering_matches_dense_with_shared_vocabulary_and_bridges():
    works = noisy_corpus()
    for threshold in (0.10, 0.16, 0.25):
        clusters, _ = _agglomerate_sparse(_build_tfidf_sparse(works, 2, 0.5), threshold)
        assert as_sets(clusters) == as_sets(dense_reference(works, threshold))
