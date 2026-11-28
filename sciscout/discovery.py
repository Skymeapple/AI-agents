"""Finding research directions without being told what they are.

Until now a research direction had to be hand-labelled, which is fine for a
watchlist of a dozen topics and useless for watching science broadly. Nobody is
going to hand-label all of it, and the labels you would write are the ones you
already know to look for -- which excludes exactly the emerging directions the
platform exists to catch.

This module clusters works into candidate directions from their text alone.
TF-IDF over titles and abstracts, cosine similarity, average-linkage
agglomerative clustering with a similarity floor. No embeddings and no network:
it runs anywhere the corpus does, and it is deterministic, so the same corpus
gives the same directions every time.

What this deliberately does not do is pretend to be better than it is. Lexical
clustering groups papers that use the same words. Two groups pursuing the same
idea in different vocabularies will land in separate clusters, and a shared
methods vocabulary can pull unrelated work together. So every cluster carries a
cohesion score, singletons are quarantined rather than promoted to directions,
and :func:`evaluate_against_labels` exists so you can measure the damage on a
corpus where you do know the answer. Embeddings would do better; when the
environment can reach a model, that is the upgrade path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .linkage import agglomerate_dense, agglomerate_sparse
from .models import Track, Work

# Words that carry no topical signal in scientific abstracts. Kept deliberately
# short: an aggressive stoplist silently removes the domain vocabulary that makes
# one field distinguishable from another.
STOPWORDS: frozenset[str] = frozenset(
    """
    the and for are but not you all can had her was one our out has him his how
    its may new now old see two way who did get let put say she too use with
    this that from they have been were will would could should there their which
    when what where while about into than then them these those such some other
    more most many much very also only just both each same over under after
    before between during above below again further once here very
    study studies result results show shows shown showed report reports reported
    present presents presented demonstrate demonstrates demonstrated find finds
    found observe observed observation using used use uses based approach method
    methods paper work works research analysis analyses data proposed propose
    proposes novel new recent recently here we our us it is as at by of in on to
    a an be do does done may might can could well may
    """.split()
)

TOKEN_RE = re.compile(r"[a-z][a-z-]{2,}")


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens of three or more characters, stopwords removed."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class DiscoveredTrack:
    """A candidate research direction found in the corpus."""

    id: str
    label: str
    works: list[Work]
    terms: list[str]
    cohesion: float

    @property
    def size(self) -> int:
        return len(self.works)

    def to_track(self, sector: str = "generic") -> Track:
        return Track(
            id=self.id,
            name=self.label,
            description=f"Discovered cluster of {self.size} works; "
            f"distinctive terms: {', '.join(self.terms[:8])}",
            works=list(self.works),
            sector=sector,
        )


@dataclass
class DiscoveryResult:
    """Everything discovery found, including what it could not place."""

    tracks: list[DiscoveredTrack]
    unclustered: list[Work] = field(default_factory=list)
    vocabulary_size: int = 0
    threshold: float = 0.0

    @property
    def clustered_count(self) -> int:
        return sum(t.size for t in self.tracks)

    def coverage(self) -> float:
        """Share of works that landed in a direction, in ``[0, 1]``."""
        total = self.clustered_count + len(self.unclustered)
        return self.clustered_count / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{len(self.tracks)} directions covering {self.clustered_count} works "
            f"({self.coverage():.0%}); {len(self.unclustered)} works too isolated "
            f"to place; vocabulary {self.vocabulary_size} terms"
        )


def _weighted_terms(
    works: list[Work], min_df: int, max_df_ratio: float
) -> tuple[list[int], list[int], list[float], list[str]]:
    """TF-IDF weights as parallel (work, term, weight) lists, and the vocabulary.

    Terms appearing in fewer than ``min_df`` documents are dropped as noise, and
    terms in more than ``max_df_ratio`` of documents are dropped because a word
    every paper uses cannot distinguish between them. Weights are not yet
    normalised per work; the index that receives them does that.
    """
    tokenized = [tokenize(w.text()) for w in works]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    n_docs = len(works)
    max_df = max(min_df, int(max_df_ratio * n_docs))
    vocabulary = sorted(
        term
        for term, count in document_frequency.items()
        if min_df <= count <= max_df
    )
    index = {term: i for i, term in enumerate(vocabulary)}
    idf = [math.log((1 + n_docs) / (1 + document_frequency[t])) + 1.0 for t in vocabulary]

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for row, tokens in enumerate(tokenized):
        counts = Counter(t for t in tokens if t in index)
        for term, count in counts.items():
            column = index[term]
            rows.append(row)
            columns.append(column)
            # Sublinear term frequency: a word used ten times is not ten times
            # as indicative as one used once.
            weights.append((1.0 + math.log(count)) * idf[column])
    return rows, columns, weights, vocabulary


class _DenseIndex:
    """Works as dense unit vectors, with every pairwise similarity in memory.

    The faster index on a small corpus. Memory grows with works x vocabulary
    for the vectors and with works squared for the similarities.
    """

    def __init__(self, rows, columns, weights, shape: tuple[int, int]) -> None:
        matrix = np.zeros(shape, dtype=np.float64)
        matrix[rows, columns] = weights
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms  # in place: a second works x vocabulary array is not free
        self.matrix = matrix
        self.similarity = matrix @ matrix.T
        self.corpus_mean = matrix.mean(axis=0)

    def clusters(self, threshold: float) -> list[list[int]]:
        return agglomerate_dense(self.similarity, threshold)

    def mean_vector(self, members: list[int]) -> np.ndarray:
        return self.matrix[members].mean(axis=0)

    def cohesion(self, members: list[int]) -> float:
        """Mean pairwise similarity within a cluster. A singleton scores 1.0."""
        if len(members) < 2:
            return 1.0
        block = self.similarity[np.ix_(members, members)]
        n = len(members)
        return float((block.sum() - np.trace(block)) / (n * (n - 1)))


class _SparseIndex:
    """Works as sparse unit vectors; no pairwise similarity is ever stored.

    Memory grows with the amount of text rather than with works squared, which
    is what lets discovery run on a corpus the dense index cannot hold.
    """

    def __init__(self, rows, columns, weights, shape: tuple[int, int]) -> None:
        matrix = sp.csr_matrix((weights, (rows, columns)), shape=shape, dtype=np.float64)
        norms = np.sqrt(_row_sums(matrix.multiply(matrix)))
        norms[norms == 0] = 1.0
        self.matrix = sp.csr_matrix(sp.diags(1.0 / norms) @ matrix)
        # Each work's similarity to itself: 1 for a unit vector, 0 for a work
        # with no vocabulary terms. Cohesion has to leave these out.
        self.self_similarity = _row_sums(self.matrix.multiply(self.matrix))
        self.corpus_mean = np.asarray(self.matrix.mean(axis=0)).ravel()

    def clusters(self, threshold: float) -> list[list[int]]:
        return agglomerate_sparse(self.matrix, threshold)

    def _sum_vector(self, members: list[int]) -> np.ndarray:
        return np.asarray(self.matrix[members].sum(axis=0)).ravel()

    def mean_vector(self, members: list[int]) -> np.ndarray:
        return self._sum_vector(members) / len(members)

    def cohesion(self, members: list[int]) -> float:
        """Mean pairwise similarity within a cluster. A singleton scores 1.0.

        The similarities among a cluster's members sum to the squared length
        of their summed vector, so no pair is ever looked up individually.
        """
        if len(members) < 2:
            return 1.0
        total = self._sum_vector(members)
        n = len(members)
        between = total @ total - self.self_similarity[members].sum()
        return float(between / (n * (n - 1)))


def _row_sums(matrix) -> np.ndarray:
    return np.asarray(matrix.sum(axis=1)).ravel()


def _distinctive_terms(
    index: _DenseIndex | _SparseIndex, vocabulary: list[str], members: list[int], top: int
) -> list[str]:
    """Terms that distinguish this cluster from the corpus as a whole."""
    distinctiveness = index.mean_vector(members) - index.corpus_mean
    order = np.argsort(distinctiveness)[::-1][:top]
    return [vocabulary[i] for i in order if distinctiveness[i] > 0]


# The dense index refuses above this many works: its memory grows with works
# squared (25k works is roughly 5GB for one similarity matrix, and it holds
# three), and refusing beats dying on an allocation.
MAX_WORKS_DENSE = 12_000

# Above this many works ``method="auto"`` uses the sparse index. The dense
# index is the faster of the two throughout its range -- about three times at
# a thousand works, under half again at this size -- but here it already needs
# around 2GB and that quadruples with each doubling of the corpus.
AUTO_SPARSE_ABOVE = 8_000

METHODS = ("auto", "dense", "sparse")


def _index_class(method: str, n_works: int) -> type[_DenseIndex] | type[_SparseIndex]:
    """The index to cluster ``n_works`` with, honouring an explicit ``method``."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")
    if method == "auto":
        method = "sparse" if n_works > AUTO_SPARSE_ABOVE else "dense"
    if method == "sparse":
        return _SparseIndex
    if n_works > MAX_WORKS_DENSE:
        raise ValueError(
            f"discover() was given {n_works:,} works, above the "
            f"{MAX_WORKS_DENSE:,} limit for dense clustering: one similarity "
            f"matrix alone would need roughly {n_works ** 2 * 8 / 1e9:.1f}GB. "
            f"Use method='sparse' (or 'auto'), which needs no such matrix"
        )
    return _DenseIndex


def _dedupe_ids(tracks: list[DiscoveredTrack]) -> None:
    """Suffix repeated ids: two clusters can slug identically from similar terms."""
    seen: Counter[str] = Counter()
    for track in tracks:
        seen[track.id] += 1
        if seen[track.id] > 1:
            track.id = f"{track.id}-{seen[track.id]}"


def discover(
    works: list[Work],
    threshold: float = 0.16,
    min_cluster_size: int = 3,
    min_df: int = 2,
    max_df_ratio: float = 0.5,
    top_terms: int = 10,
    method: str = "auto",
) -> DiscoveryResult:
    """Cluster ``works`` into candidate research directions.

    Args:
        works: The corpus to search.
        threshold: Average-linkage similarity floor for merging. Higher gives
            more, tighter directions; lower gives fewer, broader ones. The
            default suits abstracts, which share a lot of generic vocabulary.
        min_cluster_size: Clusters smaller than this are quarantined into
            ``unclustered`` rather than reported as directions. Two papers that
            resemble each other are not evidence of a research direction, and
            promoting them to one would flood the ranking with noise.
        method: ``"dense"`` holds every pairwise similarity in memory, which is
            fastest on a small corpus and impossible on a large one.
            ``"sparse"`` stores none, so it scales with the amount of text. The
            two find the same clusters except where similarities tie exactly.
            ``"auto"`` picks by corpus size.
    """
    if not works:
        return DiscoveryResult(tracks=[])
    index_class = _index_class(method, len(works))

    rows, columns, weights, vocabulary = _weighted_terms(works, min_df, max_df_ratio)
    if not vocabulary:
        return DiscoveryResult(tracks=[], unclustered=list(works), threshold=threshold)

    index = index_class(rows, columns, weights, (len(works), len(vocabulary)))
    clusters = index.clusters(threshold)

    tracks: list[DiscoveredTrack] = []
    unclustered: list[Work] = []
    for members in sorted(clusters, key=len, reverse=True):
        if len(members) < min_cluster_size:
            unclustered.extend(works[i] for i in members)
            continue
        terms = _distinctive_terms(index, vocabulary, members, top_terms)
        label = " / ".join(terms[:3]) if terms else f"cluster of {len(members)}"
        tracks.append(
            DiscoveredTrack(
                id="discovered-" + re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-"),
                label=label,
                works=[works[i] for i in members],
                terms=terms,
                cohesion=index.cohesion(members),
            )
        )
    _dedupe_ids(tracks)

    return DiscoveryResult(
        tracks=tracks,
        unclustered=unclustered,
        vocabulary_size=len(vocabulary),
        threshold=threshold,
    )


def evaluate_against_labels(
    result: DiscoveryResult, works: list[Work], label_key: str = "track"
) -> dict:
    """Score discovery against known labels, where a corpus has them.

    Reports homogeneity (do clusters mix labels?), the share of labelled
    directions recovered, and coverage. This exists because an unsupervised step
    that nobody measures is an unsupervised step nobody should trust -- and on a
    corpus with planted labels the measurement is free.
    """
    truth = {w.id: str(w.extra.get(label_key, "unlabelled")) for w in works}
    labels_present = {v for v in truth.values() if v != "unlabelled"}

    purities: list[float] = []
    recovered: set[str] = set()
    for track in result.tracks:
        counts = Counter(truth.get(w.id, "unlabelled") for w in track.works)
        dominant, count = counts.most_common(1)[0]
        purities.append(count / track.size)
        if dominant != "unlabelled":
            recovered.add(dominant)

    return {
        "clusters": len(result.tracks),
        "true_directions": len(labels_present),
        "directions_recovered": len(recovered),
        "recovery_rate": len(recovered) / len(labels_present) if labels_present else 0.0,
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
        "coverage": result.coverage(),
    }


def discover_by_discipline(
    works: list[Work], min_partition: int = 20, **kwargs
) -> DiscoveryResult:
    """Cluster within each discipline, then combine.

    The route to breadth. Partitioning keeps each clustering problem small,
    which matters because clustering time grows with roughly the square of the
    number of works, and it removes a real failure mode: shared methods
    vocabulary pulling unrelated fields together, so that a machine-learning
    paper in cardiology and one in seismology land in the same "direction"
    because both are full of the word "network".

    The cost is that genuinely cross-disciplinary directions get split across
    partitions, which is the opposite failure. Works carrying several disciplines
    are assigned to their first, so the split is deterministic but arbitrary.
    Neither behaviour is hidden: run both and compare if it matters.

    Disciplines with fewer than ``min_partition`` works are pooled together
    rather than each producing a cluster of two.
    """
    partitions: dict[str, list[Work]] = {}
    for work in works:
        key = (work.disciplines or ["_unlabelled"])[0].lower()
        partitions.setdefault(key, []).append(work)

    small: list[Work] = []
    tracks: list[DiscoveredTrack] = []
    unclustered: list[Work] = []
    vocabulary_total = 0

    for name, members in sorted(partitions.items()):
        if len(members) < min_partition:
            small.extend(members)
            continue
        result = discover(members, **kwargs)
        vocabulary_total += result.vocabulary_size
        for track in result.tracks:
            track.id = f"{re.sub(r'[^a-z0-9]+', '-', name)}--{track.id}"
            track.label = f"{track.label} [{name}]"
        tracks.extend(result.tracks)
        unclustered.extend(result.unclustered)

    if len(small) >= min_partition:
        result = discover(small, **kwargs)
        vocabulary_total += result.vocabulary_size
        tracks.extend(result.tracks)
        unclustered.extend(result.unclustered)
    else:
        unclustered.extend(small)

    tracks.sort(key=lambda t: t.size, reverse=True)
    return DiscoveryResult(
        tracks=tracks,
        unclustered=unclustered,
        vocabulary_size=vocabulary_total,
        threshold=kwargs.get("threshold", 0.16),
    )
