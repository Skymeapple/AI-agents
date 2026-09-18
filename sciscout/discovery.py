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


def _build_tfidf(
    works: list[Work], min_df: int, max_df_ratio: float
) -> tuple[np.ndarray, list[str]]:
    """TF-IDF matrix over works, L2-normalised, and its vocabulary.

    Terms appearing in fewer than ``min_df`` documents are dropped as noise, and
    terms in more than ``max_df_ratio`` of documents are dropped because a word
    every paper uses cannot distinguish between them.
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
    if not vocabulary:
        return np.zeros((n_docs, 0)), []

    index = {term: i for i, term in enumerate(vocabulary)}
    matrix = np.zeros((n_docs, len(vocabulary)), dtype=np.float64)
    for row, tokens in enumerate(tokenized):
        counts = Counter(t for t in tokens if t in index)
        for term, count in counts.items():
            # Sublinear term frequency: a word used ten times is not ten times
            # as indicative as one used once.
            matrix[row, index[term]] = 1.0 + math.log(count)

    idf = np.array(
        [math.log((1 + n_docs) / (1 + document_frequency[t])) + 1.0 for t in vocabulary]
    )
    matrix *= idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, vocabulary


def _agglomerate(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """Average-linkage agglomerative clustering down to a similarity floor.

    Merges the closest pair of clusters repeatedly, stopping when the best
    available merge falls below ``threshold``. Average linkage rather than single
    linkage because single linkage chains: one paper bridging two topics would
    otherwise fuse them into one direction.

    Implemented with cached per-row maxima rather than rescanning the whole
    matrix on every merge. The naive form is O(n^3), which takes tens of seconds
    at a thousand works and is hopeless at the scale a broad survey of science
    would produce. Here a merge updates one row and column, then refreshes only
    the rows whose best partner was one of the two clusters just merged.
    """
    n = similarity.shape[0]
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}

    # pair_sum[i, j] is the total similarity between members of i and members of
    # j; dividing by the product of sizes gives average linkage.
    pair_sum = similarity.astype(np.float64).copy()
    np.fill_diagonal(pair_sum, -np.inf)
    sizes = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)

    average = pair_sum.copy()  # sizes are all 1 initially

    def refresh(row: int) -> None:
        values = average[row]
        masked = np.where(active, values, -np.inf)
        masked[row] = -np.inf
        best = int(np.argmax(masked))
        row_best_index[row] = best
        row_best_value[row] = masked[best]

    row_best_index = np.zeros(n, dtype=np.int64)
    row_best_value = np.full(n, -np.inf)
    for row in range(n):
        refresh(row)

    while True:
        candidate_values = np.where(active, row_best_value, -np.inf)
        i = int(np.argmax(candidate_values))
        if not np.isfinite(candidate_values[i]) or candidate_values[i] < threshold:
            break
        j = int(row_best_index[i])
        if not active[j] or i == j:
            refresh(i)
            continue

        # Merge j into i.
        clusters[i].extend(clusters[j])
        del clusters[j]
        pair_sum[i, :] += pair_sum[j, :]
        pair_sum[:, i] = pair_sum[i, :]
        sizes[i] += sizes[j]
        active[j] = False
        pair_sum[i, i] = -np.inf

        # Recompute average linkage for the merged row only.
        with np.errstate(invalid="ignore"):
            denom = sizes[i] * sizes
            safe = np.where(denom > 0, denom, 1.0)
            average[i, :] = np.where(denom > 0, pair_sum[i, :] / safe, -np.inf)
        average[i, i] = -np.inf
        average[:, i] = average[i, :]

        refresh(i)
        # Any row that was pointing at i or j needs a new best partner.
        stale = np.where(active & ((row_best_index == i) | (row_best_index == j)))[0]
        for row in stale:
            if row != i:
                refresh(int(row))

    return [members for members in clusters.values()]


def _label_cluster(
    matrix: np.ndarray, vocabulary: list[str], members: list[int], top: int
) -> list[str]:
    """Terms that distinguish this cluster from the corpus as a whole."""
    if not vocabulary:
        return []
    inside = matrix[members].mean(axis=0)
    outside = matrix.mean(axis=0)
    distinctiveness = inside - outside
    order = np.argsort(distinctiveness)[::-1][:top]
    return [vocabulary[i] for i in order if distinctiveness[i] > 0]


def _cohesion(similarity: np.ndarray, members: list[int]) -> float:
    """Mean pairwise similarity within a cluster. A singleton scores 1.0."""
    if len(members) < 2:
        return 1.0
    block = similarity[np.ix_(members, members)]
    n = len(members)
    return float((block.sum() - np.trace(block)) / (n * (n - 1)))


# Above this many works the dense similarity matrix stops being reasonable:
# memory grows as n^2 (25k works is roughly 5GB) and clustering time with it.
# Rather than grinding or dying on an allocation, discovery refuses and points
# at partitioning, which is both faster and usually better clustering.
MAX_WORKS_DENSE = 12_000


def discover(
    works: list[Work],
    threshold: float = 0.16,
    min_cluster_size: int = 3,
    min_df: int = 2,
    max_df_ratio: float = 0.5,
    top_terms: int = 10,
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
    """
    if not works:
        return DiscoveryResult(tracks=[])
    if len(works) > MAX_WORKS_DENSE:
        raise ValueError(
            f"discover() was given {len(works):,} works, above the "
            f"{MAX_WORKS_DENSE:,} limit for dense clustering: the similarity "
            f"matrix alone would need roughly "
            f"{len(works) ** 2 * 8 / 1e9:.1f}GB. Use discover_by_discipline(), "
            f"which clusters within each discipline and is usually better "
            f"anyway, or narrow the corpus first"
        )

    matrix, vocabulary = _build_tfidf(works, min_df, max_df_ratio)
    if not vocabulary:
        return DiscoveryResult(tracks=[], unclustered=list(works), threshold=threshold)

    similarity = matrix @ matrix.T
    clusters = _agglomerate(similarity, threshold)

    tracks: list[DiscoveredTrack] = []
    unclustered: list[Work] = []
    for members in sorted(clusters, key=len, reverse=True):
        if len(members) < min_cluster_size:
            unclustered.extend(works[i] for i in members)
            continue
        terms = _label_cluster(matrix, vocabulary, members, top_terms)
        label = " / ".join(terms[:3]) if terms else f"cluster of {len(members)}"
        tracks.append(
            DiscoveredTrack(
                id="discovered-" + re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-"),
                label=label,
                works=[works[i] for i in members],
                terms=terms,
                cohesion=_cohesion(similarity, members),
            )
        )

    # Guard against two clusters producing the same slug from similar top terms.
    seen: Counter[str] = Counter()
    for track in tracks:
        seen[track.id] += 1
        if seen[track.id] > 1:
            track.id = f"{track.id}-{seen[track.id]}"

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

    The route to breadth. Partitioning keeps each clustering problem small enough
    for the dense method, and it removes a real failure mode: shared methods
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
