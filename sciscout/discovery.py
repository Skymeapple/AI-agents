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

import hashlib
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


def _agglomerate(
    pair_sum: np.ndarray, threshold: float, sizes: np.ndarray | None = None
) -> list[list[int]]:
    """Average-linkage agglomerative clustering down to a similarity floor.

    Merges the closest pair of clusters repeatedly, stopping when the best
    available merge falls below ``threshold``. Average linkage rather than single
    linkage because single linkage chains: one paper bridging two topics would
    otherwise fuse them into one direction.

    ``pair_sum[i, j]`` is the total similarity between the members of ``i`` and
    those of ``j``; it is modified in place, so pass a matrix you no longer need.
    Every item starts as a singleton unless ``sizes`` says otherwise, which is how
    already-formed clusters are merged further with exact average linkage.

    One matrix is kept, not three: average linkage is computed from
    ``pair_sum`` and the cluster sizes when a row is scanned, rather than
    stored as a second n-by-n matrix next to a copy of the input. The output
    is unchanged; peak memory falls by two thirds.

    Rows are refreshed lazily. Merging ``i`` and ``j`` gives every other row a
    linkage to the new cluster that is a weighted mean of its linkages to
    ``i`` and ``j``, so it cannot exceed that row's cached best. A cached best
    is therefore always an upper bound, and a row that is stale only needs
    rescanning when it actually reaches the top of the queue. Most never do --
    rows pointing at a big, growing cluster were previously rescanned on
    every merge into it, which is where nearly all the time went.
    """
    n = pair_sum.shape[0]
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}

    np.fill_diagonal(pair_sum, -np.inf)
    sizes = np.ones(n, dtype=np.float64) if sizes is None else sizes.astype(np.float64)
    active = np.ones(n, dtype=bool)
    # A row's cache is fresh while neither it nor its best partner has merged
    # since it was computed.
    version = np.zeros(n, dtype=np.int64)
    seen_version = np.zeros(n, dtype=np.int64)

    row_best_index = np.zeros(n, dtype=np.int64)
    row_best_value = np.full(n, -np.inf)

    def refresh(row: int) -> None:
        with np.errstate(invalid="ignore"):
            values = pair_sum[row] / (sizes[row] * sizes)
        masked = np.where(active, values, -np.inf)
        masked[row] = -np.inf
        best = int(np.argmax(masked))
        row_best_index[row] = best
        row_best_value[row] = masked[best]
        seen_version[row] = version[row] + version[best]

    for row in range(n):
        refresh(row)

    while True:
        candidate_values = np.where(active, row_best_value, -np.inf)
        i = int(np.argmax(candidate_values))
        if not np.isfinite(candidate_values[i]) or candidate_values[i] < threshold:
            break
        j = int(row_best_index[i])
        if not active[j] or i == j or seen_version[i] != version[i] + version[j]:
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
        version[i] += 1
        version[j] += 1

        refresh(i)

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


def _cohesion(matrix: np.ndarray, members: list[int]) -> float:
    """Mean pairwise similarity within a cluster. A singleton scores 1.0.

    Computed from the member vectors' sum rather than their similarity block:
    for rows ``x``, the sum of all pairwise dot products is ``|sum(x)|^2``, and
    the diagonal is the sum of squared norms. The same number, in O(n) memory
    instead of O(n^2) -- which matters once one cluster holds tens of thousands
    of works.
    """
    if len(members) < 2:
        return 1.0
    rows = matrix[members]
    total = rows.sum(axis=0)
    n = len(members)
    return float((total @ total - (rows * rows).sum()) / (n * (n - 1)))


# Up to this many works, clustering is exact average linkage over the full
# similarity matrix: one n-by-n float64 matrix, so 12k works is about 1.2GB.
# Above it, discovery clusters a sample exactly and extends it; see
# _cluster_at_scale for what that changes.
MAX_WORKS_DENSE = 12_000

# How many works the scaled path clusters exactly in each round.
SAMPLE_SIZE = 8_000

# Rows scored against the seed clusters at a time when assigning the rest.
_ASSIGN_CHUNK = 20_000


def _stable_order(works: list[Work]) -> np.ndarray:
    """Row indices in an order fixed by the works' ids, not by corpus order.

    Sampling the first k of this order is a deterministic, unbiased sample: the
    same corpus gives the same sample however it was harvested or sorted.
    """
    keys = [hashlib.sha1(w.id.encode()).hexdigest() for w in works]
    return np.array(sorted(range(len(works)), key=keys.__getitem__), dtype=np.int64)


def _cluster_at_scale(
    matrix: np.ndarray,
    order: np.ndarray,
    threshold: float,
    min_cluster_size: int,
    sample_size: int,
) -> list[list[int]]:
    """Average-linkage clustering for corpora too big for the dense matrix.

    Exact average linkage needs every pairwise similarity, and within one big
    direction nearly every pair is similar, so no sparse graph or partition
    avoids n^2 work. But the answer is a partition into a few dozen directions,
    and placing a work in one only takes its linkage to each direction. So:

    1. Cluster a deterministic sample exactly.
    2. Assign every other work to the sample cluster it has the highest
       average linkage to -- if that clears ``threshold``, the same test a merge
       has to pass. Linkage to a cluster is ``x . sum(cluster) / size``, so this
       costs one matrix product against the cluster sums, never an n-by-n
       matrix.
    3. Works that matched nothing, and sample clusters too small to count, go
       back into the pool and are clustered in the next round. A direction too
       rare to show up in the sample is not lost: its works fail to match any
       seed, collect in the pool, and meet each other there.
    4. Finish with exact average linkage over the clusters themselves, using
       their sums, so clusters formed in different rounds still merge if the
       dense method would have merged them.

    The approximation is in step 2: a work is placed against the sample
    clusters as they stand, not as they grow. It is measured, not assumed --
    see bench/README.md.
    """
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    pool = order
    groups: list[np.ndarray] = []

    while len(pool) > sample_size:
        sample, rest = pool[:sample_size], pool[sample_size:]
        sub = matrix[sample]
        parts = _agglomerate(sub @ sub.T, threshold)
        seeds = [sample[p] for p in parts if len(p) >= min_cluster_size]
        if not seeds:
            groups.extend(sample[p] for p in parts)
            pool = rest
            continue
        small = [sample[p] for p in parts if len(p) < min_cluster_size]

        sums = np.stack([matrix[seed].sum(axis=0) for seed in seeds])
        sizes = np.array([len(seed) for seed in seeds], dtype=np.float64)
        best = np.empty(len(rest), dtype=np.int64)
        value = np.empty(len(rest))
        for start in range(0, len(rest), _ASSIGN_CHUNK):
            rows = rest[start : start + _ASSIGN_CHUNK]
            linkage = (matrix[rows] @ sums.T) / sizes
            best[start : start + len(rows)] = linkage.argmax(axis=1)
            value[start : start + len(rows)] = linkage.max(axis=1)
        placed = value >= threshold
        for k, seed in enumerate(seeds):
            groups.append(np.concatenate([seed, rest[placed & (best == k)]]))

        pool = np.concatenate(small + [rest[~placed]]) if small else rest[~placed]
        pool = pool[np.argsort(rank[pool], kind="stable")]

    if len(pool):
        sub = matrix[pool]
        groups.extend(pool[p] for p in _agglomerate(sub @ sub.T, threshold))

    sums = np.stack([matrix[g].sum(axis=0) for g in groups])
    sizes = np.array([len(g) for g in groups], dtype=np.float64)
    merged = _agglomerate(sums @ sums.T, threshold, sizes=sizes)
    return [np.concatenate([groups[g] for g in m]).tolist() for m in merged]


def discover(
    works: list[Work],
    threshold: float = 0.16,
    min_cluster_size: int = 3,
    min_df: int = 2,
    max_df_ratio: float = 0.5,
    top_terms: int = 10,
    max_exact: int = MAX_WORKS_DENSE,
    sample_size: int = SAMPLE_SIZE,
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
        max_exact: Corpora up to this size are clustered exactly. Larger ones
            take the sampled path in :func:`_cluster_at_scale`, which is close
            to exact but not identical.
        sample_size: Works clustered exactly per round on the sampled path.
    """
    if not works:
        return DiscoveryResult(tracks=[])

    matrix, vocabulary = _build_tfidf(works, min_df, max_df_ratio)
    if not vocabulary:
        return DiscoveryResult(tracks=[], unclustered=list(works), threshold=threshold)

    if len(works) <= max_exact:
        clusters = _agglomerate(matrix @ matrix.T, threshold)
    else:
        clusters = _cluster_at_scale(
            matrix, _stable_order(works), threshold, min_cluster_size, sample_size
        )

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
                cohesion=_cohesion(matrix, members),
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
