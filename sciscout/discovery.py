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

Two things go beyond the purely lexical, dense method:

*Shared references.* Where works carry their reference lists, two works that
cite the same prior work (or one of which cites the other) gain similarity for
it. This is the one signal in the corpus that survives a change of vocabulary:
groups describing the same idea in different words still stand on the same
literature. It only ever adds evidence, so a corpus without reference lists
clusters exactly as it would have.

*Scale.* Average linkage over cosine similarity has a property that makes it
tractable well past the point where an n-by-n matrix fits in memory: the mean
pairwise similarity between two groups of vectors is the dot product of their
sums, divided by the product of their sizes. So the corpus is first cut into
many small micro-clusters, and average linkage then runs over those, exactly,
from their sum vectors. Memory is (micro-clusters x vocabulary) rather than
(works x works).
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
    # Mean pairwise similarity within the cluster. Lexical cosine, plus
    # ``citation_weight`` times the reference cosine where works carry reference
    # lists -- so with references it can exceed 1.
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


@dataclass
class _Features:
    """Sparse row-wise feature matrix (CSR), lexical columns then reference columns."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n_cols: int
    vocabulary: list[str]  # the lexical columns only
    sq_norms: np.ndarray  # squared norm of each row

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    @property
    def n_lexical(self) -> int:
        return len(self.vocabulary)

    def dense(self, start: int = 0, stop: int | None = None, dtype=np.float64) -> np.ndarray:
        stop = self.n_rows if stop is None else stop
        block = np.zeros((stop - start, self.n_cols), dtype=dtype)
        lo, hi = self.indptr[start], self.indptr[stop]
        rows = np.repeat(np.arange(stop - start), np.diff(self.indptr[start : stop + 1]))
        block[rows, self.indices[lo:hi]] = self.data[lo:hi]
        return block

    def project(self, basis: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """``self @ basis`` without densifying the whole matrix."""
        out = np.zeros((self.n_rows, basis.shape[1]), dtype=np.float32)
        for start in range(0, self.n_rows, chunk):
            stop = min(self.n_rows, start + chunk)
            out[start:stop] = self.dense(start, stop, dtype=np.float32) @ basis
        return out

    def group_sums(self, labels: np.ndarray, n_groups: int) -> np.ndarray:
        """Sum of rows per group, as a dense (groups x columns) matrix."""
        sums = np.zeros((n_groups, self.n_cols), dtype=np.float32)
        rows = np.repeat(np.arange(self.n_rows), np.diff(self.indptr))
        keep = labels[rows] >= 0
        np.add.at(sums, (labels[rows][keep], self.indices[keep]), self.data[keep])
        return sums


def _weighted_rows(
    documents: list[list[str]], min_df: int, max_df: int | None, max_terms: int | None,
    sublinear: bool,
) -> tuple[list[dict[int, float]], list[str]]:
    """TF-IDF rows, L2-normalised, as ``{column: weight}`` dicts, and the vocabulary."""
    n_docs = len(documents)
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    kept = [
        term
        for term, count in document_frequency.items()
        if count >= min_df and (max_df is None or count <= max_df)
    ]
    if max_terms is not None and len(kept) > max_terms:
        # Keep the most widely used terms: a term shared by more documents can
        # link more of them. Alphabetical tie-break keeps it deterministic.
        kept = sorted(kept, key=lambda t: (-document_frequency[t], t))[:max_terms]
    vocabulary = sorted(kept)
    index = {term: i for i, term in enumerate(vocabulary)}
    idf = [
        math.log((1 + n_docs) / (1 + document_frequency[t])) + 1.0 for t in vocabulary
    ]
    rows: list[dict[int, float]] = []
    for tokens in documents:
        counts = Counter(index[t] for t in tokens if t in index)
        # Sublinear term frequency: a word used ten times is not ten times
        # as indicative as one used once.
        row = {
            col: ((1.0 + math.log(count)) if sublinear else 1.0) * idf[col]
            for col, count in counts.items()
        }
        norm = math.sqrt(sum(v * v for v in row.values()))
        rows.append({col: v / norm for col, v in row.items()} if norm else {})
    return rows, vocabulary


def _build_features(
    works: list[Work],
    min_df: int,
    max_df_ratio: float,
    citation_weight: float = 0.0,
    max_vocabulary: int | None = None,
) -> _Features:
    """Feature rows for ``works``: TF-IDF text, plus shared references if present.

    Lexical terms appearing in fewer than ``min_df`` documents are dropped as
    noise, and terms in more than ``max_df_ratio`` of documents are dropped
    because a word every paper uses cannot distinguish between them.

    The reference block gives each work one column per work it cites *and one
    for itself*, so that a work citing another shares a column with it just as
    two works citing a common third do. Columns only one work touches are
    dropped, since they link nothing. The block is scaled so that a pair's
    similarity is ``lexical cosine + citation_weight * reference cosine``.
    """
    n_docs = len(works)
    max_df = max(min_df, int(max_df_ratio * n_docs))
    lexical, vocabulary = _weighted_rows(
        [tokenize(w.text()) for w in works], min_df, max_df, max_vocabulary, sublinear=True
    )

    reference_rows: list[dict[int, float]] = [{} for _ in works]
    n_reference_cols = 0
    if citation_weight > 0 and any(w.references for w in works):
        rows, reference_ids = _weighted_rows(
            [sorted(set(w.references) | {w.id}) for w in works],
            # A reference half the corpus cites says as little about direction
            # as a word half the corpus uses.
            min_df=2, max_df=max(2, max_df), max_terms=None, sublinear=False,
        )
        scale = math.sqrt(citation_weight)
        reference_rows = [{c: v * scale for c, v in row.items()} for row in rows]
        n_reference_cols = len(reference_ids)

    offset = len(vocabulary)
    indptr = np.zeros(n_docs + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[float] = []
    sq_norms = np.zeros(n_docs)
    for i, (lex, ref) in enumerate(zip(lexical, reference_rows)):
        for col in sorted(lex):
            indices.append(col)
            data.append(lex[col])
        for col in sorted(ref):
            indices.append(offset + col)
            data.append(ref[col])
        indptr[i + 1] = len(indices)
        sq_norms[i] = sum(v * v for v in lex.values()) + sum(v * v for v in ref.values())
    return _Features(
        indptr=indptr,
        indices=np.asarray(indices, dtype=np.int64),
        data=np.asarray(data, dtype=np.float64),
        n_cols=offset + n_reference_cols,
        vocabulary=vocabulary,
        sq_norms=sq_norms,
    )


def _build_tfidf(
    works: list[Work], min_df: int, max_df_ratio: float
) -> tuple[np.ndarray, list[str]]:
    """Dense TF-IDF matrix over works, L2-normalised, and its vocabulary."""
    features = _build_features(works, min_df, max_df_ratio)
    return features.dense(), features.vocabulary


def _agglomerate(
    pair_sum: np.ndarray, threshold: float, sizes: np.ndarray | None = None
) -> list[list[int]]:
    """Average-linkage agglomerative clustering down to a similarity floor.

    Merges the closest pair of clusters repeatedly, stopping when the best
    available merge falls below ``threshold``. Average linkage rather than single
    linkage because single linkage chains: one paper bridging two topics would
    otherwise fuse them into one direction.

    ``pair_sum[i, j]`` is the total similarity between members of item ``i`` and
    members of item ``j``, and ``sizes`` how many members each item starts with.
    For individual works that is just the similarity matrix and all ones; for
    micro-clusters it is the dot product of their sum vectors, which makes the
    linkage between them exact rather than an approximation from centroids.

    Implemented with cached per-row maxima rather than rescanning the whole
    matrix on every merge. The naive form is O(n^3), which takes tens of seconds
    at a thousand works and is hopeless at the scale a broad survey of science
    would produce. Here a merge updates one row and column, then refreshes only
    the rows whose best partner was one of the two clusters just merged.
    """
    n = pair_sum.shape[0]
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}

    pair_sum = pair_sum.astype(np.float64).copy()
    np.fill_diagonal(pair_sum, -np.inf)
    sizes = np.ones(n, dtype=np.float64) if sizes is None else sizes.astype(np.float64).copy()
    active = np.ones(n, dtype=bool)

    average = pair_sum / np.outer(sizes, sizes)

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


def _distinctive_terms(
    inside_mean: np.ndarray, corpus_mean: np.ndarray, vocabulary: list[str], top: int
) -> list[str]:
    """Terms that distinguish a cluster from the corpus as a whole."""
    if not vocabulary:
        return []
    distinctiveness = inside_mean - corpus_mean
    order = np.argsort(distinctiveness)[::-1][:top]
    return [vocabulary[i] for i in order if distinctiveness[i] > 0]


# Above this many works the dense similarity matrix stops being reasonable:
# memory grows as n^2 (25k works is roughly 5GB) and clustering time with it.
# Past it, discovery switches to micro-clusters instead of grinding or dying on
# an allocation.
MAX_WORKS_DENSE = 12_000

# The scalable path's memory is (micro-clusters x vocabulary), so both are capped.
MAX_MICRO_CLUSTERS = 4_000
MAX_VOCABULARY_SCALABLE = 20_000

_PROJECTION_DIMENSIONS = 256
_PROJECTION_SEED = 20260918


def _micro_cluster(features: _Features, n_clusters: int, iterations: int = 8) -> np.ndarray:
    """Cut the corpus into many small, tight groups. Returns a label per work.

    Spherical k-means on a random projection of the feature rows. The
    projection only has to be good enough to put near-duplicates together --
    what happens *between* micro-clusters is decided afterwards, exactly, in
    the original feature space. Deterministic: fixed projection seed, centroids
    initialised from evenly spaced works. Works with no features get -1.
    """
    rng = np.random.default_rng(_PROJECTION_SEED)
    basis = rng.standard_normal((features.n_cols, _PROJECTION_DIMENSIONS)).astype(np.float32)
    points = features.project(basis)
    norms = np.linalg.norm(points, axis=1)
    usable = np.where(norms > 0)[0]
    labels = np.full(features.n_rows, -1, dtype=np.int64)
    if len(usable) == 0:
        return labels
    points = points[usable] / norms[usable, None]

    k = max(1, min(n_clusters, len(usable)))
    centroids = points[np.linspace(0, len(usable) - 1, k).astype(int)].copy()
    assigned = np.zeros(len(usable), dtype=np.int64)
    for _ in range(iterations):
        for start in range(0, len(usable), 8192):
            block = points[start : start + 8192]
            assigned[start : start + len(block)] = np.argmax(block @ centroids.T, axis=1)
        sums = np.zeros_like(centroids)
        np.add.at(sums, assigned, points)
        lengths = np.linalg.norm(sums, axis=1)
        moved = lengths > 0
        centroids[moved] = sums[moved] / lengths[moved, None]

    _, compact = np.unique(assigned, return_inverse=True)
    labels[usable] = compact
    return labels


def _cluster_dense(features: _Features, threshold: float):
    """Exact average linkage over every pair of works."""
    matrix = features.dense()
    similarity = matrix @ matrix.T
    groups = _agglomerate(similarity, threshold)
    lexical = matrix[:, : features.n_lexical]
    corpus_mean = lexical.mean(axis=0) if features.n_lexical else np.zeros(0)
    for members in groups:
        n = len(members)
        block = similarity[np.ix_(members, members)]
        cohesion = 1.0 if n < 2 else float((block.sum() - np.trace(block)) / (n * (n - 1)))
        inside = lexical[members].mean(axis=0) if features.n_lexical else np.zeros(0)
        yield members, cohesion, inside, corpus_mean


def _cluster_scalable(features: _Features, threshold: float, micro_cluster_size: int):
    """Average linkage over micro-clusters, from their sum vectors."""
    target = min(MAX_MICRO_CLUSTERS, max(1, features.n_rows // max(1, micro_cluster_size)))
    labels = _micro_cluster(features, target)
    n_micro = int(labels.max()) + 1 if (labels >= 0).any() else 0
    orphans = np.where(labels < 0)[0].tolist()
    lexical_cols = features.n_lexical
    for orphan in orphans:
        yield [orphan], 1.0, np.zeros(lexical_cols), np.zeros(lexical_cols)
    if n_micro == 0:
        return

    sums = features.group_sums(labels, n_micro)
    sizes = np.bincount(labels[labels >= 0], minlength=n_micro).astype(np.float64)
    sq_norms = np.bincount(
        labels[labels >= 0], weights=features.sq_norms[labels >= 0], minlength=n_micro
    )
    pair_sum = (sums @ sums.T).astype(np.float64)
    members_of = [[] for _ in range(n_micro)]
    for work, label in enumerate(labels.tolist()):
        if label >= 0:
            members_of[label].append(work)

    corpus_mean = sums[:, :lexical_cols].sum(axis=0) / max(1.0, sizes.sum())
    for group in _agglomerate(pair_sum, threshold, sizes):
        n = float(sizes[group].sum())
        total = float(pair_sum[np.ix_(group, group)].sum())
        cohesion = 1.0 if n < 2 else (total - float(sq_norms[group].sum())) / (n * (n - 1))
        inside = sums[group][:, :lexical_cols].sum(axis=0) / n
        yield [w for g in group for w in members_of[g]], cohesion, inside, corpus_mean


def discover(
    works: list[Work],
    threshold: float = 0.16,
    min_cluster_size: int = 3,
    min_df: int = 2,
    max_df_ratio: float = 0.5,
    top_terms: int = 10,
    citation_weight: float = 1.0,
    method: str = "auto",
    micro_cluster_size: int = 25,
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
        citation_weight: How much a shared reference list counts beside shared
            vocabulary; a pair's similarity is the lexical cosine plus this
            times the cosine of their reference lists. Has no effect on works
            without ``references``. Zero switches the signal off.
        method: ``"dense"`` compares every pair of works and is exact;
            ``"scalable"`` clusters micro-clusters of about
            ``micro_cluster_size`` works; ``"auto"`` uses dense up to
            ``MAX_WORKS_DENSE`` works and scalable beyond.
    """
    if method not in {"auto", "dense", "scalable"}:
        raise ValueError(f"unknown method {method!r}; expected auto, dense or scalable")
    if not works:
        return DiscoveryResult(tracks=[])
    if method == "auto":
        method = "dense" if len(works) <= MAX_WORKS_DENSE else "scalable"
    if method == "dense" and len(works) > MAX_WORKS_DENSE:
        raise ValueError(
            f"dense clustering was asked for {len(works):,} works, above the "
            f"{MAX_WORKS_DENSE:,} limit: the similarity matrix alone would need "
            f"roughly {len(works) ** 2 * 8 / 1e9:.1f}GB. Use method='scalable' "
            f"(or 'auto'), or discover_by_discipline()"
        )

    features = _build_features(
        works,
        min_df,
        max_df_ratio,
        citation_weight=citation_weight,
        max_vocabulary=MAX_VOCABULARY_SCALABLE if method == "scalable" else None,
    )
    if features.n_cols == 0:
        return DiscoveryResult(tracks=[], unclustered=list(works), threshold=threshold)

    if method == "dense":
        clusters = list(_cluster_dense(features, threshold))
    else:
        clusters = list(_cluster_scalable(features, threshold, micro_cluster_size))

    tracks: list[DiscoveredTrack] = []
    unclustered: list[Work] = []
    for members, cohesion, inside, corpus_mean in sorted(
        clusters, key=lambda c: len(c[0]), reverse=True
    ):
        if len(members) < min_cluster_size:
            unclustered.extend(works[i] for i in members)
            continue
        terms = _distinctive_terms(inside, corpus_mean, features.vocabulary, top_terms)
        label = " / ".join(terms[:3]) if terms else f"cluster of {len(members)}"
        tracks.append(
            DiscoveredTrack(
                id="discovered-" + re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-"),
                label=label,
                works=[works[i] for i in members],
                terms=terms,
                cohesion=cohesion,
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
        vocabulary_size=len(features.vocabulary),
        threshold=threshold,
    )


def _contingency(pairs: list[tuple[str, str]]) -> tuple[Counter, Counter, Counter]:
    return Counter(pairs), Counter(a for a, _ in pairs), Counter(b for _, b in pairs)


def _adjusted_rand(pairs: list[tuple[str, str]]) -> float:
    """Adjusted Rand index between two labellings given as (a, b) per item."""
    n = len(pairs)
    if n < 2:
        return 1.0
    joint, left, right = _contingency(pairs)
    choose2 = lambda x: x * (x - 1) / 2.0  # noqa: E731
    index = sum(choose2(c) for c in joint.values())
    left_sum = sum(choose2(c) for c in left.values())
    right_sum = sum(choose2(c) for c in right.values())
    expected = left_sum * right_sum / choose2(n)
    maximum = (left_sum + right_sum) / 2.0
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def _homogeneity_completeness(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    """Homogeneity and completeness of (truth, predicted) pairs."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    joint, truth, predicted = _contingency(pairs)

    def entropy(counts: Counter) -> float:
        return -sum(c / n * math.log(c / n) for c in counts.values())

    h_truth, h_predicted = entropy(truth), entropy(predicted)
    h_truth_given_predicted = -sum(
        c / n * math.log(c / predicted[p]) for (_, p), c in joint.items()
    )
    h_predicted_given_truth = -sum(
        c / n * math.log(c / truth[t]) for (t, _), c in joint.items()
    )
    homogeneity = 1.0 if h_truth == 0 else 1.0 - h_truth_given_predicted / h_truth
    completeness = 1.0 if h_predicted == 0 else 1.0 - h_predicted_given_truth / h_predicted
    return homogeneity, completeness


def _assignments(result: DiscoveryResult) -> dict[str, str]:
    """Cluster id per work id. Each unplaced work is a cluster of its own."""
    assigned = {w.id: t.id for t in result.tracks for w in t.works}
    assigned.update({w.id: f"_unplaced:{w.id}" for w in result.unclustered})
    return assigned


def evaluate_against_labels(
    result: DiscoveryResult, works: list[Work], label_key: str = "track"
) -> dict:
    """Score discovery against known labels, where a corpus has them.

    This exists because an unsupervised step that nobody measures is an
    unsupervised step nobody should trust -- and on a corpus with planted labels
    the measurement is free.

    Purity and recovery are reported because they are easy to read, but they
    cannot be the whole score: both are satisfied by shattering every true
    direction into a dozen pure fragments. So the result also carries
    *completeness* (is each true direction kept in one cluster?), its harmonic
    mean with homogeneity (the V-measure), and the adjusted Rand index, which
    is zero for a random labelling and penalises splitting and merging alike.
    These three are computed over every labelled work, with each unplaced work
    counted as a cluster of one, so quarantining works is not free either.
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

    assigned = _assignments(result)
    pairs = [
        (label, assigned.get(work_id, f"_missing:{work_id}"))
        for work_id, label in truth.items()
        if label != "unlabelled"
    ]
    homogeneity, completeness = _homogeneity_completeness(pairs)
    v_measure = (
        2 * homogeneity * completeness / (homogeneity + completeness)
        if homogeneity + completeness > 0
        else 0.0
    )

    return {
        "clusters": len(result.tracks),
        "true_directions": len(labels_present),
        "directions_recovered": len(recovered),
        "recovery_rate": len(recovered) / len(labels_present) if labels_present else 0.0,
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
        "coverage": result.coverage(),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "adjusted_rand": _adjusted_rand(pairs),
    }


def agreement(first: DiscoveryResult, second: DiscoveryResult) -> float:
    """Adjusted Rand index between two clusterings of the same works.

    Needs no labels, so it works on a real corpus: 1.0 means the two runs
    grouped the works identically, 0 that they agree no better than chance.
    """
    a, b = _assignments(first), _assignments(second)
    return _adjusted_rand([(a[key], b[key]) for key in sorted(a.keys() & b.keys())])


def threshold_sensitivity(
    works: list[Work],
    thresholds: tuple[float, ...] = (0.10, 0.13, 0.16, 0.19, 0.22),
    reference: float = 0.16,
    **kwargs,
) -> list[dict]:
    """How much the directions found depend on the similarity floor.

    The threshold is the one free parameter that decides what counts as a
    direction, and on a real corpus there are no labels to tune it against.
    What can be measured is whether the answer is stable around the value in
    use. Each row reports the number of directions, coverage, and agreement
    (adjusted Rand) with the clustering at ``reference``. Agreement that stays
    near 1 means the directions are a property of the corpus; agreement that
    falls away quickly means they are a property of the threshold.
    """
    baseline = discover(works, threshold=reference, **kwargs)
    rows: list[dict] = []
    for threshold in thresholds:
        result = (
            baseline if threshold == reference else discover(works, threshold=threshold, **kwargs)
        )
        rows.append(
            {
                "threshold": threshold,
                "clusters": len(result.tracks),
                "coverage": result.coverage(),
                "agreement_with_reference": agreement(baseline, result),
            }
        )
    return rows


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
