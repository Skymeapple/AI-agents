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

import heapq
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

    Dense reference for :func:`_build_tfidf_sparse`, which :func:`discover`
    uses; the tests hold the two to identical values.

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

    Dense reference for :func:`_agglomerate_sparse`, which :func:`discover`
    uses: quadratic memory, but simple enough to check by eye.

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


@dataclass
class _SparseTfidf:
    """TF-IDF rows in compressed sparse row form, numpy only.

    Holds exactly the values :func:`_build_tfidf` would put in its dense matrix
    -- same vocabulary, same sublinear term frequency, same idf, and row norms
    computed by the same numpy call on the same dense rows -- without ever
    allocating ``n_docs x vocabulary``. A real abstract uses a few dozen distinct
    terms out of tens of thousands, so the dense matrix is almost all zeros.
    """

    indptr: np.ndarray  # row r spans indptr[r]:indptr[r + 1]
    indices: np.ndarray  # term index of each stored value, ascending within a row
    data: np.ndarray
    vocabulary: list[str]

    @property
    def n_docs(self) -> int:
        return len(self.indptr) - 1

    def row(self, r: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.indptr[r], self.indptr[r + 1]
        return self.indices[lo:hi], self.data[lo:hi]

    def column_sums(self, rows: list[int] | None = None) -> np.ndarray:
        """Per-term sums over ``rows`` (all rows if None), accumulated in row order.

        Summing in row order matches numpy's ``mean(axis=0)`` over the dense
        matrix value for value, so cluster labels come out identical.
        """
        if rows is None:
            idx, val = self.indices, self.data
        else:
            starts, ends = self.indptr[rows], self.indptr[np.asarray(rows) + 1]
            take = _ranges(starts, ends)
            idx, val = self.indices[take], self.data[take]
        return np.bincount(idx, weights=val, minlength=len(self.vocabulary))


def _ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(s, e)`` for each pair, without a Python loop."""
    lengths = ends - starts
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(starts, lengths) + (np.arange(total) - offsets)


def _build_tfidf_sparse(
    works: list[Work], min_df: int, max_df_ratio: float
) -> _SparseTfidf:
    """Sparse equivalent of :func:`_build_tfidf`."""
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
    idf = np.array(
        [math.log((1 + n_docs) / (1 + document_frequency[t])) + 1.0 for t in vocabulary]
    )

    indptr = np.zeros(n_docs + 1, dtype=np.int64)
    columns: list[int] = []
    values: list[float] = []
    for row, tokens in enumerate(tokenized):
        counts = Counter(t for t in tokens if t in index)
        entries = sorted((index[term], 1.0 + math.log(count)) for term, count in counts.items())
        columns.extend(c for c, _ in entries)
        values.extend(v for _, v in entries)
        indptr[row + 1] = len(columns)
    indices = np.array(columns, dtype=np.int64)
    data = np.array(values, dtype=np.float64)
    if len(data):
        data *= idf[indices]

    # Row norms come from the same np.linalg.norm call on the same dense rows
    # the dense builder uses, a block at a time, so every stored value matches
    # it bit for bit rather than to within rounding.
    norms = np.ones(n_docs)
    width = max(1, len(vocabulary))
    block = max(1, 16_000_000 // width)
    for start in range(0, n_docs, block):
        stop = min(n_docs, start + block)
        dense = np.zeros((stop - start, width))
        take = slice(indptr[start], indptr[stop])
        rows = np.repeat(np.arange(stop - start), np.diff(indptr[start : stop + 1]))
        dense[rows, indices[take]] = data[take]
        norms[start:stop] = np.linalg.norm(dense, axis=1)
    norms[norms == 0] = 1.0
    data /= np.repeat(norms, np.diff(indptr))
    return _SparseTfidf(indptr=indptr, indices=indices, data=data, vocabulary=vocabulary)


def _candidate_pairs(
    tfidf: _SparseTfidf, threshold: float, block_cells: int = 32_000_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every pair of works whose cosine similarity reaches ``threshold``.

    Returns ``(a, b, similarity)`` with ``a < b``. Similarities are computed a
    block of rows at a time and only pairs at or above the floor are kept, so
    memory is bounded by the block, not by ``n^2``.

    Terms are split by document frequency. Frequent terms -- the shared methods
    vocabulary that gives almost every pair a small non-zero similarity -- go
    through a dense matrix product, which BLAS does fastest. Rare terms, the
    topical ones, go through their postings lists: a term used by k works can
    only connect those k works, so its cost is k^2 however large the corpus.

    A tiny tolerance below the floor makes the result a superset: summation
    order differs from the dense product, and a pair sitting exactly on the
    floor must not be dropped by rounding.
    """
    n = tfidf.n_docs
    vocab = len(tfidf.vocabulary)
    floor = threshold - 1e-9
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    if n < 2 or vocab == 0:
        return empty

    df = np.bincount(tfidf.indices, minlength=vocab)
    frequent_cutoff = max(32, n // 50)
    frequent = np.where(df > frequent_cutoff)[0]
    if len(frequent) > 2048:
        frequent = frequent[np.argsort(-df[frequent], kind="stable")[:2048]]
    is_frequent = np.zeros(vocab, dtype=bool)
    is_frequent[frequent] = True
    column_of = np.full(vocab, -1, dtype=np.int64)
    column_of[frequent] = np.arange(len(frequent))

    row_of_entry = np.repeat(np.arange(n), np.diff(tfidf.indptr))
    dense = np.zeros((n, len(frequent)))
    mask = is_frequent[tfidf.indices]
    dense[row_of_entry[mask], column_of[tfidf.indices[mask]]] = tfidf.data[mask]

    # Postings for rare terms: the works using each term, and their weights.
    rare_entries = np.where(~mask)[0]
    order = rare_entries[np.argsort(tfidf.indices[rare_entries], kind="stable")]
    posting_doc = row_of_entry[order]
    posting_weight = tfidf.data[order]
    posting_ptr = np.zeros(vocab + 1, dtype=np.int64)
    np.cumsum(np.bincount(tfidf.indices[order], minlength=vocab), out=posting_ptr[1:])

    rows_per_block = max(16, block_cells // n)
    found_a: list[np.ndarray] = []
    found_b: list[np.ndarray] = []
    found_s: list[np.ndarray] = []
    for start in range(0, n - 1, rows_per_block):
        stop = min(n, start + rows_per_block)
        # Only columns from ``start`` on: the pair (a, b) with a < b is found
        # in a's block.
        width = n - start
        sims = dense[start:stop] @ dense[start:].T

        lo, hi = tfidf.indptr[start], tfidf.indptr[stop]
        entry = np.arange(lo, hi)[~mask[lo:hi]]
        if len(entry):
            terms = tfidf.indices[entry]
            take = _ranges(posting_ptr[terms], posting_ptr[terms + 1])
            counts = posting_ptr[terms + 1] - posting_ptr[terms]
            local_row = np.repeat(row_of_entry[entry] - start, counts)
            weight = np.repeat(tfidf.data[entry], counts) * posting_weight[take]
            column = posting_doc[take] - start
            keep = column >= 0
            flat = local_row[keep] * width + column[keep]
            sims += np.bincount(flat, weights=weight[keep], minlength=sims.size).reshape(
                sims.shape
            )

        r, c = np.nonzero(sims >= floor)
        upper = c > r
        r, c = r[upper], c[upper]
        found_a.append(r + start)
        found_b.append(c + start)
        found_s.append(sims[r, c])

    if not found_a:
        return empty
    return np.concatenate(found_a), np.concatenate(found_b), np.concatenate(found_s)


def _agglomerate_sparse(
    tfidf: _SparseTfidf, threshold: float
) -> tuple[list[list[int]], dict[int, np.ndarray]]:
    """Average-linkage clustering to a similarity floor without an n x n matrix.

    Gives the same clusters as :func:`_agglomerate`, by two facts about average
    linkage over cosine similarity of unit vectors:

    * The average similarity between clusters A and B is
      ``sum(A) . sum(B) / (|A| |B|)``, where ``sum`` adds the members' TF-IDF
      vectors. So a cluster only needs its summed vector, not a row of pairwise
      totals.
    * An average can only reach the floor if at least one member pair does. So
      two clusters can only ever merge if some pair of their members is a
      candidate from :func:`_candidate_pairs`, and every other pair can be
      skipped, not approximated.

    As in the dense version, each cluster keeps its best partner, the next
    merge is the highest of those (lowest row first on ties), and the row it
    came from survives. A heap holds one entry per cluster, tagged with a version so a
    superseded best is skipped when it surfaces. Returns clusters in the dense
    version's order, and each multi-member cluster's summed vector, which
    cohesion reuses.
    """
    n = tfidf.n_docs
    vocab = len(tfidf.vocabulary)
    a, b, sims = _candidate_pairs(tfidf, threshold)

    # average[i][k]: average linkage between clusters i and k, held only for
    # cluster pairs connected by at least one candidate pair of works.
    average: list[dict[int, float]] = [{} for _ in range(n)]
    for i, j, s in zip(a.tolist(), b.tolist(), sims.tolist()):
        average[i][j] = s
        average[j][i] = s

    best_value = [-math.inf] * n
    best_partner = [-1] * n
    version = [0] * n
    heap: list[tuple[float, int, int]] = []

    def refresh(row: int) -> None:
        version[row] += 1
        if not average[row]:
            best_value[row], best_partner[row] = -math.inf, -1
            return
        values = average[row]
        value = max(values.values())
        partner = min(k for k, v in values.items() if v == value)
        best_value[row], best_partner[row] = value, partner
        if value >= threshold:
            heapq.heappush(heap, (-value, row, version[row]))

    for row in range(n):
        refresh(row)

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    size = np.ones(n)
    sum_index = [tfidf.row(i)[0] for i in range(n)]
    sum_value = [tfidf.row(i)[1] for i in range(n)]
    scratch = np.zeros(max(1, vocab))

    while heap:
        negative, i, tag = heapq.heappop(heap)
        if tag != version[i] or i not in members:
            continue
        j = best_partner[i]

        members[i].extend(members.pop(j))
        size[i] += size[j]
        scratch[sum_index[i]] += sum_value[i]
        scratch[sum_index[j]] += sum_value[j]
        merged_index = np.union1d(sum_index[i], sum_index[j])
        sum_index[i], sum_value[i] = merged_index, scratch[merged_index].copy()
        sum_index[j] = sum_value[j] = None

        around = sorted((average[i].keys() | average[j].keys()) - {i, j})
        average[j] = {}
        version[j] += 1
        new_row: dict[int, float] = {}
        if around:
            lengths = np.array([len(sum_index[k]) for k in around])
            gathered = scratch[np.concatenate([sum_index[k] for k in around])]
            products = gathered * np.concatenate([sum_value[k] for k in around])
            dots = np.add.reduceat(products, np.cumsum(lengths) - lengths)
            new_row = dict(zip(around, (dots / (size[i] * size[around])).tolist()))
        scratch[merged_index] = 0.0
        average[i] = new_row
        refresh(i)

        for k, value in new_row.items():
            row = average[k]
            row.pop(j, None)
            row[i] = value
            # Like the dense version, only rows whose best partner just merged
            # are refreshed. Row i already holds every new average, so the
            # next merge is still the global best; this keeps tie-breaking,
            # and so member order, identical when averages are exactly equal.
            if best_partner[k] in (i, j):
                refresh(k)

    sums: dict[int, np.ndarray] = {}
    for root in members:
        if len(members[root]) > 1:
            vector = np.zeros(max(1, vocab))
            vector[sum_index[root]] = sum_value[root]
            sums[root] = vector
    return [members[root] for root in sorted(members)], sums


# Clustering no longer holds an n x n similarity matrix, so this limit is no
# longer about memory: 120k works now cluster in a few minutes within 2.5GB. It
# stays because quality is not established above it. On a synthetic benchmark,
# single-pass clustering at the default floor fused directions sharing a third
# of their vocabulary more often as the corpus grew (ARI 0.985 at 12k, 0.963 at
# 30k, 0.87 at 120k), falling below partitioning by discipline. Raise it only
# with evidence that clustering holds up at the larger size.
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
            f"{MAX_WORKS_DENSE:,} limit for single-pass clustering, past which "
            f"its quality has not been shown to hold. Use discover_by_discipline(), "
            f"which clusters within each discipline and is usually better "
            f"anyway, or narrow the corpus first"
        )

    tfidf = _build_tfidf_sparse(works, min_df, max_df_ratio)
    vocabulary = tfidf.vocabulary
    if not vocabulary:
        return DiscoveryResult(tracks=[], unclustered=list(works), threshold=threshold)

    clusters, sums = _agglomerate_sparse(tfidf, threshold)
    corpus_mean = tfidf.column_sums() / tfidf.n_docs
    row_norm_sq = np.bincount(
        np.repeat(np.arange(tfidf.n_docs), np.diff(tfidf.indptr)),
        weights=tfidf.data**2,
        minlength=tfidf.n_docs,
    )

    tracks: list[DiscoveredTrack] = []
    unclustered: list[Work] = []
    for members in sorted(clusters, key=len, reverse=True):
        if len(members) < min_cluster_size:
            unclustered.extend(works[i] for i in members)
            continue
        inside = tfidf.column_sums(members) / len(members)
        distinctiveness = inside - corpus_mean
        order = np.argsort(distinctiveness)[::-1][:top_terms]
        terms = [vocabulary[i] for i in order if distinctiveness[i] > 0]
        label = " / ".join(terms[:3]) if terms else f"cluster of {len(members)}"
        # Mean pairwise similarity: |sum|^2 counts every ordered pair plus the
        # diagonal, which is each member's own squared norm.
        total = sums[members[0]]
        n_members = len(members)
        cohesion = (float(total @ total) - float(row_norm_sq[members].sum())) / (
            n_members * (n_members - 1)
        )
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
