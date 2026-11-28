"""Average-linkage clustering down to a similarity floor, two ways.

Both functions answer the same question -- which works belong together if
clusters keep merging while their average pairwise similarity stays at or above
a threshold -- and on a corpus without exact ties they return the same clusters.
They differ in what they spend:

``agglomerate_dense`` holds every pairwise similarity in memory. It is the
faster of the two on a small corpus, and its memory grows with the square of
the number of works.

``agglomerate_sparse`` never forms that matrix. It works from the sparse
document vectors, so its memory grows with the amount of text. It is slower per
work on a small corpus and is the only one of the two that can run on a large
one.

Average linkage rather than single linkage because single linkage chains: one
paper bridging two topics would otherwise fuse them into one direction.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class _AverageLinkage:
    """Mutable state for average-linkage clustering, with cached row maxima.

    ``pair_sum[i, j]`` is the total similarity between members of ``i`` and
    members of ``j``; dividing by the product of sizes gives average linkage.
    Each active row caches its best partner, so finding the closest pair is a
    scan of one vector rather than of the whole matrix.
    """

    def __init__(self, similarity: np.ndarray) -> None:
        n = similarity.shape[0]
        self.clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
        self.pair_sum = similarity.astype(np.float64)  # astype copies
        np.fill_diagonal(self.pair_sum, -np.inf)
        self.sizes = np.ones(n, dtype=np.float64)
        self.active = np.ones(n, dtype=bool)
        self.average = self.pair_sum.copy()  # sizes are all 1 initially
        self.best_index = np.zeros(n, dtype=np.int64)
        self.best_value = np.full(n, -np.inf)
        for row in range(n):
            self.refresh(row)

    def refresh(self, row: int) -> None:
        """Recompute the cached best partner for ``row``."""
        masked = np.where(self.active, self.average[row], -np.inf)
        masked[row] = -np.inf
        best = int(np.argmax(masked))
        self.best_index[row] = best
        self.best_value[row] = masked[best]

    def closest(self) -> tuple[int, int, float]:
        """The active row with the highest cached similarity, and its partner."""
        candidates = np.where(self.active, self.best_value, -np.inf)
        i = int(np.argmax(candidates))
        return i, int(self.best_index[i]), float(candidates[i])

    def merge(self, i: int, j: int) -> None:
        """Merge cluster ``j`` into ``i`` and repair the caches it invalidates."""
        self.clusters[i].extend(self.clusters.pop(j))
        self.pair_sum[i, :] += self.pair_sum[j, :]
        self.pair_sum[:, i] = self.pair_sum[i, :]
        self.sizes[i] += self.sizes[j]
        self.active[j] = False
        self.pair_sum[i, i] = -np.inf

        # Recompute average linkage for the merged row only.
        with np.errstate(invalid="ignore"):
            denom = self.sizes[i] * self.sizes
            safe = np.where(denom > 0, denom, 1.0)
            self.average[i, :] = np.where(denom > 0, self.pair_sum[i, :] / safe, -np.inf)
        self.average[i, i] = -np.inf
        self.average[:, i] = self.average[i, :]

        self.refresh(i)
        # Any row that was pointing at i or j needs a new best partner.
        pointed_at_merge = (self.best_index == i) | (self.best_index == j)
        for row in np.where(self.active & pointed_at_merge)[0]:
            if row != i:
                self.refresh(int(row))


def agglomerate_dense(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """Cluster from a full similarity matrix, merging the closest pair each time.

    Stops when the best available merge falls below ``threshold``.

    Implemented with cached per-row maxima rather than rescanning the whole
    matrix on every merge. The naive form is O(n^3), which takes tens of seconds
    at a thousand works. Here a merge updates one row and column, then refreshes
    only the rows whose best partner was one of the two clusters just merged.
    """
    linkage = _AverageLinkage(similarity)
    while True:
        i, j, value = linkage.closest()
        if not np.isfinite(value) or value < threshold:
            break
        if not linkage.active[j] or i == j:
            linkage.refresh(i)
            continue
        linkage.merge(i, j)
    return list(linkage.clusters.values())


# Two similarities closer than this are treated as tied. Far above floating
# point noise, far below any difference that means something.
_TIE = 1e-9


class _ChainLinkage:
    """State for clustering from sparse unit vectors without a similarity matrix.

    Rests on one identity: with dot-product similarity, the total similarity
    between two clusters is the dot product of their summed vectors. So a
    cluster is represented by its sum, and its similarity to every other
    cluster comes from one pass over the postings of the terms it contains.
    """

    def __init__(self, matrix: sp.csr_matrix) -> None:
        n = matrix.shape[0]
        self.n = n
        self.postings = matrix.tocsc()  # term -> the works containing it
        self.vectors = matrix.T.tocsc()  # column i is work i's vector
        # Summed vector per merged cluster; a singleton's is its own column.
        self.sums: dict[int, sp.csc_matrix] = {}
        self.members: dict[int, list[int]] = {i: [i] for i in range(n)}
        self.label = np.arange(n)  # work -> the cluster currently holding it
        self.sizes = np.ones(n, dtype=np.float64)
        # A cluster stops being mergeable once nothing reaches the threshold.
        self.mergeable = np.ones(n, dtype=bool)

    def _sum(self, cluster: int) -> sp.csc_matrix:
        total = self.sums.get(cluster)
        return self.vectors[:, cluster] if total is None else total

    def nearest(self, cluster: int, previous: int | None) -> tuple[int, float]:
        """The mergeable cluster most similar to ``cluster``, and that similarity.

        ``previous`` wins a tie. The chain depends on it: two clusters that are
        each other's nearest must recognise it from both sides.
        """
        to_works = self.postings @ self._sum(cluster)  # n x 1, sparse
        pair_sum = np.bincount(
            self.label[to_works.indices], weights=to_works.data, minlength=self.n
        )
        average = pair_sum / (self.sizes[cluster] * self.sizes)
        average[~self.mergeable] = -np.inf
        average[cluster] = -np.inf
        best = int(np.argmax(average))
        if previous is not None and average[previous] >= average[best] - _TIE:
            best = previous
        return best, float(average[best])

    def merge(self, a: int, b: int) -> None:
        """Merge two clusters, keeping the larger one's id so less is relabelled."""
        keep, drop = (a, b) if len(self.members[a]) >= len(self.members[b]) else (b, a)
        total = self._sum(keep) + self._sum(drop)
        self.sums.pop(drop, None)
        self.sums[keep] = total
        moved = self.members.pop(drop)
        self.label[moved] = keep
        self.members[keep].extend(moved)
        self.sizes[keep] += self.sizes[drop]
        self.mergeable[drop] = False

    def retire(self, cluster: int) -> None:
        """Mark a cluster final and free its summed vector."""
        self.mergeable[cluster] = False
        self.sums.pop(cluster, None)


def agglomerate_sparse(matrix: sp.csr_matrix, threshold: float) -> list[list[int]]:
    """Cluster from sparse row vectors, without forming a similarity matrix.

    ``matrix`` holds one L2-normalised row per work.

    This is the nearest-neighbour chain algorithm. Follow each cluster to its
    nearest neighbour, and that one to its own, until two clusters name each
    other; merge those and carry on from what is left of the chain. It reaches
    the same clusters as merging the globally closest pair each time because
    average linkage is reducible: merging two clusters never brings the result
    closer to a third than the closer of the two already was. That same
    property makes the threshold safe to apply early -- a cluster whose nearest
    neighbour is below the threshold can never merge above it, so it is retired
    on the spot.
    """
    linkage = _ChainLinkage(matrix)
    chain: list[int] = []
    start = 0
    while True:
        if not chain:
            # Every mergeable cluster has an id at or after ``start``.
            while start < linkage.n and not linkage.mergeable[start]:
                start += 1
            if start == linkage.n:
                break
            chain.append(start)
        tip = chain[-1]
        previous = chain[-2] if len(chain) > 1 else None
        nearest, value = linkage.nearest(tip, previous)
        if not np.isfinite(value) or value < threshold:
            linkage.retire(chain.pop())
        elif nearest == previous:
            del chain[-2:]
            linkage.merge(tip, nearest)
        else:
            chain.append(nearest)
    # Corpus order, within and between clusters, so the result does not depend
    # on the order merges happened to occur in.
    return sorted(sorted(members) for members in linkage.members.values())
