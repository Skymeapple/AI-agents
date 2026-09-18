"""Field-normalised citation baselines.

Raw citation counts are close to meaningless across fields and ages: a
three-year-old cell-biology paper with 40 citations is unremarkable, the same
count in pure mathematics after one year is extraordinary. Comparing tracks
requires normalising against what is *expected* for a work of that field and age.

Rather than ship an invented lookup table, the baseline is computed from the
ingested corpus: the median citation count of works in the same discipline and
age bucket. That makes the normalisation ``OBSERVED`` -- reproducible from data
we hold -- and it adapts automatically as the corpus grows. The cost is that a
thin corpus gives a noisy baseline, so the baseline reports how many works backed
each bucket and falls back to broader pools when a bucket is too sparse.
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass, field

from ..models import Work
from ..provenance import Estimate, Provenance, assumed, observed

# Age buckets in years. Citation accumulation is steeply non-linear in the first
# years after publication, so the early buckets are narrow.
AGE_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 1e9),
)

# Below this many works, a bucket's median is too noisy to trust and we widen
# the pool rather than pretend to precision.
MIN_BUCKET_SIZE = 5


def _bucket(age: float) -> tuple[float, float]:
    for low, high in AGE_BUCKETS:
        if low <= age < high:
            return (low, high)
    return AGE_BUCKETS[-1]


@dataclass
class CitationBaseline:
    """Expected citation counts by (discipline, age bucket), learned from a corpus."""

    asof: dt.date
    _by_discipline: dict[tuple[str, tuple[float, float]], list[int]] = field(
        default_factory=dict
    )
    _by_age: dict[tuple[float, float], list[int]] = field(default_factory=dict)
    _all: list[int] = field(default_factory=list)

    @classmethod
    def from_works(
        cls, works: list[Work], asof: dt.date | None = None
    ) -> "CitationBaseline":
        """Learn a baseline from every work with both a date and a citation count."""
        asof = asof or dt.date.today()
        baseline = cls(asof=asof)
        for work in works:
            age = work.age_years(asof)
            if age is None or work.citations is None:
                continue
            bucket = _bucket(age)
            baseline._by_age.setdefault(bucket, []).append(work.citations)
            baseline._all.append(work.citations)
            for discipline in work.disciplines or ["_unknown"]:
                key = (discipline.lower(), bucket)
                baseline._by_discipline.setdefault(key, []).append(work.citations)
        return baseline

    def expected(self, discipline: str, age: float) -> tuple[float, Provenance]:
        """Expected citations for a work of this discipline and age.

        Falls back from the discipline+age pool, to age alone, to the whole
        corpus, to a flat prior -- reporting in the provenance which level
        actually answered, so a reader can see how specific the comparison was.
        """
        bucket = _bucket(age)
        pool = self._by_discipline.get((discipline.lower(), bucket), [])
        if len(pool) >= MIN_BUCKET_SIZE:
            return (
                max(0.5, statistics.median(pool)),
                observed(
                    f"median of {len(pool)} works in {discipline} aged "
                    f"{bucket[0]:g}-{bucket[1]:g}y"
                ),
            )
        pool = self._by_age.get(bucket, [])
        if len(pool) >= MIN_BUCKET_SIZE:
            return (
                max(0.5, statistics.median(pool)),
                observed(
                    f"median of {len(pool)} corpus works aged "
                    f"{bucket[0]:g}-{bucket[1]:g}y (discipline pool too thin)"
                ),
            )
        if len(self._all) >= MIN_BUCKET_SIZE:
            return (
                max(0.5, statistics.median(self._all)),
                observed(
                    f"median of all {len(self._all)} corpus works "
                    "(age and discipline pools too thin)"
                ),
            )
        return (
            1.0,
            assumed(
                "corpus too small to learn a citation baseline; using a flat "
                "expectation of 1 citation, which makes the resulting impact "
                "score a raw count rather than a normalised one"
            ),
        )

    def normalised_impact(self, work: Work) -> Estimate[float] | None:
        """Citations relative to expectation. 1.0 means exactly typical.

        Returns None when the work lacks the date or citation count needed --
        an absent value, not a zero. Scoring treats the two very differently.
        """
        age = work.age_years(self.asof)
        if age is None or work.citations is None:
            return None
        discipline = (work.disciplines or ["_unknown"])[0]
        expected_count, provenance = self.expected(discipline, age)
        ratio = work.citations / expected_count
        return Estimate(
            ratio,
            Provenance(
                provenance.grade,
                f"{work.citations} citations vs expected {expected_count:.1f} "
                f"({provenance.detail})",
                provenance.reference,
            ),
            label=f"normalised impact of {work.id}",
        )

    def coverage(self) -> float:
        """Share of buckets that met the minimum size, in ``[0, 1]``.

        A proxy for how much the baseline is really measuring versus falling
        back. Low coverage is a signal to ingest more corpus before trusting
        cross-track impact comparisons.
        """
        if not self._by_discipline:
            return 0.0
        adequate = sum(
            1 for pool in self._by_discipline.values() if len(pool) >= MIN_BUCKET_SIZE
        )
        return adequate / len(self._by_discipline)
