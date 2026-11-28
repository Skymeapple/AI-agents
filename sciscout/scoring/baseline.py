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
from collections import Counter
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
    # Citation counts to leave out of each pool; see :meth:`excluding`.
    _removed: dict[tuple, Counter] = field(default_factory=dict)
    _cache: dict[tuple, list[int]] = field(default_factory=dict)

    def excluding(self, works: list[Work]) -> "CitationBaseline":
        """This baseline with ``works`` taken out of every pool.

        A track must not be normalised against itself. When one direction makes
        up most of a discipline's works in the corpus, the discipline median
        *is* that direction's median, every one of its works scores about 1.0x
        expected, and the impact score says nothing -- worst for exactly the
        directions that dominate their field. Scoring a track against the
        baseline excluding its own works compares it with everyone else.

        The price is that a direction alone in its discipline has no
        same-discipline pool left and falls back to the corpus-wide pool for
        its age, which compares across fields. The provenance of each
        expectation says which pool answered.
        """
        removed: dict[tuple, Counter] = {}
        for work in works:
            age = work.age_years(self.asof)
            if age is None or work.citations is None:
                continue
            bucket = _bucket(age)
            removed.setdefault(("age", bucket), Counter())[work.citations] += 1
            removed.setdefault(("all",), Counter())[work.citations] += 1
            for discipline in work.disciplines or ["_unknown"]:
                key = ("discipline", discipline.lower(), bucket)
                removed.setdefault(key, Counter())[work.citations] += 1
        return CitationBaseline(
            asof=self.asof,
            _by_discipline=self._by_discipline,
            _by_age=self._by_age,
            _all=self._all,
            _removed=removed,
        )

    def _pool(self, key: tuple) -> list[int]:
        if key[0] == "discipline":
            pool = self._by_discipline.get((key[1], key[2]), [])
        elif key[0] == "age":
            pool = self._by_age.get(key[1], [])
        else:
            pool = self._all
        drop = self._removed.get(key)
        if not drop:
            return pool
        if key not in self._cache:
            remaining = Counter(drop)
            kept: list[int] = []
            for value in pool:
                if remaining[value] > 0:
                    remaining[value] -= 1
                else:
                    kept.append(value)
            self._cache[key] = kept
        return self._cache[key]

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
        others = " other" if self._removed else ""
        pool = self._pool(("discipline", discipline.lower(), bucket))
        if len(pool) >= MIN_BUCKET_SIZE:
            return (
                max(0.5, statistics.median(pool)),
                observed(
                    f"median of {len(pool)}{others} works in {discipline} aged "
                    f"{bucket[0]:g}-{bucket[1]:g}y"
                ),
            )
        pool = self._pool(("age", bucket))
        if len(pool) >= MIN_BUCKET_SIZE:
            return (
                max(0.5, statistics.median(pool)),
                observed(
                    f"median of {len(pool)}{others} corpus works aged "
                    f"{bucket[0]:g}-{bucket[1]:g}y (discipline pool too thin)"
                ),
            )
        pool = self._pool(("all",))
        if len(pool) >= MIN_BUCKET_SIZE:
            return (
                max(0.5, statistics.median(pool)),
                observed(
                    f"median of all {len(pool)}{others} corpus works "
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

    def _expected_for(self, work: Work, age: float) -> tuple[float, Provenance]:
        """Expectation for ``work``, from the most specific pool any of its disciplines has.

        A work listed under several disciplines is compared within the first
        one that has an adequate same-age pool, rather than being sent to the
        corpus-wide fallback because its first-listed discipline happens to be
        thin.
        """
        disciplines = work.disciplines or ["_unknown"]
        bucket = _bucket(age)
        for discipline in disciplines:
            if len(self._pool(("discipline", discipline.lower(), bucket))) >= MIN_BUCKET_SIZE:
                return self.expected(discipline, age)
        return self.expected(disciplines[0], age)

    def within_discipline(self, work: Work) -> bool:
        """Whether ``work`` can be compared against its own discipline and age."""
        age = work.age_years(self.asof)
        if age is None:
            return False
        bucket = _bucket(age)
        return any(
            len(self._pool(("discipline", d.lower(), bucket))) >= MIN_BUCKET_SIZE
            for d in (work.disciplines or ["_unknown"])
        )

    def normalised_impact(self, work: Work) -> Estimate[float] | None:
        """Citations relative to expectation. 1.0 means exactly typical.

        Returns None when the work lacks the date or citation count needed --
        an absent value, not a zero. Scoring treats the two very differently.
        """
        age = work.age_years(self.asof)
        if age is None or work.citations is None:
            return None
        expected_count, provenance = self._expected_for(work, age)
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
