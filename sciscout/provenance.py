"""Provenance-carrying quantities.

The central discipline of this platform: a number is never just a number. Every
quantity used in scoring or investment modelling records *how we came to know
it*, so a downstream reader can tell a measurement from a guess.

Three grades, in descending order of trust:

``OBSERVED``
    Computed from data we actually ingested (citation counts, publication
    volumes, author affiliations). Reproducible from the corpus.
``SOURCED``
    Taken from a named external reference -- a paper, an industry report, a
    filing. Carries a citation. Not reproducible from our corpus, but checkable.
``ASSUMED``
    A judgement call. Carries a rationale explaining the reasoning. This is
    legitimate -- most forward-looking inputs are assumptions -- but it must be
    visible, because a confident-looking ROI built from assumptions is a story,
    not an analysis.

:class:`Estimate` wraps a value with its provenance. The Monte Carlo engine
tracks which grade each sampled input carried, so a run can report what share of
its output variance traces back to assumption rather than evidence.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, TypeVar

T = TypeVar("T")


class Grade(enum.Enum):
    """How a quantity came to be known, in descending order of trust."""

    OBSERVED = "observed"
    SOURCED = "sourced"
    ASSUMED = "assumed"

    @property
    def trust(self) -> int:
        """Higher is more trustworthy. Useful for sorting and aggregation."""
        return {"observed": 3, "sourced": 2, "assumed": 1}[self.value]

    def __lt__(self, other: "Grade") -> bool:
        return self.trust < other.trust


@dataclass(frozen=True)
class Provenance:
    """Where a quantity came from.

    Args:
        grade: Trust level of the quantity.
        detail: For ``OBSERVED``, the computation performed. For ``SOURCED``, the
            citation. For ``ASSUMED``, the rationale for the judgement.
        reference: Optional URL, DOI or identifier backing a ``SOURCED`` value.
        retrieved: Optional ISO date the reference was consulted, so a reader can
            tell a fresh figure from a stale one.
        needs_review: Set when a value is a seed placeholder that the analyst is
            expected to replace before the number is relied upon.
    """

    grade: Grade
    detail: str
    reference: str | None = None
    retrieved: str | None = None
    needs_review: bool = False

    def __post_init__(self) -> None:
        if not self.detail.strip():
            raise ValueError("provenance requires a non-empty detail")
        if self.grade is Grade.SOURCED and not self.reference:
            raise ValueError(
                f"a SOURCED value must name its reference (detail={self.detail!r})"
            )

    def describe(self) -> str:
        """One-line human-readable rendering, for reports and audit tables."""
        parts = [f"{self.grade.value}: {self.detail}"]
        if self.reference:
            parts.append(f"[{self.reference}]")
        if self.retrieved:
            parts.append(f"(retrieved {self.retrieved})")
        if self.needs_review:
            parts.append("** NEEDS REVIEW **")
        return " ".join(parts)


def observed(detail: str) -> Provenance:
    """Provenance for a value computed from ingested data."""
    return Provenance(Grade.OBSERVED, detail)


def sourced(
    detail: str, reference: str, retrieved: str | None = None, needs_review: bool = False
) -> Provenance:
    """Provenance for a value taken from a named external reference."""
    return Provenance(Grade.SOURCED, detail, reference, retrieved, needs_review)


def assumed(detail: str, needs_review: bool = True) -> Provenance:
    """Provenance for a judgement call.

    Defaults to ``needs_review=True``: an assumption is a placeholder for a
    better number until someone decides otherwise.
    """
    return Provenance(Grade.ASSUMED, detail, needs_review=needs_review)


@dataclass(frozen=True)
class Estimate(Generic[T]):
    """A value bound to its provenance.

    Arithmetic is deliberately *not* defined. Combining estimates loses track of
    provenance, and silently-combined provenance is the failure mode this class
    exists to prevent. Unwrap with :attr:`value` at the point of use, and let the
    calling code record how the combination was made.
    """

    value: T
    provenance: Provenance
    unit: str | None = None
    label: str | None = None

    @property
    def grade(self) -> Grade:
        return self.provenance.grade

    def describe(self) -> str:
        name = self.label or "value"
        unit = f" {self.unit}" if self.unit else ""
        return f"{name} = {self.value}{unit}  <- {self.provenance.describe()}"

    def relabel(self, label: str) -> "Estimate[T]":
        """Return a copy carrying ``label``, for values built before naming."""
        return Estimate(self.value, self.provenance, self.unit, label)


@dataclass
class ProvenanceLedger:
    """Collects the estimates behind a result so it can be audited as a whole.

    Scoring and modelling code registers each input it consumes. The ledger then
    answers the question that matters when reading a forecast: how much of this
    rests on evidence, and how much on assumption?
    """

    entries: list[Estimate[Any]] = field(default_factory=list)

    def record(self, estimate: Estimate[Any]) -> Estimate[Any]:
        """Register ``estimate`` and return it unchanged, for inline use."""
        self.entries.append(estimate)
        return estimate

    def extend(self, estimates: Iterable[Estimate[Any]]) -> None:
        for estimate in estimates:
            self.record(estimate)

    def counts(self) -> dict[Grade, int]:
        return {
            grade: sum(1 for e in self.entries if e.grade is grade) for grade in Grade
        }

    def assumption_share(self) -> float:
        """Fraction of registered inputs that are assumptions, in ``[0, 1]``.

        Returns 0.0 for an empty ledger: nothing has been assumed because
        nothing has been claimed.
        """
        if not self.entries:
            return 0.0
        assumed_count = sum(1 for e in self.entries if e.grade is Grade.ASSUMED)
        return assumed_count / len(self.entries)

    def needing_review(self) -> list[Estimate[Any]]:
        """Estimates explicitly flagged as placeholders."""
        return [e for e in self.entries if e.provenance.needs_review]

    def weakest(self) -> Grade | None:
        """The least trustworthy grade present, which bounds the whole result."""
        if not self.entries:
            return None
        return min(e.grade for e in self.entries)

    def audit_table(self) -> str:
        """Markdown table of every input behind a result."""
        if not self.entries:
            return "_No inputs recorded._"
        rows = [
            "| Input | Value | Grade | Basis |",
            "| --- | --- | --- | --- |",
        ]
        for e in self.entries:
            value = f"{e.value:.4g}" if isinstance(e.value, float) else str(e.value)
            if e.unit:
                value = f"{value} {e.unit}"
            basis = e.provenance.detail
            if e.provenance.reference:
                basis += f" ({e.provenance.reference})"
            if e.provenance.needs_review:
                basis += " **needs review**"
            rows.append(
                f"| {e.label or '-'} | {value} | {e.grade.value} | {basis} |"
            )
        return "\n".join(rows)


def parse_provenance(raw: dict | None, context: str) -> Provenance:
    """Build a :class:`Provenance` from a config mapping.

    Shared by the sector priors and by user-supplied scenario files, so that a
    number arriving from a config file is held to exactly the same standard as
    one written in code: it must say what it is and where it came from.

    Args:
        raw: Mapping with ``grade`` and ``detail``, optionally ``reference``,
            ``retrieved`` and ``needs_review``.
        context: What the value describes, prefixed onto the detail so an audit
            table reads sensibly out of context.

    Raises:
        ValueError: If the grade is unrecognised, the detail is missing, or a
            value claims to be ``sourced`` without naming a reference. Failing
            loudly here is deliberate -- a silently-defaulted provenance would
            let an unfounded number pass as a considered one.
    """
    if not raw:
        raise ValueError(
            f"{context}: every value needs a 'source' block stating its grade "
            f"(observed / sourced / assumed) and the basis for it"
        )
    grade_text = str(raw.get("grade", "")).strip().lower()
    try:
        grade = Grade(grade_text)
    except ValueError:
        raise ValueError(
            f"{context}: unknown provenance grade {grade_text!r}; "
            f"expected one of {', '.join(g.value for g in Grade)}"
        ) from None
    detail = " ".join(str(raw.get("detail", "")).split())
    if not detail:
        raise ValueError(f"{context}: provenance needs a 'detail' explaining the basis")
    return Provenance(
        grade=grade,
        detail=f"{context}: {detail}",
        reference=raw.get("reference"),
        retrieved=raw.get("retrieved"),
        needs_review=bool(raw.get("needs_review", grade is Grade.ASSUMED)),
    )
