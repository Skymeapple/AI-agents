"""Provenance is the platform's core guarantee, so it is tested hardest."""

import pytest

from sciscout.provenance import (
    Estimate,
    Grade,
    ProvenanceLedger,
    Provenance,
    assumed,
    observed,
    sourced,
)


def test_sourced_requires_a_reference():
    """A citation-free 'sourced' value is indistinguishable from an assumption."""
    with pytest.raises(ValueError, match="must name its reference"):
        Provenance(Grade.SOURCED, "some figure")


def test_provenance_requires_detail():
    with pytest.raises(ValueError):
        observed("   ")


def test_assumptions_default_to_needing_review():
    assert assumed("a guess").needs_review is True
    assert observed("a measurement").needs_review is False


def test_grade_ordering_puts_observed_above_assumed():
    assert Grade.ASSUMED < Grade.SOURCED < Grade.OBSERVED


def test_ledger_reports_assumption_share_and_weakest_link():
    ledger = ProvenanceLedger()
    ledger.record(Estimate(1.0, observed("counted"), label="a"))
    ledger.record(Estimate(2.0, sourced("published", "Ref 2020"), label="b"))
    ledger.record(Estimate(3.0, assumed("guessed"), label="c"))

    assert ledger.assumption_share() == pytest.approx(1 / 3)
    # The weakest input bounds the credibility of the whole result.
    assert ledger.weakest() is Grade.ASSUMED
    assert len(ledger.needing_review()) == 1


def test_empty_ledger_claims_nothing():
    """No inputs means no assumptions, not 100% assumption."""
    assert ProvenanceLedger().assumption_share() == 0.0
    assert ProvenanceLedger().weakest() is None


def test_estimates_do_not_support_arithmetic():
    """Combining estimates would silently lose provenance, which is the whole point."""
    a = Estimate(1.0, observed("x"))
    b = Estimate(2.0, assumed("y"))
    with pytest.raises(TypeError):
        a + b  # type: ignore[operator]
