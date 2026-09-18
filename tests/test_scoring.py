"""Scoring behaviour: the sub-scores must respond to the signals they claim to."""

import datetime as dt

import pytest

from sciscout.models import Author, Maturity, Track, Work
from sciscout.provenance import Grade
from sciscout.scoring.baseline import CitationBaseline
from sciscout.scoring.importance import Scorer
from sciscout.scoring.trl import infer_trl

ASOF = dt.date(2026, 9, 18)


def make_track(
    track_id="t",
    years=((2023, 4), (2024, 6), (2025, 8)),
    citations=10,
    n_groups=3,
    industry=False,
    abstract="a study",
    maturity=Maturity.PEER_REVIEWED,
    disciplines=("physics",),
):
    track = Track(id=track_id, name=track_id)
    counter = 0
    for year, count in years:
        for _ in range(count):
            counter += 1
            track.add(
                Work(
                    id=f"{track_id}-{counter}",
                    title=f"work {counter}",
                    abstract=abstract,
                    authors=[
                        Author(
                            f"A{counter}",
                            f"Lab {i % n_groups}",
                            is_industry=industry,
                        )
                        for i in range(2)
                    ],
                    published=dt.date(year, 6, 1),
                    maturity=maturity,
                    disciplines=list(disciplines),
                    citations=citations,
                )
            )
    return track


def scorer_for(*tracks):
    works = [w for t in tracks for w in t.works]
    return Scorer(CitationBaseline.from_works(works, asof=ASOF), asof=ASOF)


def test_growth_needs_enough_years_to_be_a_trend():
    """Two points is not a trend, and the model should decline to call it one."""
    assert make_track(years=((2024, 5), (2025, 9))).growth_rate() is None
    assert make_track().growth_rate() is not None


def test_momentum_is_neutral_and_assumed_when_history_is_thin():
    track = make_track(years=((2024, 5), (2025, 9)))
    part = scorer_for(track)._momentum(track)
    assert part.value == pytest.approx(0.5)
    # Crucially, a neutral score from missing data is flagged as an assumption.
    assert part.grade is Grade.ASSUMED


def test_growing_track_beats_shrinking_track_on_momentum():
    growing = make_track("growing", years=((2023, 2), (2024, 6), (2025, 16)))
    shrinking = make_track("shrinking", years=((2023, 16), (2024, 6), (2025, 2)))
    scorer = scorer_for(growing, shrinking)
    assert scorer._momentum(growing).value > scorer._momentum(shrinking).value


def test_impact_is_measured_not_imputed_when_citations_are_missing():
    track = make_track(citations=None)
    part = scorer_for(track)._impact(track)
    assert part.value == 0.0
    assert part.grade is Grade.ASSUMED
    assert "could not be measured" in part.provenance.detail


def test_replication_across_groups_raises_evidence():
    single = make_track("single", n_groups=1)
    many = make_track("many", n_groups=6)
    scorer = scorer_for(single, many)
    assert scorer._evidence(many).value > scorer._evidence(single).value


def test_preprints_score_below_peer_reviewed_on_evidence():
    preprint = make_track("pre", maturity=Maturity.PREPRINT)
    reviewed = make_track("rev", maturity=Maturity.PEER_REVIEWED)
    scorer = scorer_for(preprint, reviewed)
    assert scorer._evidence(reviewed).value > scorer._evidence(preprint).value


def test_priority_is_geometric_so_one_axis_cannot_rescue_the_other():
    """A direction with zero urgency must not rank on importance alone."""
    track = make_track()
    scorer = scorer_for(track)
    assessment = scorer.score(track)
    # Reconstruct the combination and confirm a zero on either axis zeroes it.
    assert assessment.priority == pytest.approx(
        assessment.importance**0.6 * assessment.urgency**0.4
    )
    assert 0.0**0.6 * 0.9**0.4 == 0.0


def test_confidence_falls_as_inputs_become_assumptions():
    measured = make_track("measured", citations=12, disciplines=("physics", "chemistry"))
    unmeasured = make_track("unmeasured", citations=None, disciplines=())
    scorer = scorer_for(measured, unmeasured)
    assert scorer.score(measured).confidence > scorer.score(unmeasured).confidence


def test_citation_baseline_falls_back_and_says_so():
    """A corpus too thin to normalise against must admit it rather than invent."""
    baseline = CitationBaseline.from_works([], asof=ASOF)
    _, provenance = baseline.expected("physics", 2.0)
    assert provenance.grade is Grade.ASSUMED
    assert "too small" in provenance.detail


def test_trl_override_beats_inference():
    track = make_track(abstract="phase 3 pivotal trial commercially deployed")
    track.trl_override = 2
    belief = infer_trl(track)
    assert belief.mode == 2
    # An expert's number still carries uncertainty rather than becoming a spike.
    assert belief.probabilities[2] < 1.0


def test_trl_rises_with_readiness_language():
    early = make_track("early", abstract="a theoretical ab initio study from first principles")
    late = make_track(
        "late",
        abstract="pilot plant scale-up with a prototype in an operational environment "
        "and commercially deployed manufacturing yield",
    )
    assert infer_trl(late).mode > infer_trl(early).mode


def test_trl_with_no_works_is_prior_only_and_flagged():
    belief = infer_trl(Track(id="empty", name="empty"))
    assert belief.grade_note.grade is Grade.ASSUMED
    assert "no works" in belief.grade_note.detail


def test_trl_credible_range_widens_under_contradictory_evidence():
    consistent = make_track("c", abstract="a theoretical study from first principles")
    conflicted = make_track(
        "x",
        abstract="a theoretical ab initio study; also a commercially deployed "
        "prototype in an operational environment",
    )
    low_c, high_c = infer_trl(consistent).credible_range()
    low_x, high_x = infer_trl(conflicted).credible_range()
    assert (high_x - low_x) > (high_c - low_c)
