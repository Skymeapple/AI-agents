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


def test_trl_signal_strength_counts_works_not_mentions():
    """One paper repeating a phrase must not read as the whole track saying it."""
    quiet = [Work(id=f"q{i}", title="Electrolyte study", abstract="Conductivity data.") for i in range(9)]
    loud = Work(id="loud", title="A prototype", abstract="prototype " * 30)
    spread = [Work(id=f"s{i}", title="A prototype", abstract="Field trial data.") for i in range(10)]

    one_loud = infer_trl(Track(id="a", name="A", works=quiet + [loud]))
    everywhere = infer_trl(Track(id="b", name="B", works=spread))
    assert everywhere.probabilities[7] > one_loud.probabilities[7]
    assert one_loud.probabilities[7] < 0.10


# -- a track is not its own baseline -----------------------------------------


def test_a_track_that_dominates_its_discipline_is_not_normalised_against_itself():
    """The failure this guards against: a field's star scoring as merely typical.

    The star is the only track in its discipline, so the discipline's median
    citation count is the star's own median and every star paper looks like
    exactly 1.0x expected. Compared against everyone else instead, its works
    are ten times as cited as works of the same age.
    """
    star = make_track("star", citations=200, disciplines=("chemistry",))
    crowd = [
        make_track(f"c{i}", citations=20, disciplines=("physics",)) for i in range(3)
    ]
    works = [w for t in [star] + crowd for w in t.works]
    baseline = CitationBaseline.from_works(works, asof=ASOF)

    against_itself = [baseline.normalised_impact(w).value for w in star.works]
    assert max(against_itself) == pytest.approx(1.0)

    scorer = Scorer(baseline, asof=ASOF)
    star_impact = scorer.score(star).importance_parts["impact"]
    crowd_impact = scorer.score(crowd[0]).importance_parts["impact"]
    assert star_impact.value > crowd_impact.value + 0.3
    # It could only be compared across fields, and the provenance says so.
    assert "18 against the whole corpus" in star_impact.provenance.detail


def test_excluding_works_leaves_the_shared_baseline_untouched():
    a = make_track("a", citations=5)
    b = make_track("b", citations=50)
    baseline = CitationBaseline.from_works(a.works + b.works, asof=ASOF)
    before = baseline.expected("physics", 1.5)[0]
    view = baseline.excluding(b.works)
    assert view.expected("physics", 1.5)[0] == 5
    assert baseline.expected("physics", 1.5)[0] == before


def test_a_work_is_compared_within_any_discipline_that_has_a_pool():
    """Not just its first-listed one."""
    pool = make_track("pool", citations=10, disciplines=("physics",))
    baseline = CitationBaseline.from_works(pool.works, asof=ASOF)
    work = make_track("w", years=((2025, 1),), citations=30, disciplines=("rare", "physics")).works[0]
    estimate = baseline.normalised_impact(work)
    assert "in physics" in estimate.provenance.detail


# -- foundationality from the citation graph ----------------------------------


def test_foundationality_uses_citations_between_directions_when_present():
    base = make_track("base")
    builders = [make_track(f"b{i}") for i in range(3)]
    for builder in builders:
        for work in builder.works:
            work.references = [base.works[0].id]
    tracks = [base] + builders
    baseline = CitationBaseline.from_works([w for t in tracks for w in t.works], asof=ASOF)
    scorer = Scorer(baseline, asof=ASOF, tracks=tracks)

    built_on = scorer.score(base).importance_parts["foundationality"]
    endpoint = scorer.score(builders[0]).importance_parts["foundationality"]
    assert built_on.grade is Grade.OBSERVED
    assert built_on.value > 0.5 > endpoint.value
    assert endpoint.value == 0.0


def test_foundationality_stays_a_flagged_proxy_without_reference_data():
    track = make_track("t", abstract="a scalable platform")
    baseline = CitationBaseline.from_works(track.works, asof=ASOF)
    part = Scorer(baseline, asof=ASOF, tracks=[track]).score(track).importance_parts["foundationality"]
    assert part.grade is Grade.ASSUMED and part.provenance.needs_review


# -- how sure the ranking is ---------------------------------------------------


def _ranked(tracks, **kwargs):
    baseline = CitationBaseline.from_works([w for t in tracks for w in t.works], asof=ASOF)
    return Scorer(baseline, asof=ASOF).rank(tracks, **kwargs)


def test_ranking_carries_no_intervals_unless_asked():
    assert _ranked([make_track("a"), make_track("b")])[0].priority_interval is None


def test_a_thin_track_gets_a_wider_priority_interval_than_a_thick_one():
    thin = make_track("thin", years=((2023, 1), (2024, 2), (2025, 2)), n_groups=5)
    thick = make_track("thick", years=((2023, 40), (2024, 60), (2025, 80)), n_groups=5)
    for index, work in enumerate(thin.works + thick.works):
        work.citations = 5 + (index * 37) % 100
    by_id = {a.track_id: a for a in _ranked([thin, thick], resamples=100)}

    def width(assessment):
        low, high = assessment.priority_interval
        assert low <= assessment.priority <= high
        return high - low

    assert width(by_id["thin"]) > width(by_id["thick"])


def test_tracks_the_evidence_cannot_separate_have_overlapping_rank_intervals():
    twins = [make_track("a"), make_track("b")]
    weak = make_track("weak", years=((2021, 9), (2022, 6), (2023, 3)), n_groups=1, citations=1)
    by_id = {a.track_id: a for a in _ranked(twins + [weak], resamples=100)}
    assert by_id["a"].rank_interval == by_id["b"].rank_interval == (1, 2)
    assert by_id["weak"].rank_interval == (3, 3)


def test_rank_intervals_are_reproducible():
    tracks = [make_track("a"), make_track("b", citations=30)]
    first = [a.priority_interval for a in _ranked(tracks, resamples=50, seed=4)]
    assert first == [a.priority_interval for a in _ranked(tracks, resamples=50, seed=4)]


def test_too_few_resamples_is_rejected():
    with pytest.raises(ValueError, match="noise"):
        _ranked([make_track("a")], resamples=5)
