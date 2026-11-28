"""Markdown report rendering.

The simulated path is covered end to end by the CLI; these pin the branches a
normal run does not reach.
"""

import datetime as dt

from sciscout.models import Track
from sciscout.pipeline import Pipeline
from sciscout.report import _money, render_track
from sciscout.sources.local import LocalCorpus


def reports(simulate: bool, **track_changes):
    pipeline = Pipeline(asof=dt.date(2026, 9, 18), draws=500)
    works = LocalCorpus("data/demo_corpus.json").harvest("").works
    tracks = pipeline.tracks_from_works(works)[:2]
    for track in tracks:
        for name, value in track_changes.items():
            setattr(track, name, value)
    return pipeline.analyse(tracks, simulate=simulate)


def test_money_scales_to_a_readable_magnitude():
    assert _money(float("nan")) == "n/a"
    assert _money(1.5e9) == "$1.50B"
    assert _money(-2.0e6) == "-$2M"
    assert _money(1234) == "$1,234"


def test_sections_appear_in_order_with_a_simulation():
    text = render_track(reports(simulate=True)[0])
    headings = [
        "## ",
        "### Why it scores this way",
        "### Path to market",
        "### Investment case",
        "**Where the programme dies**",
        "**What drives the outcome**",
        "**Where the spread comes from:**",
    ]
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions)
    assert "| Project NPV |" in text and "| Investor MOIC |" in text


def test_detail_false_omits_only_the_sensitivity_table():
    report = reports(simulate=True)[0]
    brief = render_track(report, detail=False)
    assert "**What drives the outcome**" not in brief
    assert "**Where the programme dies**" in brief
    assert "**Where the spread comes from:**" in brief
    assert len(brief) < len(render_track(report))


def test_without_a_simulation_the_report_says_so_and_stops():
    text = render_track(reports(simulate=False)[0])
    assert "### Path to market" in text
    assert "### Investment case" not in text
    assert text.endswith("_No investment simulation was run for this direction._\n")


def test_a_commercialised_direction_has_no_stage_table():
    text = render_track(reports(simulate=True, trl_override=9)[0])
    assert "modelled as already commercialised" in text
    assert "| Stage | TRL |" not in text
    assert "_No investment simulation was run for this direction._" in text
