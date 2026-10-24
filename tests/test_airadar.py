"""AI app radar.

The tests that matter most pin the claims the radar's output rests on: a backtest
replays a past day without seeing the future, realised growth does not overlap
the window a forecast was made from, thin evidence cannot top the ranking, and
the tuner moves a parameter when the evidence supports it -- and holds still
when the evidence is noise.
"""

import datetime as dt
import json
import random

import pytest

from airadar.growth import extract_signals, percentile_ranks, score_apps
from airadar.inbox import InboxError, parse_batch
from airadar.iterate import realised_scores, source_health, spearman, tune
from airadar.models import App, AppKind, Method, Observation, canonical_id
from airadar.params import Params
from airadar.pipeline import run_day
from airadar.store import RunRecord, Store

D0 = dt.date(2026, 10, 1)


def obs(app_id, metric, value, day, method=Method.API, period=None, url="https://example.com/src"):
    return Observation(app_id, metric, float(value), day, url, method, "test", period)


def inbox_payload(**overrides):
    observation = {
        "app": {"name": "o/r", "url": "https://github.com/o/r", "kind": "repo", "tags": ["mcp"]},
        "metric": "stars_gained",
        "value": 100,
        "period_days": 7,
        "observed_on": "2026-10-19",
        "source_url": "https://github.com/trending?since=weekly",
        "method": "page_extract",
    }
    observation.update(overrides)
    return {"collected_at": "2026-10-19", "collector": "agent:test", "observations": [observation]}


# -- identity ----------------------------------------------------------------


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/Owner/Repo/tree/main", "gh:owner/repo"),
    ("github.com/owner/repo.git", "gh:owner/repo"),
    ("https://huggingface.co/spaces/A/B", "hf-space:a/b"),
    ("https://huggingface.co/org/model-7b", "hf-model:org/model-7b"),
    ("https://www.npmjs.com/package/@scope/pkg", "npm:@scope/pkg"),
    ("https://pypi.org/project/vllm/", "pypi:vllm"),
    ("https://www.cursor.com/pricing", "web:cursor.com"),
    ("https://apps.apple.com/app/x/id1", "web:apps.apple.com/app"),
    ("https://openai.com/codex", "web:openai.com/codex"),
    ("https://chatgpt.com", "web:chatgpt.com"),
])
def test_canonical_id_collapses_variants(url, expected):
    assert canonical_id(url) == expected


# -- inbox validation --------------------------------------------------------


def test_inbox_accepts_a_well_formed_batch():
    batch = parse_batch(inbox_payload())
    assert batch.apps[0].id == "gh:o/r"
    assert batch.observations[0].method is Method.PAGE_EXTRACT
    assert batch.observations[0].provenance().needs_review


@pytest.mark.parametrize("override,message", [
    ({"source_url": ""}, "source_url"),
    ({"metric": "vibes"}, "unknown metric"),
    ({"period_days": None}, "period_days"),
    ({"observed_on": "2026-10-20"}, "after collected_at"),
    ({"method": "guessed"}, "method"),
    ({"value": "lots"}, "number"),
    ({"confidence": 0.9}, "unknown field"),
])
def test_inbox_rejects_unfounded_or_malformed_numbers(override, message):
    with pytest.raises(InboxError, match=message):
        parse_batch(inbox_payload(**override))


def test_inbox_rejects_period_days_on_a_level_metric():
    with pytest.raises(InboxError, match="must not carry period_days"):
        parse_batch(inbox_payload(metric="stars", period_days=7))


# -- growth signals ----------------------------------------------------------


def test_delta_is_measured_between_two_snapshots():
    rows = [obs("a", "stars", 1000, D0), obs("a", "stars", 1700, D0 + dt.timedelta(days=7))]
    signals = extract_signals(rows, asof=D0 + dt.timedelta(days=7))
    (s,) = signals["a"]
    assert s.kind == "delta"
    assert s.abs_per_day == pytest.approx(100)
    assert s.rel_per_day == pytest.approx(0.0758, abs=1e-3)


def test_an_exact_count_beats_a_rounded_one_on_the_same_day():
    d1 = D0 + dt.timedelta(days=1)
    rounded = Observation("a", "stars", 58700.0, d1, "https://github.com/o/r",
                          Method.PAGE_EXTRACT, "test", rounded_to=100)
    exact = obs("a", "stars", 58670, d1, method=Method.PAGE_EXTRACT)
    rows = [obs("a", "stars", 58000, D0, method=Method.PAGE_EXTRACT), rounded, exact]
    (s,) = extract_signals(rows, asof=d1)["a"]
    assert s.evidence[1] is exact
    assert s.abs_per_day == pytest.approx(670)


def test_a_change_within_rounding_is_not_growth():
    d1 = D0 + dt.timedelta(days=1)
    rows = [
        obs("a", "stars", 58670, D0, method=Method.PAGE_EXTRACT),
        Observation("a", "stars", 58700.0, d1, "https://github.com/o/r",
                    Method.PAGE_EXTRACT, "test", rounded_to=100),
    ]
    assert extract_signals(rows, asof=d1) == {}
    # A change larger than the rounding still counts.
    rows[1] = Observation("a", "stars", 59900.0, d1, "https://github.com/o/r",
                          Method.PAGE_EXTRACT, "test", rounded_to=100)
    (s,) = extract_signals(rows, asof=d1)["a"]
    assert s.abs_per_day == pytest.approx(1230)


def test_inbox_carries_rounding_and_rejects_a_bad_one():
    batch = parse_batch(inbox_payload(metric="stars", period_days=None, value=58700, rounded_to=100))
    assert batch.observations[0].rounded_to == 100
    assert Observation.from_dict(batch.observations[0].to_dict()).rounded_to == 100
    for bad, message in ((0, "positive"), ("100", "number"), (True, "number")):
        with pytest.raises(InboxError, match=message):
            parse_batch(inbox_payload(metric="stars", period_days=None, value=58700, rounded_to=bad))


def test_a_single_level_is_a_baseline_not_growth():
    assert extract_signals([obs("a", "stars", 5000, D0)], asof=D0) == {}


def test_asof_hides_the_future():
    rows = [obs("a", "stars", 1000, D0), obs("a", "stars", 9000, D0 + dt.timedelta(days=3))]
    assert extract_signals(rows, asof=D0 + dt.timedelta(days=2)) == {}


def test_relative_growth_is_not_computed_below_the_minimum_base():
    rows = [obs("a", "stars", 5, D0), obs("a", "stars", 50, D0 + dt.timedelta(days=7))]
    (s,) = extract_signals(rows, asof=D0 + dt.timedelta(days=7))["a"]
    assert s.rel_per_day is None and s.abs_per_day == pytest.approx(45 / 7)


def test_period_gain_reads_against_level_and_flags_implausible_weeks():
    rows = [
        obs("a", "stars", 10_000, D0, Method.PAGE_EXTRACT),
        obs("a", "stars_gained", 8_000, D0, Method.PAGE_EXTRACT, period=7),
    ]
    (s,) = extract_signals(rows, asof=D0)["a"]
    assert s.kind == "period"
    assert s.rel_per_day == pytest.approx(0.2299, abs=1e-3)  # log(10000/2000) / 7
    assert s.anomaly and "80%" in s.anomaly


def test_a_month_of_growth_on_a_new_repo_is_not_flagged():
    rows = [
        obs("a", "stars", 10_000, D0, Method.PAGE_EXTRACT),
        obs("a", "stars_gained", 8_000, D0, Method.PAGE_EXTRACT, period=30),
    ]
    (s,) = extract_signals(rows, asof=D0)["a"]
    assert s.anomaly is None


def test_rate_change_captures_accelerating_downloads():
    rows = [
        obs("p", "downloads_period", 10_000, D0, period=7),
        obs("p", "downloads_period", 20_000, D0 + dt.timedelta(days=7), period=7),
    ]
    kinds = {s.kind for s in extract_signals(rows, asof=D0 + dt.timedelta(days=7))["p"]}
    assert "delta" in kinds


def test_daily_gains_a_day_apart_are_not_a_rate_change():
    rows = [
        obs("r", "stars_gained", 355, D0, method=Method.PAGE_EXTRACT, period=1),
        obs("r", "stars_gained", 363, D0 + dt.timedelta(days=1), method=Method.PAGE_EXTRACT, period=1),
    ]
    signals = extract_signals(rows, asof=D0 + dt.timedelta(days=1))
    assert not any(s.family == "stars_rate" for s in signals.get("r", []))
    # A week apart, the same daily metric does show acceleration.
    rows[1] = obs("r", "stars_gained", 710, D0 + dt.timedelta(days=7), method=Method.PAGE_EXTRACT, period=1)
    signals = extract_signals(rows, asof=D0 + dt.timedelta(days=7))
    assert any(s.family == "stars_rate" for s in signals["r"])


def test_realised_growth_excludes_windows_overlapping_the_forecast():
    # A weekly gain stated the day after the forecast is mostly the week before
    # it. Counting it would make momentum look more persistent than it is.
    rows = [obs("a", "stars_gained", 500, D0 + dt.timedelta(days=1), period=7)]
    assert realised_scores(rows, D0, horizon=7) == {}
    rows = [obs("a", "stars_gained", 500, D0 + dt.timedelta(days=7), period=7)]
    assert "a" in realised_scores(rows, D0, horizon=7)


# -- scoring -----------------------------------------------------------------


def test_percentile_ranks_share_ties():
    assert percentile_ranks([1, 2, 2, 3]) == [0.125, 0.5, 0.5, 0.875]


def test_thin_evidence_is_shrunk_toward_the_median():
    day = D0 + dt.timedelta(days=7)
    rows = []
    # 'strong' is measured twice by API; 'thin' has one page-read claim of the
    # same top growth. Both top their percentiles; only one should top the table.
    for app, method in (("strong", Method.API), ("thin", Method.PAGE_EXTRACT)):
        rows += [obs(app, "stars", 1000, D0, method), obs(app, "stars", 3000, day, method)]
    rows += [obs("strong", "downloads_period", 9e6, day, period=7)]
    for i in range(10):
        rows += [obs(f"slow{i}", "stars", 1000, D0), obs(f"slow{i}", "stars", 1010 + i, day)]
        rows += [obs(f"slow{i}", "downloads_period", 1000 + i, day, period=7)]
    scores = score_apps(extract_signals(rows, asof=day), Params.defaults())
    order = [s.app_id for s in scores]
    assert order.index("strong") < order.index("thin")


def test_anomalous_signals_are_discounted_not_dropped():
    rows = [
        obs("a", "stars", 10_000, D0, Method.PAGE_EXTRACT),
        obs("a", "stars_gained", 8_000, D0, Method.PAGE_EXTRACT, period=7),
    ]
    (score,) = score_apps(extract_signals(rows, asof=D0), Params.defaults())
    assert score.anomalies and score.evidence == pytest.approx(0.35)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) is None


# -- self-iteration ----------------------------------------------------------


def synthetic_world(days=24, apps=30, noise_reports=True, seed=7):
    """Apps with a persistent true growth rate, measured daily by API, plus
    (optionally) a daily 'reported growth' claim that is pure noise."""
    rng = random.Random(seed)
    rows = []
    for i in range(apps):
        level = rng.uniform(1_000, 50_000)
        rate = rng.uniform(0.002, 0.05)
        for d in range(days):
            day = D0 + dt.timedelta(days=d)
            level *= 1 + rate * rng.uniform(0.7, 1.3)
            rows.append(obs(f"app{i}", "stars", round(level), day))
            if noise_reports:
                rows.append(obs(f"app{i}", "growth_pct", rng.uniform(0, 300), day,
                                Method.REPORTED, period=1))
    return rows


def test_tuner_refuses_to_move_without_enough_matured_history():
    rows = synthetic_world(days=5)
    outcome = tune(Params.defaults(), rows, D0 + dt.timedelta(days=5))
    assert not outcome.adopted and "need 3" in outcome.reason


def test_tuner_learns_to_distrust_a_noisy_source():
    # Start from a bad setting: reported growth trusted as much as measurement.
    rows = synthetic_world()
    params = Params.defaults().with_value("trust.reported", 1.0).with_value("weight.reported_growth", 1.0)
    outcome = tune(params, rows, D0 + dt.timedelta(days=24))
    assert outcome.adopted, outcome.reason
    # Any of three steps cuts reliance on the noise: distrust the source, weigh
    # reported growth less, or weigh relative growth (the only thing a reported
    # rate supplies) less. The tuner picks whichever backtests best.
    assert outcome.knob in {"trust.reported", "weight.reported_growth", "relative_weight"}
    assert outcome.new < outcome.old
    assert outcome.candidate > outcome.baseline


def test_tuner_holds_still_when_nothing_beats_the_current_settings():
    # With no noisy source, every knob that matters already ranks persistently
    # growing apps correctly; no step clears the improvement bar.
    rows = synthetic_world(noise_reports=False)
    outcome = tune(Params.defaults(), rows, D0 + dt.timedelta(days=24))
    assert not outcome.adopted


def test_source_health_counts_consecutive_failures():
    runs = [
        RunRecord(D0, "api:github", 50),
        RunRecord(D0 + dt.timedelta(days=1), "api:github", 0, ["403"]),
        RunRecord(D0 + dt.timedelta(days=2), "api:github", 0, ["403"]),
        RunRecord(D0 + dt.timedelta(days=3), "api:github", 0, ["403"]),
    ]
    (health,) = source_health(runs)
    assert health.flagged and health.consecutive_failures == 3 and health.last_ok == D0


# -- end to end --------------------------------------------------------------


def test_daily_run_ingests_records_forecast_and_archives(tmp_path):
    store = Store(tmp_path)
    store.inbox.mkdir(parents=True)
    payload = inbox_payload()
    payload["observations"] += [
        dict(payload["observations"][0], app={"name": f"o/r{i}", "url": f"https://github.com/o/r{i}",
                                              "kind": "repo", "tags": ["mcp"]}, value=10 * i)
        for i in range(1, 6)
    ]
    (store.inbox / "2026-10-19-test.json").write_text(json.dumps(payload))
    (store.inbox / "2026-10-19-bad.json").write_text(json.dumps(inbox_payload(source_url="")))

    analysis = run_day(store, dt.date(2026, 10, 19))

    assert len(analysis.scores) == 6
    assert any("REJECTED" in line for line in analysis.ingest_log)
    assert (store.inbox / "archive" / "2026-10-19-test.json").exists()
    assert (store.inbox / "2026-10-19-bad.json").exists()  # left for a person to fix
    assert len(store.load_predictions()) == 1
    assert store.load_iterations()[0]["adopted"] is False
    assert any(h.collector == "inbox:2026-10-19-bad" and h.consecutive_failures for h in analysis.health)


# -- late and sparse reporting -----------------------------------------------


def test_a_late_reported_figure_is_invisible_before_it_was_collected():
    june, july, october = dt.date(2026, 6, 1), dt.date(2026, 7, 1), dt.date(2026, 10, 1)
    rows = [
        Observation("c", "revenue_annualized_usd", 3e9, june, "https://x/1", Method.REPORTED, "t",
                    collected_on=october),
        Observation("c", "revenue_annualized_usd", 4e9, july, "https://x/2", Method.REPORTED, "t",
                    collected_on=october),
    ]
    assert extract_signals(rows, asof=dt.date(2026, 7, 15)) == {}
    (s,) = extract_signals(rows, asof=october)["c"]
    assert s.kind == "delta" and s.start == june and s.end == july


def test_reported_figures_use_a_longer_window_and_decay_with_age():
    today = dt.date(2026, 10, 19)
    rows = [
        obs("c", "revenue_annualized_usd", 3e9, dt.date(2026, 6, 1), Method.REPORTED),
        obs("c", "revenue_annualized_usd", 4e9, dt.date(2026, 9, 1), Method.REPORTED),
        obs("r", "stars", 1000, today - dt.timedelta(days=7)),
        obs("r", "stars", 1200, today),
    ]
    signals = extract_signals(rows, asof=today)
    assert "c" in signals  # a 7-day window would have dropped it
    fresh = {s.app_id: s.evidence for s in score_apps(signals, Params.defaults(), asof=today)}
    stale = {s.app_id: s.evidence for s in score_apps(signals, Params.defaults())}
    assert fresh["c"] < stale["c"] and fresh["r"] == pytest.approx(stale["r"])


# -- shared page export ------------------------------------------------------


def _daily_store(tmp_path):
    store = Store(tmp_path)
    store.inbox.mkdir(parents=True)
    payload = inbox_payload()
    payload["observations"] += [
        dict(payload["observations"][0], app={"name": f"o/r{i}", "url": f"https://github.com/o/r{i}",
                                              "kind": "repo", "tags": ["mcp"]}, value=10 * i)
        for i in range(1, 6)
    ]
    (store.inbox / "2026-10-19-test.json").write_text(json.dumps(payload))
    run_day(store, dt.date(2026, 10, 19))
    return store


def _editorial(store, **overrides):
    body = {
        "takeaway": "MCP tooling is growing fastest.",
        "findings": [{"id": "mcp", "heading": "MCP leads", "text": "Six repos.", "source": "https://github.com/o/r"}],
        "caveats": ["Day one."],
    }
    body.update(overrides)
    store.reports.mkdir(parents=True, exist_ok=True)
    (store.reports / "2026-10-19-dashboard.json").write_text(json.dumps(body))


def test_dashboard_export_writes_every_dataset(tmp_path):
    import csv

    from airadar.dashboard import FILES, export

    store = _daily_store(tmp_path)
    _editorial(store)
    written = export(store, dt.date(2026, 10, 19), tmp_path / "out")
    assert set(written) == set(FILES)
    with written["apps"].open() as handle:
        apps = list(csv.DictReader(handle))
    assert len(apps) == 6 and apps[0]["rank"] == "1"
    with written["summary"].open() as handle:
        (summary,) = list(csv.DictReader(handle))
    assert summary["takeaway"] == "MCP tooling is growing fastest."
    assert summary["first_scorecard_expected"] == "2026-10-26"
    assert summary["tuning_status"] == "No change"


@pytest.mark.parametrize("override,message", [
    ({"takeaway": ""}, "takeaway"),
    ({"findings": []}, "at least one finding"),
    ({"findings": [{"id": "x", "heading": "h", "text": "t", "source": ""}]}, "source"),
    ({"findings": [{"id": "x", "heading": "h", "text": "t", "source": "github.com/o/r"}]}, "must be a URL"),
])
def test_dashboard_refuses_unsourced_findings(tmp_path, override, message):
    from airadar.dashboard import DashboardError, export

    store = _daily_store(tmp_path)
    _editorial(store, **override)
    with pytest.raises(DashboardError, match=message):
        export(store, dt.date(2026, 10, 19), tmp_path / "out")


def test_dashboard_needs_the_agents_editorial_file(tmp_path):
    from airadar.dashboard import DashboardError, export

    store = _daily_store(tmp_path)
    with pytest.raises(DashboardError, match="missing"):
        export(store, dt.date(2026, 10, 19), tmp_path / "out")


def test_a_gone_app_leaves_the_ranking_until_it_is_back(tmp_path):
    from airadar.pipeline import analyse
    from airadar.report import render

    store = _daily_store(tmp_path)
    day = dt.date(2026, 10, 19)
    assert "gh:o/r5" in {s.app_id for s in analyse(store, day).scores}

    store.record_availability("gh:o/r5", "gone", day, "https://github.com/o/r5", "HTTP 404")
    a = analyse(store, day)
    assert "gh:o/r5" not in {s.app_id for s in a.scores}
    assert "gh:o/r5" not in a.baseline_only
    assert "Removed from the ranking" in render(a) and "HTTP 404" in render(a)
    # The record of what was seen stays: no observation is touched.
    assert any(o.app_id == "gh:o/r5" for o in store.load_observations())
    # A gone record dated later is invisible to an earlier day.
    assert not store.gone_as_of(day - dt.timedelta(days=1))

    store.record_availability("gh:o/r5", "back", day + dt.timedelta(days=1), "https://github.com/o/r5")
    assert store.gone_as_of(day) and not store.gone_as_of(day + dt.timedelta(days=1))
    with pytest.raises(ValueError, match="source_url"):
        store.record_availability("gh:o/r5", "gone", day, " ")


def test_a_moved_app_keeps_one_history(tmp_path):
    from airadar.pipeline import analyse

    store = _daily_store(tmp_path)
    day = dt.date(2026, 10, 19)
    store.record_move("https://github.com/o/r5", "https://github.com/neworg/r5", day,
                      "https://github.com/neworg/r5", "transferred")
    # Old observations now belong to the new identity; the file is untouched.
    assert {o.app_id for o in store.load_observations()} >= {"gh:neworg/r5"}
    assert "gh:o/r5" not in {o.app_id for o in store.load_observations()}
    assert '"gh:o/r5"' in store.observations_path.read_text()
    apps = store.load_apps()
    assert "gh:o/r5" not in apps and apps["gh:neworg/r5"].url == "https://github.com/neworg/r5"
    # A sighting under either url lands on the one app.
    later = dt.date(2026, 10, 20)
    store.add([], [obs("gh:o/r5", "stars", 900, later, method=Method.PAGE_EXTRACT),
                   obs("gh:neworg/r5", "stars", 950, later, method=Method.PAGE_EXTRACT,
                       url="https://github.com/trending")])
    rows = [o for o in store.load_observations() if o.metric == "stars"]
    assert {o.app_id for o in rows} == {"gh:neworg/r5"}
    assert "gh:neworg/r5" in {s.app_id for s in analyse(store, later).scores}
    # Gone records follow the move.
    store.record_availability("gh:o/r5", "gone", later, "https://github.com/o/r5")
    assert "gh:neworg/r5" in store.gone_as_of(later)
    with pytest.raises(ValueError, match="same app"):
        store.record_move("https://github.com/A/B", "https://github.com/a/b", day, "https://x")
    with pytest.raises(ValueError, match="loop"):
        store.record_move("https://github.com/neworg/r5", "https://github.com/o/r5", day, "https://x")


def test_a_research_verdict_travels_with_its_hypothesis(tmp_path, monkeypatch):
    import airadar.pipeline as pipeline
    from airadar.dashboard import export
    from airadar.directions import Opportunity
    from airadar.report import render

    store = _daily_store(tmp_path)
    day = dt.date(2026, 10, 19)
    monkeypatch.setattr(pipeline, "opportunities", lambda directions, apps: [
        Opportunity("fast and uncrowded", "mcp", "few tracked apps"),
        Opportunity("open source ahead of product", "mcp", "no product tracked"),
    ])
    assert all(o.verdict is None for o in pipeline.analyse(store, day).opportunities)

    store.record_verdict("mcp", "*", "partly disproved", day - dt.timedelta(days=1),
                         "https://example.com/older", "older, applies to every pattern")
    store.record_verdict("mcp", "fast and uncrowded", "disproved", day,
                         "https://example.com/analysis", "incumbents exist")
    store.record_verdict("mcp", "*", "holds", day + dt.timedelta(days=1), "https://example.com/later")
    a = pipeline.analyse(store, day)
    fast, open_ = a.opportunities
    assert fast.verdict["verdict"] == "disproved"        # specific and newest as of day
    assert open_.verdict["verdict"] == "partly disproved"  # falls back to the "*" verdict
    text = render(a)
    assert "Researched 2026-10-19: disproved" in text and "https://example.com/analysis" in text
    assert "https://example.com/later" not in text        # a later verdict is invisible

    out = tmp_path / "page"
    _editorial(store)
    export(store, day, out)
    rows = (out / "hypotheses.csv").read_text().splitlines()
    assert rows[0].endswith("researched") and "disproved (2026-10-19): incumbents exist" in rows[1]

    with pytest.raises(ValueError, match="verdict must be"):
        store.record_verdict("mcp", "*", "maybe", day, "https://x")
    with pytest.raises(ValueError, match="source_url"):
        store.record_verdict("mcp", "*", "holds", day, "")
