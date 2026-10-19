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
