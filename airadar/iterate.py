"""Self-iteration: the radar scores its own forecasts and tunes itself on the result.

A ranking of "fastest-growing" apps is implicitly a forecast -- that these apps
will keep growing fastest -- and a forecast can be checked. Each day's run:

1. **Records a prediction**: the day's scores, made with the day's parameters.
2. **Scores matured predictions**: once ``horizon_days`` have passed, compares
   the predicted ranking with the growth that actually followed (Spearman rank
   correlation, and precision of the top-k). That scorecard is in every report,
   so the radar's track record is never more than one page away.
3. **Proposes one parameter step** by backtest. Every knob is nudged up and down
   one step; each candidate is replayed over every past day that has a matured
   outcome, seeing only the data available on that day. The best candidate is
   adopted only if it beats the current parameters by a margin, across enough
   days, on most of them. Otherwise nothing changes, and the report says why.
4. **Checks its sources**: a collector failing day after day is flagged.

What this deliberately does *not* do is let the radar grade itself on its own
terms. The outcome a forecast is scored against is computed with the *default*
parameters, never the tuned ones -- a tuner that could move its own target
would always find a way to hit it.

Steps are bounded (one knob, one step, per day, inside each knob's range), so a
run of noisy evaluations cannot walk the scoring somewhere absurd overnight, and
every adopted change is written to ``iterations.jsonl`` with its evidence.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .growth import extract_signals, score_apps
from .models import Observation
from .params import FIXED, KNOBS, Params
from .store import RunRecord, Store

MIN_EVAL_DAYS = 3
MIN_APPS_PER_EVAL = 15
MIN_IMPROVEMENT = 0.02
MIN_WIN_SHARE = 2 / 3
BACKTEST_DAYS = 30
# Like-for-like scorecard: Spearman over only the forecast's own top N of the
# apps compared. The full figure rises with pool size alone -- a pool padded
# with near-zero-growth search hits is easy to rank -- so it cannot be compared
# across days whose pools differ tenfold. Reported beside it, never tuned on.
LIKE_FOR_LIKE_N = 100


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation with mid-ranks for ties; None below 3 points."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


@dataclass
class Evaluation:
    made_on: dt.date
    horizon_days: int
    apps: int
    spearman: float | None
    precision_at_k: float | None
    k: int
    spearman_top: float | None = None
    top_n: int = 0

    def to_dict(self) -> dict:
        return {
            "made_on": self.made_on.isoformat(),
            "horizon_days": self.horizon_days,
            "apps": self.apps,
            "spearman": self.spearman,
            "precision_at_k": self.precision_at_k,
            "k": self.k,
            "spearman_top": self.spearman_top,
            "top_n": self.top_n,
        }

    @property
    def lift(self) -> float | None:
        """Top-k hit rate over what random picks would get (k / apps): the one
        leader-picking figure comparable across pools of 50 and 2,500 apps."""
        if self.precision_at_k is None or not self.apps or not self.k:
            return None
        return self.precision_at_k / (self.k / self.apps)

    @classmethod
    def from_dict(cls, raw: dict) -> "Evaluation":
        return cls(dt.date.fromisoformat(raw["made_on"]), int(raw["horizon_days"]), int(raw["apps"]),
                   raw["spearman"], raw["precision_at_k"], int(raw["k"]),
                   raw.get("spearman_top"), int(raw.get("top_n", 0)))


def realised_scores(observations: list[Observation], made_on: dt.date, horizon: int) -> dict[str, float]:
    """Growth that actually happened in ``(made_on, made_on + horizon]``,
    scored with the default parameters -- the fixed yardstick."""
    signals = extract_signals(
        observations, asof=made_on + dt.timedelta(days=horizon), after=made_on
    )
    return {s.app_id: s.score for s in score_apps(signals, Params.defaults())}


def evaluate(predicted: dict[str, float], realised: dict[str, float], made_on: dt.date, horizon: int) -> Evaluation:
    common = sorted(set(predicted) & set(realised))
    k = max(1, min(FIXED["top_k"], len(common) // 3))
    rho = spearman([predicted[a] for a in common], [realised[a] for a in common])
    precision = None
    if len(common) >= 3:
        top_pred = set(sorted(common, key=lambda a: -predicted[a])[:k])
        top_real = set(sorted(common, key=lambda a: -realised[a])[:k])
        precision = len(top_pred & top_real) / k
    top = sorted(common, key=lambda a: (-predicted[a], a))[:LIKE_FOR_LIKE_N]
    rho_top = spearman([predicted[a] for a in top], [realised[a] for a in top])
    return Evaluation(made_on, horizon, len(common), rho, precision, k, rho_top, len(top))


def record_prediction(store: Store, made_on: dt.date, scores, params: Params) -> None:
    already = {p["made_on"] for p in store.load_predictions()}
    if made_on.isoformat() in already:
        return
    store.append_predictions([{
        "made_on": made_on.isoformat(),
        "horizon_days": FIXED["horizon_days"],
        "params": params.values,
        "scores": {s.app_id: round(s.score, 6) for s in scores},
    }])


def scorecard(store: Store, today: dt.date, observations: list[Observation]) -> list[Evaluation]:
    """Evaluate every recorded prediction whose horizon has passed.

    A score already frozen (``scorecard.jsonl``, written by the daily run the
    first day it scores a forecast) is reported as frozen: a published row
    does not drift when later records -- a repo move, a late reading --
    would change its recomputation.

    A prediction names apps by the ids they had on the day it was made. An app
    that has moved since (a repo transfer) is looked up under its current id,
    or its realised growth -- recorded under the new id -- would never be
    compared and the app would silently drop out of the scorecard.
    """
    aliases = store.aliases()
    frozen = store.frozen_scores()
    out = []
    for row in store.load_predictions():
        made_on = dt.date.fromisoformat(row["made_on"])
        horizon = int(row["horizon_days"])
        if made_on + dt.timedelta(days=horizon) > today:
            continue
        if row["made_on"] in frozen:
            out.append(Evaluation.from_dict(frozen[row["made_on"]]))
            continue
        predicted: dict[str, float] = {}
        for app_id, score in row["scores"].items():
            predicted.setdefault(aliases.get(app_id, (app_id,))[0], score)
        realised = realised_scores(observations, made_on, horizon)
        out.append(evaluate(predicted, realised, made_on, horizon))
    return out


EVIDENCE_CLASSES = ("measured", "stated by API", "stated on a page", "company-reported")


def evidence_class(signals) -> str:
    """The strongest kind of evidence behind an app's score, in the order the
    radar trusts it: its own measured change, then a gain an API stated (a
    newly found search hit's stars over its age), a gain a page stated, then
    a company's own figure."""
    kinds = {(s.kind, s.method.value) for s in signals}
    if any(kind == "delta" for kind, _ in kinds):
        return "measured"
    if any(kind == "period" and method == "api" for kind, method in kinds):
        return "stated by API"
    if any(kind == "period" and method == "page_extract" for kind, method in kinds):
        return "stated on a page"
    return "company-reported"


@dataclass
class ClassSplit:
    made_on: dt.date
    evidence: str
    apps: int
    spearman: float | None


def like_for_like_by_evidence(store: Store, today: dt.date,
                              observations: list[Observation]) -> list[ClassSplit]:
    """Split each matured forecast's own top ``LIKE_FOR_LIKE_N`` apps by the
    kind of evidence behind their score on the day it was made, and give the
    rank correlation within each kind. Diagnostic only: it shows which kind
    of evidence a weak like-for-like figure comes from. Nothing tunes on it.
    """
    aliases = store.aliases()

    def current(app_id: str) -> str:
        return aliases.get(app_id, (app_id,))[0]

    out: list[ClassSplit] = []
    for row in store.load_predictions():
        made_on = dt.date.fromisoformat(row["made_on"])
        horizon = int(row["horizon_days"])
        if made_on + dt.timedelta(days=horizon) > today:
            continue
        params = Params.defaults()
        params.values.update(row.get("params", {}))
        classes: dict[str, str] = {}
        for app in score_apps(extract_signals(observations, asof=made_on), params, asof=made_on):
            classes.setdefault(current(app.app_id), evidence_class(app.signals))
        predicted: dict[str, float] = {}
        for app_id, score in row["scores"].items():
            predicted.setdefault(current(app_id), score)
        realised = realised_scores(observations, made_on, horizon)
        common = [a for a in predicted if a in realised]
        top = sorted(common, key=lambda a: (-predicted[a], a))[:LIKE_FOR_LIKE_N]
        for kind in EVIDENCE_CLASSES:
            group = [a for a in top if classes.get(a) == kind]
            if not group:
                continue
            rho = spearman([predicted[a] for a in group], [realised[a] for a in group])
            out.append(ClassSplit(made_on, kind, len(group), rho))
    return out


def _evaluable_days(observations: list[Observation], today: dt.date, horizon: int) -> list[dt.date]:
    days = sorted({o.known_on for o in observations if o.known_on + dt.timedelta(days=horizon) <= today})
    return days[-BACKTEST_DAYS:]


def backtest(params: Params, observations: list[Observation], days: list[dt.date], horizon: int,
             realised_cache: dict | None = None) -> list[Evaluation]:
    cache = realised_cache if realised_cache is not None else {}
    out = []
    for day in days:
        if day not in cache:
            cache[day] = realised_scores(observations, day, horizon)
        predicted = {s.app_id: s.score for s in score_apps(extract_signals(observations, asof=day), params, asof=day)}
        evaluation = evaluate(predicted, cache[day], day, horizon)
        if evaluation.spearman_top is not None and evaluation.apps >= MIN_APPS_PER_EVAL:
            out.append(evaluation)
    return out


def span_backtest(store: Store, today: dt.date, observations: list[Observation],
                  min_spans: tuple[int, ...] = (0, 3, 7)) -> list[dict]:
    """Re-score every matured forecast with measured deltas restricted to a
    minimum span, and report the full and like-for-like rank correlations.

    Diagnostic only (2026-11-18): it tests whether short-span deltas are what
    keep the like-for-like score at 0.40-0.53. Each forecast is recomputed
    with its own stored params; the yardstick (``realised_scores``) is the
    fixed one. Nothing is tuned or adopted from it.
    """
    out = []
    for row in store.load_predictions():
        made_on = dt.date.fromisoformat(row["made_on"])
        horizon = int(row["horizon_days"])
        if made_on + dt.timedelta(days=horizon) > today:
            continue
        params = Params.defaults()
        params.values.update(row.get("params", {}))
        realised = realised_scores(observations, made_on, horizon)
        result = {"made_on": made_on}
        for n in min_spans:
            signals = extract_signals(observations, asof=made_on, min_delta_span_days=n)
            predicted = {s.app_id: s.score for s in score_apps(signals, params, asof=made_on)}
            e = evaluate(predicted, realised, made_on, horizon)
            result[n] = {"apps": e.apps, "spearman": e.spearman, "spearman_top": e.spearman_top,
                         "precision_at_k": e.precision_at_k}
        out.append(result)
    return out


BASELINE_FLOOR = -1e18  # rank for apps with no measured change: below every real gain


def latest_measured_gain(signals: dict[str, list], rate: str = "abs") -> dict[str, float]:
    """Each app's most recent measured change (a ``delta`` signal on a level,
    not a rate) as of the signals' date: stars or likes per day (``abs``) or
    relative growth per day (``rel``). The naive "keep doing what you did"
    forecast: no weights, no shrinkage, no other evidence."""
    out: dict[str, float] = {}
    for app_id, sigs in signals.items():
        deltas = [s for s in sigs if s.kind == "delta" and not s.family.endswith("_rate")
                  and (s.abs_per_day if rate == "abs" else s.rel_per_day) is not None]
        if not deltas:
            continue
        latest = max(deltas, key=lambda s: (s.end, s.days))
        out[app_id] = latest.abs_per_day if rate == "abs" else latest.rel_per_day
    return out


def persistence_backtest(store: Store, today: dt.date, observations: list[Observation]) -> list[dict]:
    """Score every matured forecast beside a naive persistence baseline.

    Diagnostic only (2026-11-19). Lift and full Spearman both rise with pool
    size, so neither says whether the scoring model adds anything over the
    raw measurements. The baseline ranks the *same apps* by their latest
    measured gain alone (apps with none rank last) and is scored exactly
    like the forecast, against the same fixed yardstick. If the forecast does
    not beat it, the model adds nothing over yesterday's gainers. Nothing is
    tuned or adopted from it.
    """
    aliases = store.aliases()

    def current(app_id: str) -> str:
        return aliases.get(app_id, (app_id,))[0]

    out = []
    for row in store.load_predictions():
        made_on = dt.date.fromisoformat(row["made_on"])
        horizon = int(row["horizon_days"])
        if made_on + dt.timedelta(days=horizon) > today:
            continue
        predicted: dict[str, float] = {}
        for app_id, score in row["scores"].items():
            predicted.setdefault(current(app_id), score)
        realised = realised_scores(observations, made_on, horizon)
        signals = extract_signals(observations, asof=made_on)
        result = {"made_on": made_on, "forecast": evaluate(predicted, realised, made_on, horizon)}
        measured: set[str] = set()
        for rate in ("abs", "rel"):
            gains: dict[str, float] = {}
            for app_id, gain in latest_measured_gain(signals, rate).items():
                gains.setdefault(current(app_id), gain)
            naive = {a: gains.get(a, BASELINE_FLOOR) for a in predicted}
            result[rate] = evaluate(naive, realised, made_on, horizon)
            result[rate + "_measured"] = evaluate({a: naive[a] for a in predicted if a in gains},
                                                  realised, made_on, horizon)
            measured = {a for a in predicted if a in gains}
        result["forecast_measured"] = evaluate({a: predicted[a] for a in measured}, realised, made_on, horizon)
        result["with_gain"] = sum(1 for a in measured if a in realised)
        out.append(result)
    return out


@dataclass
class TuningOutcome:
    adopted: bool
    reason: str
    knob: str | None = None
    old: float | None = None
    new: float | None = None
    baseline: float | None = None
    candidate: float | None = None
    days: int = 0
    candidates_tried: list[dict] = field(default_factory=list)


def _mean(evals: list[Evaluation]) -> float:
    """The tuner's objective: mean like-for-like rank correlation, each
    forecast's own top ``LIKE_FOR_LIKE_N`` apps against the fixed yardstick.

    Chosen by the user on 2026-11-19 ("switch the tuner to the top-100
    score"), replacing full-pool Spearman, which rose with pool size and
    could reward a setting that only sorted easy, stale apps better."""
    return sum(e.spearman_top for e in evals) / len(evals)


def _mean_precision(evals: list[Evaluation]) -> float:
    """Mean top-k hit rate: the guard beside the objective. A top-100 figure
    cannot penalise a forecast for missing a fast grower ranked below 100."""
    values = [e.precision_at_k for e in evals if e.precision_at_k is not None]
    return sum(values) / len(values) if values else 0.0


def tune(params: Params, observations: list[Observation], today: dt.date) -> TuningOutcome:
    """Try one step on every knob; adopt the best if the evidence clears the bar."""
    horizon = FIXED["horizon_days"]
    days = _evaluable_days(observations, today, horizon)
    cache: dict = {}
    base = backtest(params, observations, days, horizon, cache)
    if len(base) < MIN_EVAL_DAYS:
        return TuningOutcome(
            False,
            f"only {len(base)} past day(s) have a matured {horizon}-day outcome with at least "
            f"{MIN_APPS_PER_EVAL} comparable apps; need {MIN_EVAL_DAYS} before any parameter may move",
            days=len(base),
        )
    base_by_day = {e.made_on: e.spearman_top for e in base}
    base_eval = {e.made_on: e for e in base}
    baseline = _mean(base)

    best = None
    tried = []
    for name, knob in KNOBS.items():
        for direction in (-1, 1):
            value = params[name] + direction * knob.step
            if not knob.low <= value <= knob.high:
                continue
            candidate = params.with_value(name, value)
            evals = [e for e in backtest(candidate, observations, days, horizon, cache) if e.made_on in base_by_day]
            if len(evals) < MIN_EVAL_DAYS:
                continue
            mean = _mean(evals)
            wins = sum(1 for e in evals if e.spearman_top > base_by_day[e.made_on]) / len(evals)
            hits = _mean_precision(evals)
            base_hits = _mean_precision([base_eval[e.made_on] for e in evals])
            tried.append({"knob": name, "value": candidate[name], "spearman_top": round(mean, 4),
                          "win_share": round(wins, 3), "precision": round(hits, 4),
                          "base_precision": round(base_hits, 4)})
            if best is None or mean > best[0]:
                best = (mean, wins, name, candidate[name], len(evals), hits, base_hits)

    if best is None:
        return TuningOutcome(False, "no candidate could be evaluated", baseline=baseline, days=len(base))
    mean, wins, name, value, n, hits, base_hits = best
    outcome = TuningOutcome(
        False, "", name, params[name], value, baseline, mean, n, tried,
    )
    if mean - baseline < MIN_IMPROVEMENT:
        outcome.reason = (
            f"best candidate ({name} -> {value:g}) improves top-100 rank correlation by only "
            f"{mean - baseline:+.3f}; the bar is {MIN_IMPROVEMENT:+.2f}"
        )
    elif wins < MIN_WIN_SHARE:
        outcome.reason = (
            f"best candidate ({name} -> {value:g}) wins on only {wins:.0%} of {n} days; "
            f"needs {MIN_WIN_SHARE:.0%} so one lucky day cannot carry it"
        )
    elif hits < base_hits:
        outcome.reason = (
            f"best candidate ({name} -> {value:g}) raises top-100 rank correlation "
            f"{baseline:.3f} -> {mean:.3f} but lowers the mean top-{FIXED['top_k']} hit rate "
            f"{base_hits:.1%} -> {hits:.1%}; it may not trade leaders for ordering"
        )
    else:
        outcome.adopted = True
        outcome.reason = (
            f"{name} {params[name]:g} -> {value:g}: mean top-100 Spearman {baseline:.3f} -> {mean:.3f} "
            f"over {n} days, better on {wins:.0%} of them; top-{FIXED['top_k']} hit rate "
            f"{base_hits:.1%} -> {hits:.1%}"
        )
    return outcome


def apply_tuning(store: Store, params: Params, outcome: TuningOutcome, today: dt.date) -> Params:
    row = {
        "date": today.isoformat(),
        "adopted": outcome.adopted,
        "reason": outcome.reason,
        "knob": outcome.knob,
        "old": outcome.old,
        "new": outcome.new,
        "objective": "spearman_top",
        "baseline_spearman": outcome.baseline,
        "candidate_spearman": outcome.candidate,
        "days": outcome.days,
        "candidates": outcome.candidates_tried,
    }
    store.append_iteration(row)
    if not outcome.adopted:
        return params
    updated = params.with_value(outcome.knob, outcome.new)
    updated.provenance[outcome.knob] = (
        f"observed: tuned by backtest on {today.isoformat()} ({outcome.reason})"
    )
    updated.history.append({k: row[k] for k in ("date", "knob", "old", "new", "reason")})
    updated.save(store.params_path)
    return updated


@dataclass
class SourceHealth:
    collector: str
    last_ok: dt.date | None
    consecutive_failures: int
    last_count: int
    last_errors: list[str]

    @property
    def flagged(self) -> bool:
        return self.consecutive_failures >= 3


def source_health(runs: list[RunRecord]) -> list[SourceHealth]:
    by_collector: dict[str, list[RunRecord]] = {}
    for run in sorted(runs, key=lambda r: r.date):
        by_collector.setdefault(run.collector, []).append(run)
    out = []
    for collector, rows in sorted(by_collector.items()):
        failures = 0
        for run in reversed(rows):
            if run.ok:
                break
            failures += 1
        last_ok = max((r.date for r in rows if r.ok), default=None)
        out.append(SourceHealth(collector, last_ok, failures, rows[-1].observations, rows[-1].errors))
    return out
