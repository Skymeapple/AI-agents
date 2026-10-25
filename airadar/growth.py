"""From observations to growth signals, and from signals to a ranking.

Growth is never read off a single number. It comes from one of three kinds of
evidence, each turned into a :class:`Signal` with an absolute velocity (units
per day) and, where it means something, a relative one (log-growth per day):

``delta``
    The same level metric seen on two of our own snapshots. The best evidence,
    because the radar measured both ends itself -- and the reason the radar has
    to run every day: a level seen once is a baseline, not growth.
``period``
    A source stating a gain over a period ("+4,774 stars this week"). Read
    against the same family's level, it gives relative growth too.
``reported_growth``
    A stated growth rate ("ARR tripled in six months"). Relative only, and
    discounted: a company choosing what to disclose chooses its baseline.

Comparing a repository's stars to a product's revenue directly is meaningless,
so signals are compared by **percentile**: relative growth across all signals
(it is scale-free), absolute velocity only within its own metric family. An
app's score is the evidence-weighted mean of its signal percentiles, shrunk
toward the median by a pseudo-count, so one lucky signal cannot top the table.

Every function takes an ``asof`` date and ignores anything observed after it.
That is what makes backtesting honest: replaying a past day sees only what was
known on that day.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field

from .models import METRICS, Method, MetricType, Observation
from .params import FIXED, Params

METHOD_RANK = {Method.API: 3, Method.PAGE_EXTRACT: 2, Method.REPORTED: 1}
MAX_DELTA_BASELINE_DAYS = 28


@dataclass
class Signal:
    app_id: str
    kind: str  # delta | period | reported_growth
    family: str
    method: Method
    start: dt.date
    end: dt.date
    abs_per_day: float | None
    rel_per_day: float | None
    evidence: list[Observation]
    anomaly: str | None = None

    @property
    def days(self) -> float:
        return max(1.0, (self.end - self.start).days)

    def describe(self) -> str:
        unit = self.family.replace("_", " ")
        parts = []
        if self.abs_per_day is not None:
            parts.append(f"{_fmt(self.abs_per_day * self.days)} {unit} over {self.days:g}d")
        if self.rel_per_day is not None:
            weekly = math.expm1(self.rel_per_day * 7) * 100
            parts.append(f"{weekly:+.1f}%/wk")
        if self.kind == "delta" and self.method is Method.API:
            how = "measured"
        elif self.kind == "delta":
            how = f"between two {self.method.value.replace('_', ' ')} figures"
        else:
            how = {"period": "stated", "reported_growth": "reported rate"}[self.kind]
            how = f"{how}, {self.method.value.replace('_', ' ')}"
        return f"{', '.join(parts)} ({how})"


def _fmt(x: float) -> str:
    sign = "-" if x < 0 else "+"
    x = abs(x)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{sign}{x / div:.1f}{suffix}"
    return f"{sign}{x:.0f}"


def _best_per_day(observations: list[Observation]) -> dict[dt.date, Observation]:
    """One value per day: the most trusted method wins, then the most precise
    reading, then the larger value.

    Two pages read minutes apart differ by a handful of stars; which one wins
    does not matter, but the choice has to be deterministic. An exact count has
    to beat a rounded one, though: "58.7k" rounds 58,670 up, and letting it win
    as the larger value would invent 30 stars.
    """
    def rank(o: Observation) -> tuple:
        return (METHOD_RANK[o.method], -(o.rounded_to or 0.0), o.value)

    best: dict[dt.date, Observation] = {}
    for o in observations:
        current = best.get(o.observed_on)
        if current is None or rank(o) > rank(current):
            best[o.observed_on] = o
    return best


def extract_signals(
    observations: list[Observation],
    asof: dt.date,
    after: dt.date | None = None,
    lookback_days: int = FIXED["lookback_days"],
) -> dict[str, list[Signal]]:
    """Growth signals per app, using only what was known on ``asof``.

    With ``after`` set, only growth that happened after that date counts: this
    is the *realised* growth a prediction made on ``after`` is scored against.
    The baseline of a delta may sit on or before ``after`` -- it has to, to
    measure growth across it -- but its endpoint must fall after.
    """
    def window(method: Method) -> tuple[dt.date, int, int]:
        """(endpoint must fall after, baseline target offset, max baseline span)."""
        if method is Method.REPORTED:
            back, span = FIXED["reported_lookback_days"], FIXED["reported_max_span_days"]
        else:
            back, span = lookback_days, MAX_DELTA_BASELINE_DAYS
        start = after if after is not None else asof - dt.timedelta(days=back)
        return start, back, span

    # Known by asof means *collected* by asof: a late-reported figure about an
    # earlier date was not available on that earlier date.
    known = [o for o in observations if o.known_on <= asof]

    levels: dict[tuple[str, str], dict[dt.date, Observation]] = {}
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for o in known:
        if o.spec.type is MetricType.LEVEL:
            grouped[(o.app_id, o.metric)].append(o)
    for key, rows in grouped.items():
        levels[key] = _best_per_day(rows)

    family_levels: dict[tuple[str, str], dict[dt.date, Observation]] = defaultdict(dict)
    for (app_id, metric), by_day in levels.items():
        family_levels[(app_id, METRICS[metric].family)].update(by_day)

    signals: dict[str, list[Signal]] = defaultdict(list)

    # delta: our own two snapshots of one level metric
    for (app_id, metric), by_day in levels.items():
        days = sorted(by_day)
        endpoints = [d for d in days if d > window(by_day[d].method)[0]]
        if not endpoints:
            continue
        end = endpoints[-1]
        _, back, span_limit = window(by_day[end].method)
        target = end - dt.timedelta(days=back)
        baselines = [
            d for d in days
            if d < end and (end - d).days <= span_limit
            and (after is None or d <= after)
        ]
        if not baselines:
            continue
        start = min(baselines, key=lambda d: (abs((d - target).days), -d.toordinal()))
        v0, v1 = by_day[start].value, by_day[end].value
        # A change no larger than the two readings' rounding is not resolved:
        # 58.7k one day and 58.7k the next says nothing about growth.
        slack = ((by_day[start].rounded_to or 0.0) + (by_day[end].rounded_to or 0.0)) / 2
        if slack and abs(v1 - v0) <= slack:
            continue
        span = (end - start).days
        family = METRICS[metric].family
        rel = None
        if v0 >= max(1.0, FIXED["min_base"][family]) and v1 > 0:
            rel = math.log(v1 / v0) / span
        weakest = min((by_day[start], by_day[end]), key=lambda o: METHOD_RANK[o.method])
        signals[app_id].append(Signal(
            app_id, "delta", family, weakest.method, start, end,
            (v1 - v0) / span, rel, [by_day[start], by_day[end]],
        ))

    # rate change: the same period metric (weekly downloads) measured on two of
    # our own runs, at least one period apart. Downloads have no meaningful
    # cumulative level, so this is how package adoption shows acceleration.
    rates: dict[tuple[str, str, float], list[Observation]] = defaultdict(list)
    for o in known:
        if o.spec.type is MetricType.PERIOD and o.metric != "growth_pct":
            rates[(o.app_id, o.metric, o.period_days)].append(o)
    for (app_id, metric, period), rows in rates.items():
        by_day = _best_per_day(rows)
        days = sorted(by_day)
        endpoints = [d for d in days if d > window(by_day[d].method)[0]]
        if not endpoints:
            continue
        end = endpoints[-1]
        span_limit = window(by_day[end].method)[2]
        baselines = [
            d for d in days
            if (end - d).days >= max(period, FIXED["min_rate_change_span_days"])
            and (end - d).days <= span_limit
            and (after is None or d <= after)
        ]
        if not baselines:
            continue
        start = baselines[-1]
        v0, v1 = by_day[start].value, by_day[end].value
        family = METRICS[metric].family
        if v0 < max(1.0, FIXED["min_base"][family]) or v1 <= 0:
            continue
        weakest = min((by_day[start], by_day[end]), key=lambda o: METHOD_RANK[o.method])
        signals[app_id].append(Signal(
            app_id, "delta", family + "_rate", weakest.method, start, end,
            None, math.log(v1 / v0) / (end - start).days, [by_day[start], by_day[end]],
        ))

    # period and reported_growth: a source stating growth over a period
    for o in known:
        if o.spec.type is not MetricType.PERIOD or o.observed_on <= window(o.method)[0]:
            continue
        start = o.observed_on - dt.timedelta(days=o.period_days)
        # Realised growth must be growth that happened after the prediction.
        # A weekly gain stated the day after mostly re-measures the week the
        # prediction was made from, and would flatter persistence.
        if after is not None and start < after:
            continue
        if o.metric == "growth_pct":
            if o.value <= -100:
                continue
            signals[o.app_id].append(Signal(
                o.app_id, "reported_growth", o.spec.family, o.method, start, o.observed_on,
                None, math.log1p(o.value / 100) / o.period_days, [o],
            ))
            continue
        level_obs = _nearest_level(family_levels.get((o.app_id, o.spec.family), {}), o.observed_on)
        rel, anomaly, evidence = None, None, [o]
        if level_obs is not None:
            evidence.append(level_obs)
            level = level_obs.value
            base = level - o.value
            if (level > 0 and o.value / level > FIXED["anomaly_share"]
                    and o.period_days <= FIXED["anomaly_max_period_days"]):
                anomaly = (
                    f"gained {o.value:,.0f} of {level:,.0f} total in {o.period_days:g}d "
                    f"({o.value / level:.0%}): extraction error or inorganic growth?"
                )
            if base >= max(1.0, FIXED["min_base"][o.spec.family]):
                rel = math.log(level / base) / o.period_days
        signals[o.app_id].append(Signal(
            o.app_id, "period", o.spec.family, o.method, start, o.observed_on,
            o.value / o.period_days, rel, evidence, anomaly,
        ))
    return dict(signals)


@dataclass
class Mover:
    """A level metric read on two of the radar's own runs: the newest reading
    on ``asof`` and the one before it."""
    app_id: str
    metric: str
    start: dt.date
    end: dt.date
    before: float
    after: float
    rounded: bool  # either reading rounded by its page

    @property
    def change(self) -> float:
        return self.after - self.before

    @property
    def per_day(self) -> float:
        return self.change / max(1, (self.end - self.start).days)


def measured_movers(observations: list[Observation], asof: dt.date, metric: str = "stars",
                    max_gap_days: int = 3) -> list[Mover]:
    """Apps read on ``asof`` and on an earlier run at most ``max_gap_days``
    back, largest change per day first. This is the growth the radar measured
    itself, as opposed to growth a page stated. A change within the readings'
    rounding is left out, as it is for delta signals."""
    rows: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        if o.metric == metric and o.known_on <= asof:
            rows[o.app_id].append(o)
    out = []
    for app_id, obs in rows.items():
        by_day = _best_per_day(obs)
        if asof not in by_day:
            continue
        earlier = [d for d in by_day if d < asof and (asof - d).days <= max_gap_days]
        if not earlier:
            continue
        o0, o1 = by_day[max(earlier)], by_day[asof]
        slack = ((o0.rounded_to or 0.0) + (o1.rounded_to or 0.0)) / 2
        if slack and abs(o1.value - o0.value) <= slack:
            continue
        out.append(Mover(app_id, metric, o0.observed_on, asof, o0.value, o1.value,
                         bool(o0.rounded_to or o1.rounded_to)))
    out.sort(key=lambda m: (-m.per_day, m.app_id))
    return out


def _nearest_level(by_day: dict[dt.date, Observation], when: dt.date, tolerance: int = 3):
    candidates = [d for d in by_day if abs((d - when).days) <= tolerance]
    if not candidates:
        return None
    return by_day[min(candidates, key=lambda d: (abs((d - when).days), -d.toordinal()))]


def percentile_ranks(values: list[float]) -> list[float]:
    """Mid-rank percentiles in ``(0, 1)``; ties share a rank."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(values)
    out = []
    for v in values:
        below = _bisect_left(order, v)
        equal = _bisect_right(order, v) - below
        out.append((below + 0.5 * equal) / n)
    return out


def _bisect_left(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_right(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass
class AppScore:
    app_id: str
    score: float
    evidence: float
    signals: list[Signal] = field(default_factory=list)
    signal_scores: list[float] = field(default_factory=list)

    @property
    def anomalies(self) -> list[str]:
        return [s.anomaly for s in self.signals if s.anomaly]

    def headline(self) -> str:
        """The strongest signal, in words."""
        if not self.signals:
            return "-"
        best = max(zip(self.signal_scores, range(len(self.signals))))[1]
        return self.signals[best].describe()


def score_apps(signals: dict[str, list[Signal]], params: Params,
               asof: dt.date | None = None) -> list[AppScore]:
    """Rank apps by growth. See the module docstring for the method.

    With ``asof`` set, a signal's weight halves for every
    ``staleness_half_life_days`` between its end and ``asof``: September's
    revenue figure is weaker evidence of today's growth than this week's stars.
    """
    flat = [s for rows in signals.values() for s in rows]
    if not flat:
        return []
    half_life = FIXED["staleness_half_life_days"]

    def freshness(s: Signal) -> float:
        if asof is None:
            return 1.0
        return 0.5 ** (max(0, (asof - s.end).days) / half_life)

    rel_index = [i for i, s in enumerate(flat) if s.rel_per_day is not None]
    rel_pct = dict(zip(rel_index, percentile_ranks([flat[i].rel_per_day for i in rel_index])))
    abs_pct: dict[int, float] = {}
    by_family: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(flat):
        if s.abs_per_day is not None:
            by_family[s.family].append(i)
    for members in by_family.values():
        abs_pct.update(zip(members, percentile_ranks([flat[i].abs_per_day for i in members])))

    w_rel = params["relative_weight"]
    per_signal: dict[int, float] = {}
    for i in range(len(flat)):
        parts = []
        if i in rel_pct:
            parts.append((w_rel, rel_pct[i]))
        if i in abs_pct:
            parts.append((1 - w_rel, abs_pct[i]))
        total = sum(w for w, _ in parts)
        per_signal[i] = (
            sum(w * p for w, p in parts) / total if total > 0 else sum(p for _, p in parts) / len(parts)
        )

    shrink = params["shrinkage"]
    results = []
    cursor = 0
    for app_id, rows in signals.items():
        indices = list(range(cursor, cursor + len(rows)))
        cursor += len(rows)
        # Overlapping signals of one kind (stars gained today, this week, this
        # month) are one piece of evidence seen three ways: they share a weight.
        group_sizes: dict[tuple[str, str], int] = defaultdict(int)
        for s in rows:
            group_sizes[(s.family, s.kind)] += 1
        weights = [
            params.trust(s.method.value) * params[f"weight.{s.kind}"] / group_sizes[(s.family, s.kind)]
            * (FIXED["anomaly_penalty"] if s.anomaly else 1.0)
            * freshness(s)
            for s in rows
        ]
        evidence = sum(weights)
        scores = [per_signal[i] for i in indices]
        score = (sum(w * x for w, x in zip(weights, scores)) + shrink * 0.5) / (evidence + shrink)
        results.append(AppScore(app_id, score, evidence, list(rows), scores))
    results.sort(key=lambda r: (-r.score, r.app_id))
    return results
