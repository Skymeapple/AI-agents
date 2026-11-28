"""The forecast ledger: dated predictions from every project, scored as reality arrives.

Most projects in this repository say something about the future -- a
probability that a milestone lands by a date, a quantile band for a global
series in 2030. Each said it in its own report and none was ever checked. The
ledger holds them in one file, in one format, so that when a date passes the
forecast is confronted with what happened and the project that made it
acquires a record.

Three rules, all enforced on load rather than left to care:

*A forecast must be checkable.* It names a quantity or event, a resolution
date, a value, and a rule saying what data would settle it. An entry missing
any of these is rejected, because a prediction nobody can settle cannot be
wrong and so is not worth recording.

*A forecast must be traceable.* It names the file it came from and an anchor
string that is still in that file. A forecast known only from a session summary
may be imported, but it says so and carries the reason it cannot be verified.

*An outcome must be sourced.* A resolution names where the outcome was read
and on what date. Nothing is resolved from memory, and a "no" cannot be
recorded before the deadline has passed.

Scores are reported with their counts and a Wilson interval. A Brier score on
three forecasts is an anecdote; the report says so instead of ranking projects
on it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from sciscout.provenance import Grade, Provenance, parse_provenance

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "data" / "nexus" / "ledger"
FORECASTS_PATH = LEDGER_DIR / "forecasts.yaml"
RESOLUTIONS_PATH = LEDGER_DIR / "resolutions.yaml"

KINDS = ("binary", "interval", "quantiles")

# Below this many resolved forecasts a group's score is shown but labelled as
# too thin to read as a track record. This is a reading convention, not a
# statistic: the Wilson interval beside every rate is the real statement of
# how little a small sample says.
MIN_TRACK_RECORD = 20


class LedgerError(ValueError):
    """A ledger or resolution entry that breaks one of the rules above."""


# -- entries ---------------------------------------------------------------------


def _date(value, context: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        raise LedgerError(f"{context}: {value!r} is not an ISO date (YYYY-MM-DD)") from None


def _text(raw: dict, key: str, context: str) -> str:
    value = " ".join(str(raw.get(key) or "").split())
    if not value:
        raise LedgerError(f"{context}: missing '{key}'")
    return value


@dataclass(frozen=True)
class Source:
    """Where a forecast was read. Either a file in the repository or a session."""

    file: str | None = None
    locator: str | None = None
    anchor: str | None = None
    session: str | None = None
    note: str | None = None

    @property
    def in_repository(self) -> bool:
        return self.file is not None

    def describe(self) -> str:
        if self.in_repository:
            return f"{self.file} ({self.locator})"
        return f"session summary: {self.session} -- not in this repository"


@dataclass(frozen=True)
class Forecast:
    id: str
    statement: str
    project: str
    source: Source
    made: dt.date
    resolves: dt.date
    kind: str
    provenance: Provenance
    resolution_rule: str
    probability: float | None = None
    probability_range: tuple[float, float] | None = None
    low: float | None = None
    high: float | None = None
    level: float | None = None
    quantiles: dict[float, float] = field(default_factory=dict)
    unit: str | None = None
    evidence: str | None = None
    conditional: str | None = None
    note: str | None = None

    @property
    def grade(self) -> Grade:
        return self.provenance.grade

    def central_interval(self) -> tuple[float, float, float]:
        """``(low, high, level)`` for interval and quantile forecasts."""
        if self.kind == "interval":
            return self.low, self.high, self.level
        if self.kind == "quantiles":
            lo, hi = min(self.quantiles), max(self.quantiles)
            return self.quantiles[lo], self.quantiles[hi], hi - lo
        raise LedgerError(f"{self.id}: a binary forecast has no interval")

    def stated(self) -> str:
        """The forecast as a short string, for tables."""
        if self.kind == "binary":
            if self.probability_range:
                lo, hi = self.probability_range
                return f"{lo:.0%}-{hi:.0%}"
            return f"{self.probability:.1%}" if self.probability > 0.95 else f"{self.probability:.0%}"
        low, high, level = self.central_interval()
        unit = f" {self.unit}" if self.unit else ""
        middle = f"{self.quantiles[0.5]:g} " if 0.5 in self.quantiles else ""
        return f"{middle}({low:g} to {high:g}, {level:.0%}){unit}"


def _parse_source(raw, context: str) -> Source:
    if not isinstance(raw, dict):
        raise LedgerError(f"{context}: missing 'source' saying where the forecast was read")
    if raw.get("file"):
        for key in ("locator", "anchor"):
            _text(raw, key, f"{context}: source")
        return Source(file=str(raw["file"]), locator=str(raw["locator"]), anchor=str(raw["anchor"]))
    if raw.get("session"):
        note = _text(raw, "note", f"{context}: a forecast imported from a session needs a 'note' "
                                  f"saying why it cannot be checked here; source")
        return Source(session=str(raw["session"]), note=note)
    raise LedgerError(f"{context}: source needs a 'file' (with locator and anchor) or a 'session'")


def _probability(value, context: str) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise LedgerError(f"{context}: probability {value!r} is not a number") from None
    if not 0.0 < p < 1.0:
        raise LedgerError(
            f"{context}: probability {p} must lie strictly between 0 and 1 -- a certainty "
            f"cannot be scored with a log score and is not a forecast"
        )
    return p


def parse_forecast(raw: dict) -> Forecast:
    """Build a :class:`Forecast`, rejecting anything that could not be settled or traced."""
    if not isinstance(raw, dict) or not raw.get("id"):
        raise LedgerError(f"ledger entry without an id: {raw!r}")
    ctx = str(raw["id"])
    kind = str(raw.get("kind", ""))
    if kind not in KINDS:
        raise LedgerError(f"{ctx}: kind {kind!r} must be one of {', '.join(KINDS)}")
    made = _date(raw.get("made"), f"{ctx}: made")
    resolves = _date(raw.get("resolves"), f"{ctx}: resolves")
    if resolves < made:
        raise LedgerError(f"{ctx}: resolves on {resolves}, before it was made on {made}")
    try:
        provenance = parse_provenance(raw.get("provenance"), ctx)
    except ValueError as exc:
        raise LedgerError(str(exc)) from None

    fields: dict = {}
    if kind == "binary":
        fields["probability"] = _probability(raw.get("probability"), ctx)
        if raw.get("probability_range") is not None:
            lo, hi = (_probability(v, ctx) for v in raw["probability_range"])
            if not lo <= fields["probability"] <= hi:
                raise LedgerError(f"{ctx}: probability lies outside its stated range")
            fields["probability_range"] = (lo, hi)
    elif kind == "interval":
        try:
            low, high, level = float(raw["low"]), float(raw["high"]), float(raw["level"])
        except (KeyError, TypeError, ValueError):
            raise LedgerError(f"{ctx}: an interval forecast needs numeric low, high and level") from None
        if not low < high or not 0.0 < level < 1.0:
            raise LedgerError(f"{ctx}: interval needs low < high and a level between 0 and 1")
        fields.update(low=low, high=high, level=level)
    else:
        raw_q = raw.get("quantiles")
        if not isinstance(raw_q, dict) or len(raw_q) < 2:
            raise LedgerError(f"{ctx}: a quantile forecast needs at least two quantiles")
        quantiles = {float(k): float(v) for k, v in raw_q.items()}
        levels = sorted(quantiles)
        if not all(0.0 < q < 1.0 for q in levels):
            raise LedgerError(f"{ctx}: quantile levels must lie between 0 and 1")
        if any(quantiles[a] > quantiles[b] for a, b in zip(levels, levels[1:])):
            raise LedgerError(f"{ctx}: quantile values must not decrease")
        if abs(levels[0] + levels[-1] - 1.0) > 1e-9:
            raise LedgerError(f"{ctx}: outer quantiles must be symmetric (e.g. 0.1 and 0.9)")
        if quantiles[levels[0]] == quantiles[levels[-1]]:
            raise LedgerError(f"{ctx}: the quantile band has zero width and cannot be scored")
        fields["quantiles"] = quantiles

    return Forecast(
        id=ctx,
        statement=_text(raw, "statement", ctx),
        project=_text(raw, "project", ctx),
        source=_parse_source(raw.get("source"), ctx),
        made=made,
        resolves=resolves,
        kind=kind,
        provenance=provenance,
        resolution_rule=_text(
            raw, "resolution_rule",
            f"{ctx}: a forecast must say what data would settle it; entry"),
        unit=raw.get("unit"),
        evidence=raw.get("evidence"),
        conditional=raw.get("conditional"),
        note=raw.get("note"),
        **fields,
    )


def load_forecasts(path: Path = FORECASTS_PATH) -> list[Forecast]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    forecasts = [parse_forecast(entry) for entry in raw]
    seen: set[str] = set()
    for f in forecasts:
        if f.id in seen:
            raise LedgerError(f"{f.id}: duplicate forecast id")
        seen.add(f.id)
    return forecasts


def check_anchor(forecast: Forecast, root: Path = ROOT) -> str | None:
    """Why the forecast can no longer be traced to its source, or ``None`` if it can."""
    if not forecast.source.in_repository:
        return None
    path = root / forecast.source.file
    if not path.exists():
        return f"{forecast.id}: source file {forecast.source.file} does not exist"
    if forecast.source.anchor not in path.read_text():
        return f"{forecast.id}: anchor text not found in {forecast.source.file}"
    return None


# -- resolutions -----------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    id: str
    outcome: bool | float | None  # None when void
    source: str
    resolved_on: dt.date
    void: bool = False
    note: str | None = None


def parse_resolution(raw: dict, forecasts: dict[str, Forecast]) -> Resolution:
    """Validate one outcome against the forecast it settles."""
    if not isinstance(raw, dict) or not raw.get("id"):
        raise LedgerError(f"resolution without an id: {raw!r}")
    ctx = f"resolution of {raw['id']}"
    forecast = forecasts.get(str(raw["id"]))
    if forecast is None:
        raise LedgerError(f"{ctx}: no such forecast in the ledger")
    source = _text(raw, "source", f"{ctx}: an outcome must name where it was read; entry")
    resolved_on = _date(raw.get("resolved_on"), f"{ctx}: resolved_on")
    if resolved_on < forecast.made:
        raise LedgerError(f"{ctx}: resolved on {resolved_on}, before the forecast was made")
    note = raw.get("note")
    outcome = raw.get("outcome")

    if isinstance(outcome, str) and outcome.strip().lower() == "void":
        if not note:
            raise LedgerError(f"{ctx}: a void resolution needs a note saying why")
        return Resolution(forecast.id, None, source, resolved_on, void=True, note=str(note))

    if forecast.kind == "binary":
        if isinstance(outcome, str):
            lowered = outcome.strip().lower()
            outcome = {"yes": True, "true": True, "no": False, "false": False}.get(lowered)
        if not isinstance(outcome, bool):
            raise LedgerError(f"{ctx}: a binary forecast resolves to yes or no (or void)")
        if outcome is False and resolved_on < forecast.resolves:
            raise LedgerError(
                f"{ctx}: cannot record 'no' on {resolved_on}; the deadline is "
                f"{forecast.resolves} and the event may still happen"
            )
    else:
        if isinstance(outcome, bool):
            raise LedgerError(f"{ctx}: a {forecast.kind} forecast resolves to a number")
        try:
            outcome = float(outcome)
        except (TypeError, ValueError):
            raise LedgerError(f"{ctx}: a {forecast.kind} forecast resolves to a number") from None
        if not math.isfinite(outcome):
            raise LedgerError(f"{ctx}: outcome must be finite")
    return Resolution(forecast.id, outcome, source, resolved_on, note=note)


def load_resolutions(
    forecasts: list[Forecast], path: Path = RESOLUTIONS_PATH
) -> dict[str, Resolution]:
    by_id = {f.id: f for f in forecasts}
    path = Path(path)
    raw = (yaml.safe_load(path.read_text()) or []) if path.exists() else []
    resolutions: dict[str, Resolution] = {}
    for entry in raw:
        resolution = parse_resolution(entry, by_id)
        if resolution.id in resolutions:
            raise LedgerError(f"{resolution.id}: resolved twice")
        resolutions[resolution.id] = resolution
    return resolutions


def record_resolution(
    raw: dict, forecasts: list[Forecast], path: Path = RESOLUTIONS_PATH
) -> Resolution:
    """Validate ``raw`` and append it to the resolutions file."""
    existing = load_resolutions(forecasts, path)
    resolution = parse_resolution(raw, {f.id: f for f in forecasts})
    if resolution.id in existing:
        raise LedgerError(f"{resolution.id}: already resolved; edit {path} to correct it")
    entry = {
        "id": resolution.id,
        "outcome": "void" if resolution.void else resolution.outcome,
        "source": resolution.source,
        "resolved_on": resolution.resolved_on.isoformat(),
    }
    if resolution.note:
        entry["note"] = resolution.note
    path = Path(path)
    text = path.read_text() if path.exists() else ""
    comments = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
    current = (yaml.safe_load(text) or []) if text else []
    current.append(entry)
    path.write_text(comments + yaml.safe_dump(current, sort_keys=False, allow_unicode=True, width=1000))
    return resolution


# -- scoring ---------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` of ``n``. Same formula as ``aisci.verifiers.wilson``,
    repeated here so the ledger does not need sympy to load."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def brier(probability: float, happened: bool) -> float:
    """Squared error of a probability. 0 is perfect; always saying 50% scores 0.25."""
    return (probability - float(happened)) ** 2


def log_score(probability: float, happened: bool) -> float:
    """Negative log likelihood in nats. 0 is perfect; always saying 50% scores ln 2."""
    return -math.log(probability if happened else 1.0 - probability)


def interval_score(low: float, high: float, level: float, outcome: float) -> float:
    """Winkler's interval score: width plus ``2 / alpha`` times any miss distance.

    Proper for a central interval at ``level``: a forecaster cannot improve it
    by stating a band wider or narrower than they believe.
    """
    alpha = 1.0 - level
    miss = max(low - outcome, 0.0) + max(outcome - high, 0.0)
    return (high - low) + 2.0 / alpha * miss


def pinball(quantiles: dict[float, float], outcome: float) -> float:
    """Mean pinball loss across the stated quantiles."""
    total = 0.0
    for q, value in quantiles.items():
        diff = outcome - value
        total += q * diff if diff >= 0 else (q - 1.0) * diff
    return total / len(quantiles)


@dataclass(frozen=True)
class Scored:
    forecast: Forecast
    resolution: Resolution
    brier: float | None = None
    log: float | None = None
    covered: bool | None = None
    interval: float | None = None
    # Interval score divided by the stated width, so forecasts in different
    # units can sit in one column. 1.0 means the outcome fell inside.
    interval_relative: float | None = None
    pinball: float | None = None


def score_forecast(forecast: Forecast, resolution: Resolution | None) -> Scored:
    """Score one resolved forecast. An unresolved or void one is an error, never a zero."""
    if resolution is None:
        raise LedgerError(f"{forecast.id}: not resolved, so it has no score")
    if resolution.void:
        raise LedgerError(f"{forecast.id}: resolved void, so it has no score")
    if forecast.kind == "binary":
        return Scored(forecast, resolution,
                      brier=brier(forecast.probability, resolution.outcome),
                      log=log_score(forecast.probability, resolution.outcome))
    low, high, level = forecast.central_interval()
    score = interval_score(low, high, level, resolution.outcome)
    return Scored(
        forecast, resolution,
        covered=low <= resolution.outcome <= high,
        interval=score,
        interval_relative=score / (high - low),
        pinball=pinball(forecast.quantiles, resolution.outcome) if forecast.quantiles else None,
    )


@dataclass
class Scorecard:
    """Aggregate over one group of scored forecasts."""

    name: str
    open: int = 0
    void: int = 0
    binary: list[Scored] = field(default_factory=list)
    ranged: list[Scored] = field(default_factory=list)

    @property
    def resolved(self) -> int:
        return len(self.binary) + len(self.ranged)

    @property
    def thin(self) -> bool:
        return self.resolved < MIN_TRACK_RECORD

    def _mean(self, items: list[Scored], attr: str) -> float | None:
        values = [getattr(s, attr) for s in items if getattr(s, attr) is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_brier(self) -> float | None:
        return self._mean(self.binary, "brier")

    @property
    def mean_log(self) -> float | None:
        return self._mean(self.binary, "log")

    @property
    def mean_interval_relative(self) -> float | None:
        return self._mean(self.ranged, "interval_relative")

    def hit_rate(self) -> tuple[int, int, tuple[float, float]] | None:
        """Binary forecasts on the right side of 50%: ``(hits, n, wilson)``."""
        if not self.binary:
            return None
        hits = sum((s.forecast.probability >= 0.5) == s.resolution.outcome for s in self.binary)
        return hits, len(self.binary), wilson(hits, len(self.binary))

    def coverage(self) -> tuple[int, int, tuple[float, float], float] | None:
        """Outcomes inside the stated band: ``(covered, n, wilson, mean nominal level)``."""
        if not self.ranged:
            return None
        covered = sum(bool(s.covered) for s in self.ranged)
        nominal = sum(s.forecast.central_interval()[2] for s in self.ranged) / len(self.ranged)
        return covered, len(self.ranged), wilson(covered, len(self.ranged)), nominal


def score(
    forecasts: list[Forecast], resolutions: dict[str, Resolution]
) -> dict[str, dict[str, Scorecard]]:
    """Scorecards overall, per project and per provenance grade.

    Only resolved forecasts contribute to a score. Open and void ones are
    counted beside it so a mean over two forecasts cannot pass for a mean over
    sixty.
    """
    cards: dict[str, dict[str, Scorecard]] = {"all": {}, "project": {}, "grade": {}}

    def card(group: str, name: str) -> Scorecard:
        return cards[group].setdefault(name, Scorecard(name))

    for forecast in forecasts:
        targets = [card("all", "all forecasts"), card("project", forecast.project),
                   card("grade", forecast.grade.value)]
        resolution = resolutions.get(forecast.id)
        if resolution is None:
            for c in targets:
                c.open += 1
            continue
        if resolution.void:
            for c in targets:
                c.void += 1
            continue
        scored = score_forecast(forecast, resolution)
        for c in targets:
            (c.binary if forecast.kind == "binary" else c.ranged).append(scored)
    return cards


def due(
    forecasts: list[Forecast], resolutions: dict[str, Resolution], asof: dt.date
) -> list[Forecast]:
    """Unresolved forecasts whose deadline has passed, oldest first."""
    pending = [f for f in forecasts if f.id not in resolutions and f.resolves <= asof]
    return sorted(pending, key=lambda f: (f.resolves, f.id))


def upcoming(
    forecasts: list[Forecast], resolutions: dict[str, Resolution], asof: dt.date, days: int
) -> list[Forecast]:
    horizon = asof + dt.timedelta(days=days)
    pending = [f for f in forecasts if f.id not in resolutions and asof < f.resolves <= horizon]
    return sorted(pending, key=lambda f: (f.resolves, f.id))


# -- report ----------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _card_rows(cards: dict[str, Scorecard]) -> list[str]:
    rows = ["| Group | Open | Resolved | Void | Brier | Log score | Right side of 50% | "
            "Band coverage | Read as |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in sorted(cards):
        c = cards[name]
        hit, cover = c.hit_rate(), c.coverage()
        hit_text = "-" if hit is None else (
            f"{hit[0]} of {hit[1]} (95% CI {hit[2][0]:.0%}-{hit[2][1]:.0%})")
        cover_text = "-" if cover is None else (
            f"{cover[0]} of {cover[1]} (95% CI {cover[2][0]:.0%}-{cover[2][1]:.0%}; "
            f"stated {cover[3]:.0%})")
        if c.resolved == 0:
            reading = "nothing to score"
        elif c.thin:
            reading = f"too few to be a track record (under {MIN_TRACK_RECORD})"
        else:
            reading = "a record"
        rows.append(f"| {name} | {c.open} | {c.resolved} | {c.void} | {_fmt(c.mean_brier)} | "
                    f"{_fmt(c.mean_log)} | {hit_text} | {cover_text} | {reading} |")
    return rows


def _forecast_rows(items: list[Forecast]) -> list[str]:
    rows = ["| Resolves | Forecast | Stated | Project | Settled by |", "| --- | --- | --- | --- | --- |"]
    for f in items:
        rows.append(f"| {f.resolves} | {f.statement} | {f.stated()} | {f.project} | "
                    f"{f.resolution_rule} |")
    return rows


def render(
    forecasts: list[Forecast],
    resolutions: dict[str, Resolution],
    asof: dt.date,
    horizon_days: int = 400,
    root: Path = ROOT,
) -> str:
    cards = score(forecasts, resolutions)
    overall = cards["all"].get("all forecasts", Scorecard("all forecasts"))
    overdue = due(forecasts, resolutions, asof)
    soon = upcoming(forecasts, resolutions, asof, horizon_days)
    broken = [msg for msg in (check_anchor(f, root) for f in forecasts) if msg]
    imported = [f for f in forecasts if not f.source.in_repository]
    conditional = [f for f in forecasts if f.conditional]

    lines = [
        "# Forecast ledger", "",
        f"_As of {asof}: {len(forecasts)} forecasts from "
        f"{len(cards['project'])} projects; {overall.resolved} resolved, {overall.void} void, "
        f"{overall.open} open, of which {len(overdue)} past their deadline and awaiting an "
        f"outcome._", "",
    ]
    if overall.resolved == 0:
        lines += [
            "**No forecast has been resolved, so there are no scores.** Every figure below "
            "is a count. The first deadlines have only just arrived, and an outcome is "
            "recorded only with a named source, never from memory.", "",
        ]
    by_grade = {g: sum(f.grade.value == g for f in forecasts) for g in ("observed", "sourced", "assumed")}
    kinds = {k: sum(f.kind == k for f in forecasts) for k in KINDS}
    lines += [
        f"- By kind: {kinds['binary']} probabilities of an event, {kinds['quantiles']} quantile "
        f"bands, {kinds['interval']} interval(s).",
        f"- By provenance of the forecast itself: {by_grade['observed']} observed, "
        f"{by_grade['sourced']} sourced, {by_grade['assumed']} assumed.",
        f"- {len(imported)} imported from a session summary whose model is not in this "
        f"repository; {len(conditional)} conditional on a stated scenario.",
        f"- Traceability: {len(forecasts) - len(imported) - len(broken)} of "
        f"{len(forecasts) - len(imported)} repository forecasts still match their source text.",
        "",
    ]
    if broken:
        lines += ["**Broken source links** (the source file changed or moved):", ""]
        lines += [f"- {msg}" for msg in broken] + [""]

    lines += ["## Past deadline, awaiting an outcome", ""]
    if overdue:
        lines += _forecast_rows(overdue) + [""]
        lines += ["Record each with `python -m nexus.cli ledger resolve --id ... --outcome ... "
                  "--source ... --date ...`.", ""]
    else:
        lines += ["None.", ""]

    lines += [f"## Due in the next {horizon_days} days", ""]
    lines += (_forecast_rows(soon) if soon else ["None."]) + [""]

    for title, group in (("Scores by project", "project"), ("Scores by provenance grade", "grade")):
        lines += [f"## {title}", ""] + _card_rows(cards[group]) + [""]
    lines += [
        "Brier: 0 is perfect, 0.25 is what always answering 50% earns. Log score: 0 is "
        "perfect, 0.693 is what 50% earns. Band coverage should match the stated level; the "
        "interval beside it shows how far a small sample is from establishing that. Forecasts "
        "with two deadlines for one milestone are separate entries and are not independent, so "
        "the intervals are somewhat too narrow.", "",
    ]

    scored = [s for c in cards["all"].values() for s in c.binary + c.ranged]
    if scored:
        lines += ["## Resolved", "",
                  "| Forecast | Stated | Outcome | Brier | In band | Source of outcome |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for s in sorted(scored, key=lambda s: s.resolution.resolved_on):
            outcome = s.resolution.outcome
            shown = ("yes" if outcome else "no") if isinstance(outcome, bool) else f"{outcome:g}"
            band = "-" if s.covered is None else ("yes" if s.covered else "no")
            lines.append(f"| {s.forecast.statement} | {s.forecast.stated()} | {shown} | "
                         f"{_fmt(s.brier)} | {band} | {s.resolution.source} "
                         f"({s.resolution.resolved_on}) |")
        lines.append("")

    if imported:
        lines += ["## Imported, not verifiable here", ""]
        lines += [f"- **{f.statement}** ({f.stated()}, resolves {f.resolves}). {f.source.note}"
                  for f in imported] + [""]
    return "\n".join(lines)


# -- command line ----------------------------------------------------------------


def _load(args) -> tuple[list[Forecast], dict[str, Resolution]]:
    forecasts = load_forecasts(Path(args.file))
    return forecasts, load_resolutions(forecasts, Path(args.resolutions))


def _asof(args) -> dt.date:
    return _date(args.asof, "--asof") if args.asof else dt.date.today()


def cmd_report(args) -> int:
    forecasts, resolutions = _load(args)
    text = render(forecasts, resolutions, _asof(args), args.horizon)
    if getattr(args, "out", None):
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_due(args) -> int:
    forecasts, resolutions = _load(args)
    asof = _asof(args)
    overdue = due(forecasts, resolutions, asof)
    print(f"{len(overdue)} forecasts past their deadline as of {asof}, awaiting an outcome\n")
    if overdue:
        print("\n".join(_forecast_rows(overdue)))
        print("\nids: " + ", ".join(f.id for f in overdue))
    return 0


def cmd_resolve(args) -> int:
    forecasts = load_forecasts(Path(args.file))
    raw = {"id": args.id, "outcome": args.outcome, "source": args.source,
           "resolved_on": args.date, "note": args.note}
    resolution = record_resolution(raw, forecasts, Path(args.resolutions))
    forecast = next(f for f in forecasts if f.id == resolution.id)
    if resolution.void:
        print(f"{forecast.id}: recorded as void")
        return 0
    scored = score_forecast(forecast, resolution)
    detail = (f"Brier {scored.brier:.3f}" if scored.brier is not None
              else f"{'inside' if scored.covered else 'outside'} the stated band")
    print(f"{forecast.id}: recorded; {detail}")
    return 0


def cmd_score(args) -> int:
    forecasts, resolutions = _load(args)
    cards = score(forecasts, resolutions)
    overall = cards["all"].get("all forecasts", Scorecard("all forecasts"))
    if overall.resolved == 0:
        print(f"No forecast has been resolved ({overall.open} open, {overall.void} void): "
              f"there is nothing to score.")
        return 0
    for title, group in (("Overall", "all"), ("By project", "project"), ("By grade", "grade")):
        print(f"## {title}\n")
        print("\n".join(_card_rows(cards[group])) + "\n")
    return 0


def _run(func):
    def wrapped(args) -> int:
        try:
            return func(args)
        except LedgerError as exc:
            print(f"error: {exc}")
            return 2
    return wrapped


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "ledger", help="dated forecasts from every project, scored as reality arrives",
        description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--file", default=str(FORECASTS_PATH), help="ledger file")
    common.add_argument("--resolutions", default=str(RESOLUTIONS_PATH), help="outcomes file")
    common.add_argument("--asof", default=None, help="date to report as of (default today)")
    common.add_argument("--horizon", type=int, default=400,
                        help="how many days ahead to list upcoming deadlines")
    for action in common._actions:
        parser._add_action(action)
    parser.add_argument("--out", default=None, help="write the report here instead of printing")
    parser.set_defaults(func=_run(cmd_report))

    actions = parser.add_subparsers(dest="ledger_action")
    report = actions.add_parser("report", parents=[common], help="the full summary (default)")
    report.add_argument("--out", default=None)
    report.set_defaults(func=_run(cmd_report))
    actions.add_parser("due", parents=[common], help="forecasts past their deadline"
                       ).set_defaults(func=_run(cmd_due))
    actions.add_parser("score", parents=[common], help="scores of resolved forecasts"
                       ).set_defaults(func=_run(cmd_score))
    resolve = actions.add_parser("resolve", parents=[common], help="record an outcome")
    resolve.add_argument("--id", required=True)
    resolve.add_argument("--outcome", required=True, help="yes / no / a number / void")
    resolve.add_argument("--source", required=True, help="where the outcome was read")
    resolve.add_argument("--date", required=True, help="date the outcome was read (YYYY-MM-DD)")
    resolve.add_argument("--note", default=None)
    resolve.set_defaults(func=_run(cmd_resolve))
