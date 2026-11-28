"""Markdown reporting.

The report is the product. A score nobody can interrogate is worse than no score,
so every section is built to be argued with: sub-scores are shown with their
bases, simulation results are shown with the inputs that drove them, and the
provenance of every number is on the page.

The one formatting rule that matters: a number never appears without its grade
nearby. A reader skimming for the ROI figure should not be able to avoid noticing
that it rests on assumptions.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .commercial.profile import CommercialProfile
from .invest.engine import MonteCarloResult
from .pipeline import TrackReport
from .provenance import Grade
from .scoring.importance import Assessment

_GRADE_MARK = {Grade.OBSERVED: "measured", Grade.SOURCED: "sourced", Grade.ASSUMED: "ASSUMED"}


def _money(value: float) -> str:
    """Format a dollar amount at a readable magnitude."""
    if not np.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{sign}${magnitude / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:,.0f}M"
    return f"{sign}${magnitude:,.0f}"


def render_track(report: TrackReport, detail: bool = True) -> str:
    """Full markdown section for one research direction."""
    lines = [
        *_header_lines(report),
        *_scoring_lines(report.assessment),
        *_path_lines(report.profile),
        *_constraint_lines(report.profile),
    ]
    if report.simulation is None:
        lines += ["_No investment simulation was run for this direction._", ""]
    else:
        lines += _investment_lines(report.simulation)
        lines += _failure_lines(report.simulation)
        if detail:
            lines += _sensitivity_lines(report.simulation)
        lines += _provenance_lines(report.simulation)
    return "\n".join(lines)


def _header_lines(report: TrackReport) -> list[str]:
    assessment = report.assessment
    return [
        f"## {report.name}",
        "",
        f"**Priority {assessment.priority:.2f}** "
        f"(importance {assessment.importance:.2f}, urgency {assessment.urgency:.2f}) "
        f"| evidence confidence {assessment.confidence:.0%} "
        f"| {len(report.track.works)} works "
        f"| {report.track.independent_groups(months=None)} groups",
        "",
        f"Readiness: {report.profile.trl.describe()}",
        "",
    ]


def _scoring_lines(assessment: Assessment) -> list[str]:
    lines = [
        "### Why it scores this way",
        "",
        "| Axis | Sub-score | Value | Basis |",
        "| --- | --- | --- | --- |",
    ]
    for axis, parts in (
        ("importance", assessment.importance_parts),
        ("urgency", assessment.urgency_parts),
    ):
        for key, estimate in parts.items():
            basis = estimate.provenance.detail
            if estimate.provenance.reference:
                basis += f" ({estimate.provenance.reference})"
            mark = _GRADE_MARK[estimate.grade]
            lines.append(f"| {axis} | {key} | {estimate.value:.2f} | _{mark}_ — {basis} |")
    lines.append("")
    return lines


def _path_lines(profile: CommercialProfile) -> list[str]:
    lines = ["### Path to market", ""]
    if not profile.remaining_stages:
        return lines + [
            "_No development stages remain at the inferred readiness level: this "
            "direction is modelled as already commercialised._",
            "",
        ]
    lines += [
        f"Sector: **{profile.sector_label}**. "
        f"Nominally **{profile.nominal_years_to_market():.1f} years** and "
        f"**{_money(profile.nominal_cost_usd())}** to market if every stage is "
        f"survived, with cumulative technical success of "
        f"**{profile.cumulative_probability():.1%}**.",
        "",
        "| Stage | TRL | Duration (y) | Cost | P(success) | Basis |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for stage in profile.remaining_stages:
        lines.append(
            f"| {stage.name} | {stage.from_trl}-{stage.to_trl} "
            f"| {stage.duration_years} | {stage.cost_usd} "
            f"| {stage.p_success} | _{_GRADE_MARK[stage.provenance.grade]}_ |"
        )
    lines.append("")
    return lines


def _constraint_lines(profile: CommercialProfile) -> list[str]:
    if not profile.constraints:
        return []
    lines = ["**Constraints**", ""]
    for constraint in sorted(profile.constraints, key=lambda c: -c.severity_rank):
        lines.append(f"- _{constraint.severity}_ **{constraint.kind}** — {constraint.note}")
    lines.append("")
    return lines


def _percentile_row(label: str, values: np.ndarray, fmt) -> str:
    """One P10/P50/P90 table row, each cell formatted by ``fmt``."""
    cells = " | ".join(fmt(np.percentile(values, q)) for q in (10, 50, 90))
    return f"| {label} | {cells} |"


def _investment_lines(simulation: MonteCarloResult) -> list[str]:
    npv = simulation.percentiles(simulation.project_npv)
    moic = simulation.percentiles(simulation.investor_moic)
    success = simulation.reached_market
    lines = [
        "### Investment case",
        "",
        f"_{simulation.draws:,} Monte Carlo draws, seed {simulation.seed}._",
        "",
        "| Measure | P10 | P50 | P90 |",
        "| --- | --- | --- | --- |",
        f"| Project NPV | {_money(npv[10])} | {_money(npv[50])} | {_money(npv[90])} |",
        f"| Investor MOIC | {moic[10]:.2f}x | {moic[50]:.2f}x | {moic[90]:.2f}x |",
    ]
    if success.any():
        lines += [
            _percentile_row(
                "Years to market _(successes)_",
                simulation.years_to_market[success],
                "{:.1f}".format,
            ),
            _percentile_row(
                "Capital consumed _(successes)_", simulation.total_spend[success], _money
            ),
            _percentile_row(
                "Investor stake at exit _(successes)_",
                simulation.final_ownership[success],
                "{:.2%}".format,
            ),
        ]
    lines += [
        "",
        f"- Probability of reaching market: **{simulation.success_rate:.1%}**",
        f"- Expected investor multiple across all outcomes: "
        f"**{simulation.expected_moic():.2f}x** "
        f"(**{simulation.conditional_moic():.2f}x** conditional on reaching market)",
        f"- Probability the project destroys value: **{simulation.probability_of_loss():.1%}**",
        "",
    ]
    return lines


def _failure_lines(simulation: MonteCarloResult) -> list[str]:
    lines = ["**Where the programme dies**", ""]
    for name, share in simulation.failure_breakdown().items():
        lines.append(f"- {name}: {share:.1%}")
    lines.append("")
    return lines


def _sensitivity_lines(simulation: MonteCarloResult) -> list[str]:
    lines = [
        "**What drives the outcome** (rank correlation with project NPV)",
        "",
        "| Input | All draws | Conditional on success | Provenance |",
        "| --- | --- | --- | --- |",
    ]
    conditional = {
        label: value
        for label, value, _ in simulation.sensitivity(top=30, conditional_on_success=True)
    }
    for label, value, grade in simulation.sensitivity(top=8):
        cond = conditional.get(label)
        cond_text = f"{cond:+.2f}" if cond is not None else "—"
        lines.append(f"| {label} | {value:+.2f} | {cond_text} | _{_GRADE_MARK[grade]}_ |")
    lines.append("")
    return lines


def _provenance_lines(simulation: MonteCarloResult) -> list[str]:
    grade_mix = simulation.variance_by_grade()
    lines = [
        f"**Provenance of this forecast:** "
        f"{grade_mix[Grade.OBSERVED]:.0%} of the spread traces to measured inputs, "
        f"{grade_mix[Grade.SOURCED]:.0%} to sourced figures, and "
        f"**{grade_mix[Grade.ASSUMED]:.0%} to assumptions**.",
        "",
    ]
    needing = simulation.ledger.needing_review()
    if needing:
        lines += [
            f"{len(needing)} of {len(simulation.ledger.entries)} simulation inputs are "
            "flagged as placeholders awaiting a real number.",
            "",
        ]
    return lines


def render_portfolio(reports: list[TrackReport], asof: dt.date | None = None) -> str:
    """Full report: ranking table, then a section per direction."""
    asof = asof or dt.date.today()
    lines = [
        "# Research portfolio review",
        "",
        f"_Generated {asof.isoformat()} by sciscout._",
        "",
        "## Ranking",
        "",
        "| # | Direction | Priority | Importance | Urgency | Confidence | TRL | P(market) | E[MOIC] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, report in enumerate(reports, start=1):
        assessment = report.assessment
        simulation = report.simulation
        p_market = f"{simulation.success_rate:.0%}" if simulation else "—"
        moic = f"{simulation.expected_moic():.2f}x" if simulation else "—"
        lines.append(
            f"| {index} | {report.name} | **{assessment.priority:.2f}** "
            f"| {assessment.importance:.2f} | {assessment.urgency:.2f} "
            f"| {assessment.confidence:.0%} | {report.profile.trl.mode} "
            f"| {p_market} | {moic} |"
        )
    lines.append("")
    lines.append(
        "Priority combines importance and urgency geometrically, so a direction "
        "cannot rank highly on one axis alone. Confidence is the share of scoring "
        "inputs that were measured rather than assumed — read it alongside "
        "priority, because a confident 0.5 is a better basis for a decision than "
        "an unfounded 0.8."
    )
    lines.append("")
    for report in reports:
        lines.append(render_track(report))
    return "\n".join(lines)
