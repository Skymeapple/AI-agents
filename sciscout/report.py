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

from .pipeline import TrackReport
from .provenance import Grade

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
    assessment = report.assessment
    profile = report.profile
    lines: list[str] = []

    lines.append(f"## {report.name}")
    lines.append("")
    lines.append(
        f"**Priority {assessment.priority:.2f}** "
        f"(importance {assessment.importance:.2f}, urgency {assessment.urgency:.2f}) "
        f"| evidence confidence {assessment.confidence:.0%} "
        f"| {len(report.track.works)} works "
        f"| {report.track.independent_groups(months=None)} groups"
    )
    lines.append("")
    lines.append(f"Readiness: {profile.trl.describe()}")
    lines.append("")

    # -- scoring breakdown ---------------------------------------------------
    lines.append("### Why it scores this way")
    lines.append("")
    lines.append("| Axis | Sub-score | Value | Basis |")
    lines.append("| --- | --- | --- | --- |")
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

    # -- commercialisation path ----------------------------------------------
    lines.append("### Path to market")
    lines.append("")
    if not profile.remaining_stages:
        lines.append(
            "_No development stages remain at the inferred readiness level: this "
            "direction is modelled as already commercialised._"
        )
        lines.append("")
    else:
        lines.append(
            f"Sector: **{profile.sector_label}**. "
            f"Nominally **{profile.nominal_years_to_market():.1f} years** and "
            f"**{_money(profile.nominal_cost_usd())}** to market if every stage is "
            f"survived, with cumulative technical success of "
            f"**{profile.cumulative_probability():.1%}**."
        )
        lines.append("")
        lines.append("| Stage | TRL | Duration (y) | Cost | P(success) | Basis |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for stage in profile.remaining_stages:
            lines.append(
                f"| {stage.name} | {stage.from_trl}-{stage.to_trl} "
                f"| {stage.duration_years} | {stage.cost_usd} "
                f"| {stage.p_success} | _{_GRADE_MARK[stage.provenance.grade]}_ |"
            )
        lines.append("")

    if profile.constraints:
        lines.append("**Constraints**")
        lines.append("")
        for constraint in sorted(
            profile.constraints, key=lambda c: -c.severity_rank
        ):
            lines.append(f"- _{constraint.severity}_ **{constraint.kind}** — {constraint.note}")
        lines.append("")

    # -- simulation ----------------------------------------------------------
    simulation = report.simulation
    if simulation is None:
        lines.append("_No investment simulation was run for this direction._")
        lines.append("")
        return "\n".join(lines)

    npv = simulation.percentiles(simulation.project_npv)
    moic = simulation.percentiles(simulation.investor_moic)
    success = simulation.reached_market
    lines.append("### Investment case")
    lines.append("")
    lines.append(f"_{simulation.draws:,} Monte Carlo draws, seed {simulation.seed}._")
    lines.append("")
    lines.append("| Measure | P10 | P50 | P90 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| Project NPV | {_money(npv[10])} | {_money(npv[50])} | {_money(npv[90])} |"
    )
    lines.append(
        f"| Investor MOIC | {moic[10]:.2f}x | {moic[50]:.2f}x | {moic[90]:.2f}x |"
    )
    if success.any():
        years = simulation.years_to_market[success]
        spend = simulation.total_spend[success]
        own = simulation.final_ownership[success]
        lines.append(
            f"| Years to market _(successes)_ | {np.percentile(years, 10):.1f} "
            f"| {np.percentile(years, 50):.1f} | {np.percentile(years, 90):.1f} |"
        )
        lines.append(
            f"| Capital consumed _(successes)_ | {_money(np.percentile(spend, 10))} "
            f"| {_money(np.percentile(spend, 50))} | {_money(np.percentile(spend, 90))} |"
        )
        lines.append(
            f"| Investor stake at exit _(successes)_ | {np.percentile(own, 10):.2%} "
            f"| {np.percentile(own, 50):.2%} | {np.percentile(own, 90):.2%} |"
        )
    lines.append("")
    lines.append(
        f"- Probability of reaching market: **{simulation.success_rate:.1%}**"
    )
    lines.append(
        f"- Expected investor multiple across all outcomes: "
        f"**{simulation.expected_moic():.2f}x** "
        f"(**{simulation.conditional_moic():.2f}x** conditional on reaching market)"
    )
    lines.append(
        f"- Probability the project destroys value: **{simulation.probability_of_loss():.1%}**"
    )
    lines.append("")

    lines.append("**Where the programme dies**")
    lines.append("")
    for name, share in simulation.failure_breakdown().items():
        lines.append(f"- {name}: {share:.1%}")
    lines.append("")

    if detail:
        lines.append("**What drives the outcome** (rank correlation with project NPV)")
        lines.append("")
        lines.append("| Input | All draws | Conditional on success | Provenance |")
        lines.append("| --- | --- | --- | --- |")
        conditional = dict(
            (label, value)
            for label, value, _ in simulation.sensitivity(
                top=30, conditional_on_success=True
            )
        )
        for label, value, grade in simulation.sensitivity(top=8):
            cond = conditional.get(label)
            cond_text = f"{cond:+.2f}" if cond is not None else "—"
            lines.append(
                f"| {label} | {value:+.2f} | {cond_text} | _{_GRADE_MARK[grade]}_ |"
            )
        lines.append("")

    grade_mix = simulation.variance_by_grade()
    lines.append(
        f"**Provenance of this forecast:** "
        f"{grade_mix[Grade.OBSERVED]:.0%} of the spread traces to measured inputs, "
        f"{grade_mix[Grade.SOURCED]:.0%} to sourced figures, and "
        f"**{grade_mix[Grade.ASSUMED]:.0%} to assumptions**."
    )
    lines.append("")
    needing = simulation.ledger.needing_review()
    if needing:
        lines.append(
            f"{len(needing)} of {len(simulation.ledger.entries)} simulation inputs are "
            "flagged as placeholders awaiting a real number."
        )
        lines.append("")
    return "\n".join(lines)


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
