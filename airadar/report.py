"""The daily report.

Ordered so the reader meets the claim and its warrant together: every growth
figure says whether the radar measured it or a page stated it, every heuristic
says it is one, and the radar's own forecasting record sits in the same document
as its forecasts.
"""

from __future__ import annotations

from sciscout.provenance import Estimate, Grade, ProvenanceLedger

from .pipeline import Analysis
from .iterate import MIN_IMPROVEMENT, MIN_WIN_SHARE
from .params import KNOBS

TOP_N = 25
MOVERS_N = 15


def _link(analysis: Analysis, app_id: str) -> str:
    app = analysis.apps.get(app_id)
    return f"[{app.name}]({app.url})" if app else app_id


def _evidence_ledger(analysis: Analysis) -> ProvenanceLedger:
    ledger = ProvenanceLedger()
    for score in analysis.scores[:TOP_N]:
        for signal in score.signals:
            for o in signal.evidence:
                ledger.record(Estimate(o.value, o.provenance(), label=f"{o.app_id} {o.metric}"))
    return ledger


def render(analysis: Analysis) -> str:
    a = analysis
    lines = [f"# AI app radar — {a.date.isoformat()}", ""]

    ledger = _evidence_ledger(a)
    counts = ledger.counts()
    total = sum(counts.values()) or 1
    healthy = sum(1 for h in a.health if h.consecutive_failures == 0)
    lines += [
        f"**{len(a.apps)} apps tracked** · {len(a.scores)} with growth evidence · "
        f"{len(a.baseline_only)} baseline-only (growth measurable from the next run) · "
        f"sources healthy {healthy}/{len(a.health)}",
        "",
        f"Evidence behind the top {min(TOP_N, len(a.scores))}: "
        f"{counts[Grade.OBSERVED] / total:.0%} measured by the radar from APIs, "
        f"{counts[Grade.SOURCED] / total:.0%} read off pages or reported by companies "
        f"(sourced, needs review).",
        "",
    ]
    if a.ingest_log:
        lines += ["Ingested today:", ""] + [f"- {entry}" for entry in a.ingest_log] + [""]

    lines += ["## Fastest-growing apps", ""]
    if not a.scores:
        lines += ["_No growth evidence yet._", ""]
    else:
        lines += [
            "Score is the evidence-weighted percentile of each app's growth signals, shrunk toward "
            "0.5 when evidence is thin. Evidence is the summed signal weight: below ~0.7 means one "
            "page-read signal.",
            "",
            "| # | App | Kind | Tags | Score | Evidence | Strongest signal | Flags |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for rank, s in enumerate(a.scores[:TOP_N], 1):
            app = a.apps[s.app_id]
            flags = "; ".join(s.anomalies) or ""
            lines.append(
                f"| {rank} | {_link(a, s.app_id)} | {app.kind.value} | {', '.join(app.tags[:3])} | "
                f"{s.score:.2f} | {s.evidence:.2f} | {s.headline()} | {flags} |"
            )
        lines.append("")
    if a.movers:
        lines += [
            "### Measured since the last reading", "",
            "Stars gained between the radar's own two latest readings of each app, not a gain a "
            "page stated. ≈ marks a rounded page reading (accurate to about ±50). Per day before is "
            "the pace between the two readings before that, so a faster or slower day shows; – "
            "means the radar has no earlier pair.", "",
            "| App | Previous | Now | Change | Per day | Per day before |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for m in a.movers[:MOVERS_N]:
            def mark(rounded: bool) -> str:
                return "≈" if rounded else ""
            lines.append(
                f"| {_link(a, m.app_id)} | {mark(m.before_rounded)}{m.before:,.0f} ({m.start.isoformat()}) | "
                f"{mark(m.after_rounded)}{m.after:,.0f} | {mark(m.rounded)}{m.change:+,.0f} | "
                f"{mark(m.rounded)}{m.per_day:+,.0f} | "
                + (f"{mark(m.prior_rounded)}{m.prior_per_day:+,.0f} |" if m.prior_per_day is not None else "– |")
            )
        lines += ["", f"{len(a.movers)} apps were read on both runs.", ""]
    if a.gone:
        lines += ["### Removed from the ranking", "",
                  "Pages found gone (deleted or made private). Their observations are kept; "
                  "they are not scored while gone.", ""]
        for app_id, row in sorted(a.gone.items()):
            lines.append(f"- {_link(a, app_id)}: gone since {row['observed_on']} "
                         f"([checked]({row['source_url']})). {row['note']}".rstrip())
        lines.append("")

    lines += ["## Directions", ""]
    if not a.directions:
        lines += ["_Too few tagged apps with growth evidence to form directions._", ""]
    else:
        lines += [
            "Top-quartile share is the fraction of a direction's scored apps in the top quarter of "
            "the whole ranking. Status compares the mean score with the previous snapshot.",
            "",
            "| Direction | Status | Top-quartile share | Mean score | Change | Apps (open / commercial) | Leaders |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for d in a.directions[:20]:
            change = f"{d.change:+.2f}" if d.change is not None else "–"
            leaders = ", ".join(_link(a, i) for i, _ in d.scored[:3])
            lines.append(
                f"| {d.key} | {d.status} | {d.top_share:.0%} | {d.mean_score:.2f} | {change} | "
                f"{len(d.members)} ({d.open_count} / {d.commercial_count}) | {leaders} |"
            )
        lines.append("")

    if a.emergent:
        lines += [
            "### Emergent groupings",
            "",
            "Lexical clusters whose distinctive terms match no existing tag — candidate directions "
            "the vocabulary has not named yet.",
            "",
        ]
        for g in a.emergent:
            members = ", ".join(_link(a, m) for m in g.members[:6])
            lines.append(f"- **{g.label}** (cohesion {g.cohesion:.2f}): {members}")
        lines.append("")
    if a.pairs:
        lines += ["### Convergence", "", "Tag pairs that co-occur on several top-quartile apps:", ""]
        lines += [f"- {x} + {y}: {n} apps" for x, y, n in a.pairs[:10]] + [""]

    lines += [
        "## Innovation hypotheses",
        "",
        "_Graded assumed: these are rule-based patterns in the direction table, worth research "
        "time, not established gaps._",
        "",
    ]
    if a.opportunities:
        for o in a.opportunities:
            lead = f" Leaders: {', '.join(o.evidence)}." if o.evidence else ""
            seen = ""
            if o.verdict:
                v = o.verdict
                seen = (f" **Researched {v['observed_on']}: {v['verdict']}** "
                        f"([source]({v['source_url']})). {v['note']}".rstrip())
            lines.append(f"- **{o.kind}: {o.direction}**: {o.rationale}.{lead}{seen}")
    else:
        lines.append("_None triggered today._")
    lines.append("")

    lines += ["## How well is the radar doing?", "", "### Forecast scorecard", ""]
    if a.evaluations:
        lines += [
            "Each day's ranking is a forecast. Once its horizon passes it is scored against the "
            "growth that actually followed, measured with the fixed default parameters.",
            "",
            "| Made on | Horizon | Apps compared | Spearman | Precision@k |",
            "| --- | --- | --- | --- | --- |",
        ]
        for e in a.evaluations[-10:]:
            rho = f"{e.spearman:.2f}" if e.spearman is not None else "–"
            prec = f"{e.precision_at_k:.0%} (k={e.k})" if e.precision_at_k is not None else "–"
            lines.append(f"| {e.made_on} | {e.horizon_days}d | {e.apps} | {rho} | {prec} |")
    else:
        lines.append(
            "_No forecast has matured yet. The first scorecard appears once a forecast is "
            "older than its horizon and the same apps have been observed again._"
        )
    lines += ["", "### Self-tuning", ""]
    if a.tuning is not None:
        verdict = "**Adopted**" if a.tuning.adopted else "No change"
        lines.append(f"{verdict}: {a.tuning.reason}.")
        t = a.tuning
        if t.candidates_tried and t.baseline is not None:
            lines += [
                "",
                f"Every one-step candidate, backtested over the same {t.days} matured days. A candidate "
                f"must raise mean rank correlation by at least {MIN_IMPROVEMENT:+.2f} and win on at least "
                f"{MIN_WIN_SHARE:.0%} of days.",
                "",
                "| Candidate | Mean Spearman | Change | Days better |",
                "| --- | --- | --- | --- |",
                f"| current settings | {t.baseline:.3f} | – | – |",
            ]
            for c in sorted(t.candidates_tried, key=lambda c: -c["spearman"]):
                lines.append(
                    f"| {c['knob']} → {c['value']:g} | {c['spearman']:.3f} | "
                    f"{c['spearman'] - t.baseline:+.3f} | {c['win_share']:.0%} |"
                )
    else:
        lines.append("_Tuning was not run._")
    lines += ["", "| Parameter | Value | Basis |", "| --- | --- | --- |"]
    for name in KNOBS:
        lines.append(f"| {name} | {a.params[name]:g} | {a.params.provenance[name]} |")
    if a.watch_added:
        lines += ["", f"Watch list expanded with: {', '.join(a.watch_added)}."]
    lines.append("")

    lines += ["## Source health", "", "| Collector | Last OK | Consecutive failures | Last count | Last error |",
              "| --- | --- | --- | --- | --- |"]
    for h in a.health:
        err = (h.last_errors[0][:120] + "…") if h.last_errors else ""
        flag = " ⚠" if h.flagged else ""
        lines.append(f"| {h.collector}{flag} | {h.last_ok or 'never'} | {h.consecutive_failures} | "
                     f"{h.last_count} | {err} |")
    lines.append("")

    measured = sum(
        1 for score in a.scores for s in score.signals
        if s.kind == "delta" and s.method.value != "reported"
    )
    if a.scores and not measured:
        lines += [
            "## Caveat for today",
            "",
            "No growth here was measured between two of the radar's own snapshots yet. Every signal "
            "is a gain stated by a page, a rate a company reported, or the difference between two "
            "company-reported figures. Measured deltas start from the next run that re-observes "
            "these apps.",
            "",
        ]
    return "\n".join(lines)
