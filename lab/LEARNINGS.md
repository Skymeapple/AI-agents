# Lab learnings

This file holds what we've learned about what drives downloads. Every daily mission reads it first and appends to it at the end. Keep entries short and evidence-based, and cite metric deltas from `DASHBOARD.md`.

## Confirmed by data

* **L1. Check the market, not names.** Day 1's gap check (two exact npm names) missed a 39,400/wk superset with the *same name* and five scoped clones. The gap gate must search the category and read download counts (`scripts/npm_market.py`). Source: `lab/research/2026-11-20-market-review.md`.
* **L2. AI-dev-tool niches saturate fast.** Every niche checked on 2026-11-20 had 5–10 entrants. Most new entrants sit at 30–450 weekly downloads, and the leaders launched 7–10 months earlier.
* **L3. Being in the right place beats features.** Tools that run automatically lead the field: `block-no-verify` (agent hook, 55,877/wk) and `@yawlabs/ctxlint` (documented as `npx …@latest`, 39,400/wk). Feature-similar manual CLIs get 31–443/wk. This is being tested as experiment E2.

* **L4. Ignore launch week. The day-30 rate predicts the winners.** Launch curves, collected 2026-11-20 from npm's daily history: nearly every package gets 300–2,000 downloads in its first 7 days, mostly mirrors, scanners and bots, then collapses. Today's category leaders were all still at **324–538/wk** by day 30 (`@yawlabs/ctxlint` 538, `agnix` 492, `block-no-verify` 417, `claude-code-lint` 324). Packages that stayed small had dropped to **12–64/wk** (`unrot`: 2,078 in week 1, then 34/wk at day 30). Judge projects by the day-30 rate in `DASHBOARD.md`, never by first-week totals.

* **L5. Check names across every ecosystem.** The Day 2 tool was named `agentscrub` after an npm-only check, but PyPI already had `agentscrub` (44 releases since May 2026) doing agent-log redaction. Publishing would have imitated it. The market check is now cross-ecosystem (`npm_market.py --name` checks PyPI, and the protocol requires a GitHub search). Being blocked from publishing turned out to be the lucky break that made the rename free.

## Hypotheses (unproven)

* H1 (*superseded by L3*): Developer CLIs that run with `npx` and no install or config get the most trial. Zero dependencies lowers the trust barrier.
* H2: Riding a current wave (AI coding agents, Nov 2026) beats evergreen problems for early downloads.
* H3: A ready-made GitHub Action or pre-commit snippet turns one-off trials into recurring CI downloads, and recurring downloads compound.
* H4: README "proof" (findings in famous repos) increases conversion from page view to install.

## Evidence log

| Date | Observation | Metric | Action taken |
|---|---|---|---|
| 2026-11-20 | ctxlint would enter a saturated category below a same-name 39k/wk leader | competitors: 31–42,530/wk | Parked ctxlint (not published); protocol v2; experiments E1–E3 |
| 2026-11-20 | Launch curves of 16 competitors: week-1 downloads are noise; the day-30 rate separates leaders (≥ 324/wk) from the tail (≤ 64/wk) | launch_curves.json | Protocol Phase 0 now judges projects by day-30 rate |
| 2026-11-20 | First gated mission: 1 of 5 candidates passed the market gate. The winner was a 2-week-old problem (Apiiro, 5 Nov) with only one 12-day-old entrant at 358/wk. Fresh incidents leave gaps; established pain points were all taken. | gate: 4 FAIL / 1 PASS | Shipped agentscrub (gated, hook + pre-commit + action) |
| 2026-11-21 | 2nd gated mission: 1/5 passed again, and again it was a days-old incident (agent-framework PyPI compromise, 17 Nov). Real-environment validation caught a false positive that would have hit every coverage user. Unit tests alone would have shipped it. | gate 4 FAIL / 1 PASS; 1 FP fixed | Shipped pthscan (gated, action + hook). Publishing still blocked: neither NPM_TOKEN nor PYPI_TOKEN works |
| 2026-11-23 | Publishing blocked for 4 days (npm E403, no PyPI token): 3 projects built, 0 installable, 0 downloads. The lab's bottleneck is distribution, not ideas. Added a 'distribution blocked' mode. Tavily hit its plan limit (HTTP 432), so research also depends on outside quota. The 2026-11-22 trigger fired while the session was idle and was folded into today's run. | 0/3 published | Improve mode: agentscrub 0.2.0 (--home); protocol rule |
| 2026-11-24 | Publishing blocked for 5 days. Found a same-name, same-purpose PyPI project for our unpublished `agentscrub` | name collision | Renamed to agent-commit-guard 0.3.0 and repositioned on the git guard; name check now covers PyPI |
| 2026-11-25 | Publishing blocked for 6 days; Tavily still over quota. An improvement grounded in primary docs (Python's site module), not web research, was still possible, and real-system validation again set the severity rules (Debian apport = known-good; the sandbox's own sitecustomize = warning, not critical). | 0/2 published | pthscan 0.2.0 (sitecustomize/usercustomize) |
| 2026-11-26 | Day 7 blocked. The verified minimum versions (Py 3.9, Node 18) had never been tested; now confirmed. With nothing to research or publish, a short maintenance day beats invented work. | 0/2 published | Compatibility verified; no-ship day |
