# Lab learnings

This file holds what we've learned about what drives downloads. Every daily mission reads it first and appends to it at the end. Keep entries short and evidence-based, and cite metric deltas from `DASHBOARD.md`.

## Confirmed by data

* **L1. Check the market, not names.** Day 1's gap check (two exact npm names) missed a 39,400/wk superset with the *same name* and five scoped clones. The gap gate must search the category and read download counts (`scripts/npm_market.py`). Source: `lab/research/2026-11-20-market-review.md`.
* **L2. AI-dev-tool niches saturate fast.** Every niche checked on 2026-11-20 had 5–10 entrants. Most new entrants sit at 30–450 weekly downloads, and the leaders launched 7–10 months earlier.
* **L3. Being in the right place beats features.** Tools that run automatically lead the field: `block-no-verify` (agent hook, 55,877/wk) and `@yawlabs/ctxlint` (documented as `npx …@latest`, 39,400/wk). Feature-similar manual CLIs get 31–443/wk. This is being tested as experiment E2.

* **L4. Ignore launch week. The day-30 rate predicts the winners.** Launch curves, collected 2026-11-20 from npm's daily history: nearly every package gets 300–2,000 downloads in its first 7 days, mostly mirrors, scanners and bots, then collapses. Today's category leaders were all still at **324–538/wk** by day 30 (`@yawlabs/ctxlint` 538, `agnix` 492, `block-no-verify` 417, `claude-code-lint` 324). Packages that stayed small had dropped to **12–64/wk** (`unrot`: 2,078 in week 1, then 34/wk at day 30). Judge projects by the day-30 rate in `DASHBOARD.md`, never by first-week totals.

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
