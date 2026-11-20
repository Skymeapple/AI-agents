# Lab learnings

This file holds what we've learned about what drives downloads. Every daily mission reads it first and appends to it at the end. Keep entries short and evidence-based, and cite metric deltas from `DASHBOARD.md`.

## Confirmed by data

* **L1. Check the market, not names.** Day 1's gap check (two exact npm names) missed a 39,400/wk superset with the *same name* and five scoped clones. The gap gate must search the category and read download counts (`scripts/npm_market.py`). Source: `lab/research/2026-11-20-market-review.md`.
* **L2. AI-dev-tool niches saturate fast.** Every niche checked on 2026-11-20 had 5–10 entrants. Most new entrants sit at 30–450 weekly downloads, and the leaders launched 7–10 months earlier.
* **L3. Being in the right place beats features.** Tools that run automatically lead the field: `block-no-verify` (agent hook, 55,877/wk) and `@yawlabs/ctxlint` (documented as `npx …@latest`, 39,400/wk). Feature-similar manual CLIs get 31–443/wk. This is being tested as experiment E2.

## Hypotheses (unproven)

* H1 (*superseded by L3*): Developer CLIs that run with `npx` and no install or config get the most trial. Zero dependencies lowers the trust barrier.
* H2: Riding a current wave (AI coding agents, Nov 2026) beats evergreen problems for early downloads.
* H3: A ready-made GitHub Action or pre-commit snippet turns one-off trials into recurring CI downloads, and recurring downloads compound.
* H4: README "proof" (findings in famous repos) increases conversion from page view to install.

## Evidence log

| Date | Observation | Metric | Action taken |
|---|---|---|---|
| 2026-11-20 | ctxlint would enter a saturated category below a same-name 39k/wk leader | competitors: 31–42,530/wk | Parked ctxlint (not published); protocol v2; experiments E1–E3 |
