# Radar backlog

The daily agent picks one item a day (see `DAILY_AGENT.md`, step 6), or a
problem it found itself. Keep *Next* ordered by expected value to the report.

## Next

- **Verify reported figures at source.** Day one's commercial figures came from
  search-result summaries because the fetch tool could not open the pages.
  Re-collect them from the cited URLs when fetching works, and record a
  disagreement as a new observation rather than editing the old one.
- **Measured deltas for products.** Repos and Spaces can be re-observed from
  their own pages; commercial apps cannot. Find a repeatable, citable per-app
  number (app-store ranking pages, public traffic estimates, Chrome Web Store
  user counts) and record it daily.
- **Hugging Face and Hacker News are unreachable from the cloud sandbox** (DNS
  failure through WebFetch, 403 from the proxy). Allowing `huggingface.co` and
  `hn.algolia.com` in the environment's network settings, or running
  `python -m airadar.cli collect` from GitHub Actions, would restore them.
- **Product Hunt**: needs an API token (`PRODUCTHUNT_TOKEN`); add a collector
  that is skipped cleanly when the token is absent.
- **Tag vocabulary drift.** Collectors invent tags freely. Grow `TAG_ALIASES` in
  `directions.py` from what actually appears in `apps.json`, so one direction is
  not split across synonyms.
- **Embeddings for emergent groupings**, once a model is reachable from the
  Python process. Lexical clustering over one-line descriptions is weak.
- **Star-farming check.** Repos flagged as anomalous could be checked against
  forks, issues and contributor counts automatically, not only by the agent.

## Done

- 2026-10-19: initial system: inbox contract, growth model, directions,
  hypotheses, forecast scorecard, backtest tuner, source health, daily runbook.
- 2026-10-19: first real run exposed three faults, now fixed. (1) Reported
  figures dated months back fell outside the 7-day window, so every commercial
  app was baseline-only; they now get a 120-day window with a 45-day staleness
  half-life. (2) Point-in-time filtering used the date a figure describes, not
  the date it was collected, which let backtests see late-reported figures early;
  observations now carry `collected_on`. (3) Products on multi-product company
  domains (`openai.com/codex`) collapsed into one identity.
