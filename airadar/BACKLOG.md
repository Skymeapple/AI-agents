# Radar backlog

The daily agent picks one item a day (see `DAILY_AGENT.md`, step 6), or a
problem it found itself. Keep *Next* ordered by expected value to the report.

## Next

- **Deleted repos keep ranking.** YuJunZhiXue/dsh-purge returned 404 on
  2026-10-21 but ranked #3 on signals from before. Record a "gone" observation
  (e.g. a recheck that finds a 404) and drop or flag gone apps in the report.
- **Hypotheses forget research.** The rule engine re-proposes directions the
  analysis already disproved (agent-memory, agent-orchestration) because the
  commercial incumbents (Mem0, Letta, LangChain, CrewAI) are not tracked apps.
  Track the incumbents the research names, or keep a verdict list that the
  hypothesis rules consult.

- **One metric, mixed definitions.** `users` mixes WAU, MAU and web unique
  visitors, and `revenue_annualized_usd` mixes gross and net-of-partner run-rates
  (OpenAI: $70B per Axios on 2026-09-29, $50B per the FT on 2026-09-30). Deltas
  between them are artefacts (ChatGPT −$20B in a day; Claude 245M MAU -> 111M
  web uniques). Give observations a definition qualifier (e.g. `basis: wau|mau|
  web_uniques`, `gross|net`) and only diff like with like.
- **Re-observation pages are rounded.** GitHub repo pages read through WebFetch
  show "58.7k", never the exact count. Repos on a trending page get exact counts
  the same day; the rest are only resolvable to 100 stars. Find a page that shows
  exact counts (e.g. the stargazers page) for the recheck list.
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
- 2026-10-20: rounding-aware level readings. Re-observed GitHub pages show
  rounded counts ("58.7k"), and the same-day tie-break preferred the larger
  value, so a rounded-up figure beat the exact count from a trending page and
  invented stars. Observations now carry an optional `rounded_to`; the most
  precise reading wins a same-day tie, and a delta no larger than the two
  readings' rounding is not counted as growth. Tests pin all three.
- 2026-10-21: rate changes need a week between readings. A `stars_gained`
  over 1 day seen on two consecutive runs counted as a rate change, so day-to-day
  wobble (Kronos 355 -> 363 stars/day) read as +16.9%/wk and was the #8 app's
  strongest signal. Two readings of a period gain must now sit at least
  `min_rate_change_span_days` (7) apart, however short the period. A fixed
  extraction rule, not a tuner knob.
