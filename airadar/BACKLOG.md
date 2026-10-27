# Radar backlog

The daily agent picks one item a day (see `DAILY_AGENT.md`, step 6), or a
problem it found itself. Keep *Next* ordered by expected value to the report.

## Next

- **Stale stated gains outrank fresh measurements.** On 2026-10-23
  ai-job-search ranked #2 on a one-day gain stated on Oct 19, while the radar
  measured it slowing from ~2.9k to ~0.6k a day. Let the scorecard and tuner
  (`weight.delta` vs `weight.period`) decide first; if they can't, give
  stated period signals the same age decay reported figures already have.

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
- 2026-10-22: gone apps leave the ranking. YuJunZhiXue/dsh-purge returned 404
  on 2026-10-21 yet ranked #3 on older signals. `airadar mark-gone` appends an
  availability record (url checked, date, evidence) to `availability.jsonl`; an
  app whose latest record is "gone" is not scored and is listed under "Removed
  from the ranking", and `--back` restores it. Observations are untouched.
- 2026-10-23: moved apps keep one history. MadsLorentzen/ai-job-search was
  transferred to ai-job-search-dev/ai-job-search, and apps are keyed by URL, so
  a sighting under the new owner would have started a second app with no
  history. `airadar mark-moved` appends an alias (old id -> new id and url, with
  the page that shows it) to `aliases.jsonl`. Observations, apps and gone
  records resolve through it when read; `observations.jsonl` is never rewritten.
- 2026-10-24: hypotheses carry their research verdict. The rules re-proposed
  agent-memory, agent-orchestration, generative-ui and llm-gateway every day
  after research had disproved them. `airadar verdict` appends what research
  found (disproved / partly disproved / holds, with the analysis it rests on) to
  `verdicts.jsonl`; the report and the page's hypotheses table show it beside
  the rule. Nine verdicts from 2026-10-20 to 2026-10-23 recorded. The rules
  themselves are unchanged: a verdict annotates, it does not suppress.
- 2026-10-25: the report shows measured movers. From 2026-10-20 the daily
  analysis computed the day-on-day star changes between the radar's own
  readings by hand, because the report only showed each app's strongest signal,
  often a gain a page stated days earlier. `growth.measured_movers` now lists
  every app read on the latest run and on one up to 3 days before, largest
  change per day first, skipping changes within rounding; the report renders
  the top 15 under "Measured since the last reading".
- 2026-10-26: the scorecard follows moves. Forecasts store app ids as they
  were on the day they were made; ai-job-search moved on 2026-10-22, so the
  first matured forecast (2026-10-19) named it by an id no realised growth was
  recorded under, and it would have silently dropped out of the first
  scorecard. `scorecard` now resolves each forecast id through `aliases.jsonl`.
  The yardstick (`realised_scores`, default parameters) is unchanged.
- 2026-10-27: the shared page shows measured movers. The report has listed the
  growth measured between the radar's own readings since 2026-10-25; the page
  showed only the ranking, which leans on stated gains. The export now writes a
  `movers` dataset (top 25) and the page has a "Measured since the last
  reading" table beside the ranking.
