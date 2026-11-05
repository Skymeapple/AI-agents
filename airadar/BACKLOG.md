# Radar backlog

The daily agent picks one item a day (see `DAILY_AGENT.md`, step 6), or a
problem it found itself. Keep *Next* ordered by expected value to the report.

## Next

- **Stale stated gains outrank fresh measurements.** On 2026-10-23
  ai-job-search ranked #2 on a one-day gain stated on Oct 19, while the radar
  measured it slowing from ~2.9k to ~0.6k a day. Let the scorecard and tuner
  (`weight.delta` vs `weight.period`) decide first; if they can't, give
  stated period signals the same age decay reported figures already have.
  Tested 2026-10-30, not adopted: dropping a stated gain from the forecast
  when the app has a newer measured delta of the same family changed the
  4-day backtest mean Spearman from 0.656 to 0.655 (Oct 20 +0.019, Oct 21
  -0.002, Oct 22 -0.021). Revisit once more forecasts have matured.

- **Re-observation pages are rounded.** Solved while the GitHub API is
  reachable (2026-11-01, `api:github-repos`); the page path below remains the
  fallback when it is not. GitHub repo pages read through WebFetch
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
- **Scope by rule for the API search.** Out-of-scope marking is manual. A
  cheap automatic first pass (an AI-terms check on description and topics,
  flagged for review rather than applied) would keep the API's daily intake
  clean; the 2026-11-03 review found the keyword check misses about 1 in 9
  AI repos, so it can only suggest, not decide.
- **The API search adds ~1,000 repos a day to the registry.** On 2026-11-01
  the topic search added about 970 apps. Most are slow-moving; watch whether
  the ranking and the page stay readable, and consider a minimum-stars or
  minimum-gain bar for what the search registers.
- **Hugging Face and Hacker News are unreachable from the cloud sandbox** (DNS
  failure through WebFetch, 403 from the proxy). Allowing `huggingface.co` and
  `hn.algolia.com` in the environment's network settings, or running
  `python -m airadar.cli collect` from GitHub Actions, would restore them.
- **Product Hunt**: needs an API token (`PRODUCTHUNT_TOKEN`); add a collector
  that is skipped cleanly when the token is absent.
- **Tag vocabulary drift.** `DIRECTION_TAGS` is now fixed and API topics map in
  through `TAG_ALIASES` (2026-11-02). Review the most common unmapped topics
  weekly; add an alias where one names an existing direction, and a new
  direction only once its emergent grouping has held for several days.
- **Embeddings for emergent groupings**, once a model is reachable from the
  Python process. Lexical clustering over one-line descriptions is weak.
- **Star-farming check.** Repos flagged as anomalous could be checked against
  forks, issues and contributor counts automatically, not only by the agent.
  Recheck notes now carry forks and watchers as text; recording them as
  metrics that don't feed the growth score would make this automatic.

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
- 2026-10-28: the report shows the tuner's evidence. On the first day three
  forecasts matured, the tuner can move a parameter for the first time; until
  now the report gave only a one-line verdict, though every candidate's
  backtest was already stored in `iterations.jsonl`. The Self-tuning section now
  lists the current settings' mean rank correlation and every one-step
  candidate with its change and share of days won, against the fixed bars
  (`MIN_IMPROVEMENT`, `MIN_WIN_SHARE`, unchanged).
- 2026-10-29: levels are compared only within one basis. `users` mixed WAU,
  MAU, registered and web-unique counts, and revenue mixed gross and net
  run-rates, so the radar read ChatGPT's $70B gross and $50B net run-rates a
  day apart as -$20B. Observations take an optional `basis` from a fixed
  vocabulary per metric (users: dau/wau/mau/registered/paid_seats/web_uniques;
  revenue: gross/net/contracted); level deltas are computed within one basis
  only. Observations recorded before today carry no basis and form their own
  group, so old artefacts remain until those apps are re-reported with one.
- 2026-10-29 (recommendation, not a change; made by the orchestrating
  session, not by a person, who has not weighed in): the `weight.delta` and
  `weight.period` caps stay at 1.0. Lowering `weight.period` achieves the same relative effect
  within the current bounds, so no guardrail is loosened.
- 2026-10-30: measured movers show the pace before. The analysis kept
  describing movers as "easing" or "accelerating" from memory of earlier
  reports. Each mover now carries the per-day pace between the two readings
  before the latest pair (none when there is no earlier pair, or when that
  change is within the readings' rounding), and the report's table shows it as
  "Per day before".
- 2026-10-31: anomaly reviews. The same launch-week repos (thinkrail, cindy,
  openbot) tripped the "most stars gained this week" flag every day, and each
  day's run had to re-establish that they had already been checked. A check
  is now recorded (`airadar review <url> organic|inorganic|unclear <checked
  url>`, append-only `reviews.jsonl`, following repo moves), and the report and
  shared page show the latest review beside the flag. The flag and its scoring
  penalty are unchanged: a review explains a flag, it doesn't clear it.
- 2026-11-01: re-observation by API. The GitHub API became reachable today (a
  token is set in the environment), and `recheck` still listed 594 apps,
  almost all of them repos the API search had just read. `recheck` now leaves
  out any app already read that day, and `collect` re-reads the remaining
  repos through the search endpoint (`repo:owner/name` qualifiers, batched
  under GitHub's 256-character query cap) as collector `api:github-repos`.
  That turns rounded page reads ("58.7k") into exact counts. Per-repo REST
  reads (`/repos/owner/name`) are refused by the environment's proxy (403);
  search is the endpoint the topic collector already uses.
- 2026-11-01 (follow-on, same cause): generic words stopped becoming search
  terms. The watch list's automatic terms had picked up "engine", "without",
  "working", "unified" and "ad-free" from emergent groupings. Harmless while
  the API was blocked; once it answered, each pulled in 150 repos, 168 of
  them seen only through those terms (Godot, Redis, Elasticsearch). They are
  now generic terms: never mined, and pruned from the watch list. The repos
  already registered stay (observations are append-only).
- 2026-11-02: directions come from a tag vocabulary. API repos bring raw GitHub
  topics as tags ("ai", "cli", "mcdonalds"), and each topic on two scored repos
  became a direction: 236 directions and 83 rule hypotheses on 2026-11-01,
  against 32 and 3 the day before. Directions and convergence pairs now count
  only `DIRECTION_TAGS`, the vocabulary the collectors tag with. Common topics
  that name a real direction join through `TAG_ALIASES` (claude-code and codex
  -> coding-agent, text-to-speech -> voice, ...). Rerun on 2026-11-01's data:
  34 directions, 8 hypotheses. Emergent groupings still find what the
  vocabulary lacks.
- 2026-11-03: apps can be marked out of scope. The API search sweeps in repos
  that are not AI apps (an Android TV debloater ranked #2 on 2026-11-02).
  `airadar mark-out-of-scope <url> <page> "<what it is>"` (append-only
  `scope.jsonl`, `--back` to undo, following repo moves) leaves an app out of
  the ranking, directions, movers and recheck list while its observations stay;
  the report lists them. First use: 50 apps, the 49 that came in only through
  the generic search terms and whose descriptions show no AI (each read; 6
  AI-related ones in that group were kept), plus cleanyourtv.
- 2026-11-04: the API re-read suggests where a missing repo moved. Two repos
  transferred in two days (phone-harness on 11-03, dsh-purge on 11-04); each
  showed up only as "not returned" and needed a page read to find its new
  owner. github.com pages are refused to Python by the proxy, so redirects
  can't be followed. A transferred repo keeps its name and stars, so the
  re-read now searches the name and reports any same-named repo within a
  quarter of the last star reading as a possible move. It is a suggestion
  for `mark-moved`, never applied. Live check: it names HunYiSage/dsh-purge and
  phone-harness/phone-harness for the two real moves.
- 2026-11-05: `computer-use` joins the direction vocabulary. Agents that operate
  a desktop, browser or phone led new growth for three days running
  (vercel-labs/agent-desktop, phone-harness, Mika, dsh-Go), which is the bar the
  2026-11-02 rule set for a new direction. GitHub topics computer-use-agent,
  computer-agent, agent-computer, browser-use and browser-use-agent map to it.
  Rerun on 2026-11-04's data: 13 scored apps, led by agent-desktop.
