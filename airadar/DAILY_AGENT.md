# Daily agent runbook

This is the procedure a scheduled Claude Code session follows once a day. The
Python radar does the measuring, scoring and self-tuning, deterministically. The
agent does three things the Python cannot: it reads the web with tools a sandboxed
process may not have, it writes analysis in prose, and it improves the code.

Work through the steps in order. Every number you write anywhere must carry a
source URL. If a step fails, record the failure and keep going: a partial day
with its gaps stated is worth more than a skipped day.

## 0. Set up

```bash
cd AI-agents   # the repo clone
git fetch origin ai-radar main
```

Work on the `ai-radar` branch. If its pull request has been merged, follow the
session's git instructions: restart `ai-radar` from the latest `main` and open a
fresh draft PR for today's work.

```bash
pip install numpy pyyaml requests pytest
python -m pytest tests -q          # must pass before you change anything
TODAY=$(date -u +%F)
```

## 1. Live collectors

```bash
python -m airadar.cli collect
```

Where outbound network is blocked these fail into the run log. That is expected,
and the report shows it under source health. Do not work around a blocked host.

## 2. Collect through the inbox

Read `airadar/INBOX.md` first; it is the contract. Write one file per source to
`data/radar/inbox/$TODAY-<source>.json`. Delegate each source to a subagent so the
raw pages stay out of your context. Pass the subagent the INBOX.md rules verbatim.

1. **Re-observe** (most important). Run `python -m airadar.cli recheck` and
   record the current `stars`/`likes` for every app it lists, from each app's
   own page. A second observation of the same app is what gives the radar
   *measured* growth, which outweighs everything else it collects.
2. **GitHub trending**: daily, weekly and monthly; also the Python and TypeScript
   weekly lists. Keep AI-related repos only. Record `stars` and `stars_gained`.
3. **Reported growth**: web research for company-reported users, ARR, traffic
   and growth rates of AI applications, dated within the last 90 days. Also
   write a `.notes.md` file with the qualitative trends seen, each with a URL.
4. **Any source the source-health table flags** (three or more consecutive
   failures): try once through a different route (another page, a mirror, a
   different tool). If it still fails, say so in the analysis and stop.

Then:

```bash
python -m airadar.cli validate data/radar/inbox/*.json
```

Fix any file that fails, at the source of the error. Never "fix" a file by
deleting the observations that fail validation without saying so.

## 3. Run the radar

```bash
python -m airadar.cli daily
```

This ingests the inbox, scores, finds directions, records today's forecast,
scores matured forecasts, runs one bounded tuning step and writes
`data/radar/reports/$TODAY.md`. Read that report in full.

## 4. Verify anomalies

For every app in the report's top 25 with a flag, open its page and check whether
the number is real: is it a launch-week spike, a page-read error, or star farming
(many new accounts, star count far ahead of forks, issues and contributors)?
Record your verdict, with evidence, in the analysis. Do not edit or delete
observations; the record of what the radar was told stays intact.

## 5. Write the analysis

Write `data/radar/reports/$TODAY-analysis.md`. Readers are people deciding what
to build or back, so be specific and cite everything. Sections:

1. **Summary**: 5 bullets. What is growing fastest today, and the one thing
   that changed most since yesterday.
2. **Fastest-growing apps**: the top 10, each with *why* it is growing (launch,
   integration, viral post, pricing change) found by research, with URLs.
   Separate measured growth from claims.
3. **Directions**: which are accelerating or cooling and why; anything the
   emergent groupings suggest that the tags miss.
4. **Innovation opportunities**: take the report's hypotheses and research each:
   who is already there, what the gap actually is, and what would have to be true
   for it to be worth building. Drop any hypothesis that research disproves, and
   say so. Add at most two of your own, labelled as yours.
5. **Radar self-critique**: where the data was thin or biased today (one source
   dominating, a category the collectors cannot see), what the scorecard says
   about the radar's forecasts, and what single change would most improve
   tomorrow's report.

## 6. Improve the system (one change)

Read the scorecard, the source-health table, today's self-critique and
`airadar/BACKLOG.md`. Pick **one** improvement and make it: a new collector or
inbox source, a fix to a recurring data problem, better tagging, a clearer
report section, a new test. Write a test that pins the improvement, and run the
full suite. If tests fail and you cannot fix them, revert the change and note
it in the backlog.

Record the change in `airadar/BACKLOG.md` under *Done* with the date and why.
Add newly noticed problems under *Next*.

**Guardrails.** These rules are not yours to change; a change that touches them
is out of scope however good it looks:

- `realised_scores` uses the default parameters. The radar must never be graded
  by a yardstick it can tune.
- The tuner's acceptance thresholds (`MIN_EVAL_DAYS`, `MIN_IMPROVEMENT`,
  `MIN_WIN_SHARE`) and knob bounds only ever get stricter.
- Observations are append-only. Nothing in `observations.jsonl` is edited or
  deleted.
- The inbox loader stays strict: no defaulting of missing fields, no accepting
  numbers without a source URL.
- No new dependency that needs an API key in order to run tests.

## 7. Commit and push

```bash
git add -A data/radar airadar tests
git commit -m "Radar $TODAY: <one line on the biggest finding> + <the improvement>"
git push -u origin ai-radar
```

Make sure a draft PR exists for `ai-radar`. End the session with a summary: the
top 3 findings, the improvement made, and anything a person needs to decide.
