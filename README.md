# sciscout

A platform for tracking research directions across the basic and applied
sciences, scoring how much they matter and how quickly, estimating what it would
take to commercialise them, and modelling the investment case quantitatively.

It is built around one rule: **every number carries where it came from.** A
figure is `observed` (computed from ingested data), `sourced` (from a named
reference) or `assumed` (a judgement call with a stated rationale). That
distinction survives all the way through to the final ROI distribution, which
reports what share of its spread traces back to assumption rather than evidence.

This matters because the failure mode of a platform like this is not being
wrong — it is being confidently wrong in a way nobody can audit. A number that
looks quantitative carries authority whether or not it deserves any, and the
seed priors shipped here mostly do not deserve it yet. The provenance machinery
exists to keep that visible instead of letting it quietly disappear behind a
percentage.

## What it does

```
harvest  ->  cluster into      ->  score        ->  infer      ->  build         ->  simulate
(arXiv,      research               importance      technology     commercial        investment
 OpenAlex)   directions             and urgency     readiness      profile           scenarios
```

**Importance and urgency are scored separately**, then combined geometrically
into a priority. They are kept apart because they routinely disagree, and the
disagreement is the useful part: a foundational result in a slow field is
important and not urgent; a crowded race for a modest prize is urgent and not
important. Collapsing them hides exactly what a prioritisation decision turns on.

| Axis | Sub-scores |
| --- | --- |
| Importance | field-normalised citation impact, publication momentum, cross-disciplinary breadth, evidence strength, foundationality |
| Urgency | competitive density, acceleration against the track's own history, industry entry |

**Citation impact is normalised against the corpus itself** — the median
citations of works in the same discipline and age bucket — rather than against
an invented lookup table. That makes the normalisation reproducible from data
you hold, and it adapts as the corpus grows. When a bucket is too thin to trust,
the baseline says so rather than pretending.

**Technology readiness is a distribution, not a number.** Reading TRL off
abstracts is a weak signal. The inference returns a probability distribution over
TRL 1-9, and the Monte Carlo consumes the whole thing, so that uncertainty
propagates into the ROI spread instead of being discarded at the first step. An
analyst who knows the field can override it.

**The investment model simulates staged abandonment.** Research programmes are a
sequence of gates, and a programme that dies at the first gate never spends the
money budgeted for the fifth. Modelling this correctly is the difference between
a usable number and a badly wrong one — multiplying a full development budget by
a cumulative success probability overstates expected cost by a large factor. That
abandonment option is also most of why early-stage research investment can be
rational despite single-digit success rates.

## Two things this model gets right that are easy to get wrong

**Dilution is capital-weighted, not a fixed percentage.** Each stage is funded by
a round raising that stage's cost, at a valuation stepping up from the last. A
programme needing $700M of development capital dilutes an early investor to
near-nothing however well the science goes. A fixed-percentage dilution model
will happily report a 90x return on a $5M seed cheque into a company that
consumed three quarters of a billion dollars — which is not a return, it is an
accounting error. For capital-intensive hardware this effect, not technical risk,
is often what actually kills early-stage returns.

**The discount rate is a cost of capital, not a venture hurdle rate.** A 25-30%
VC target return is priced to compensate for programme failure. This model
already simulates that failure explicitly, draw by draw. Discounting the
survivors at a venture rate on top charges for the same risk twice, and over a
decade-long development path the double charge is enough to turn genuinely
attractive programmes negative. The default is 12%.

## Finding directions without labels

Watching science broadly rules out hand-labelling research directions. Nobody is
going to label all of it, and the labels you would write are the ones you already
know to look for — which excludes exactly the emerging directions worth catching.

`discover` clusters works into candidate directions from their text alone, then
routes each to a commercialisation model:

```bash
python -m sciscout.cli discover --corpus data/demo_corpus.json --evaluate --rank
```

TF-IDF, cosine similarity, average-linkage agglomerative clustering with a
similarity floor. No embeddings and no network, so it runs anywhere the corpus
does, and it is deterministic — the same corpus gives the same directions.
Average linkage rather than single linkage because single linkage chains: one
paper bridging two topics would otherwise fuse them into one direction.

**Discovery is measured, not asserted.** `--evaluate` scores the clustering
against known labels where a corpus has them. On the shipped demo corpus — 1,174
works across 16 planted directions, including deliberately adjacent pairs like
quantum error correction beside topological qubits, and protein design beside
climate emulators — it recovers **16 of 16 at purity 1.00**.

**That number is weaker evidence than it looks**, and worth being plain about: the
demo corpus is synthetic and its vocabulary was written alongside the classifier's
keyword sets. It demonstrates the mechanism is sound and would catch a regression.
It does not tell you how the method performs on real abstracts, which are messier,
and where two groups pursuing one idea in different vocabularies will land in
separate clusters. Embeddings would do better; that is the upgrade path when the
environment can reach a model.

### Routing, and admitting when it fails

The sector decides every downstream number — which stages lie ahead, their cost,
their duration, which constraints bite. Getting it wrong doesn't give a
slightly-off answer, it gives an answer about a different kind of technology. So
routing is graded `assumed`, never `observed`, and two conditions must both hold
before a sector is trusted: it must take a third of the total score *and* beat the
runner-up by 1.5x. Below that it routes to `generic` and `coverage_report` names
it.

That refusal is the point. The tempting design picks the best-scoring sector
regardless, so every direction lands somewhere and the pipeline never visibly
fails. On a corpus spanning all of science most directions will not match any
modelled archetype, and a platform that hides that is lying about its coverage.

Two rules stop a lexical classifier from quietly misrouting whole fields:

- **Ambiguous words are excluded even though they look discriminative.** `model`
  ("mouse model", "manufacturing model"), `cell` (a battery cell and a biological
  one), `yield` (fab yield and crop yield), `strain` (bacterial and mechanical).
- **A discipline label with no supporting vocabulary is discounted.**
  "Electrical engineering" covers battery packs and lithography alike. Before this
  rule, one incidental discipline label scored as highly as a direction saturated
  in a sector's actual vocabulary, and sent batteries, protein design and quantum
  error correction to `generic`.

## Scenarios

A single simulation answers "what does this look like?". The question that drives
a decision is "what would have to be true for this to work, and how far off are
we?" — so scenarios are first-class. A scenario file supplies the assumptions the
platform cannot derive from a research corpus, and runs several cases side by
side against the same science.

**Every value in a scenario file must carry a `source` block.** This is enforced,
not encouraged. A config loader that quietly accepted bare numbers would undo the
provenance discipline the rest of the system rests on, because a config file is
exactly where an unfounded figure would enter wearing the same clothes as a
measured one. A misspelled field name or a stage that isn't on the programme's
remaining path is an error too, so you can't believe you've overridden something
you haven't.

```yaml
track: solid-state-sodium
scenarios:
  - name: Conservative
    description: The case to beat — if it works here, the thesis needs no optimism.
    market:
      tam_usd:
        median: 4.0e9
        spread: 2.5              # p95 / median
        source:
          grade: assumed
          detail: grid storage niche where sodium's cost advantage is decisive
    stages:
      Pilot and scale-up:
        cost_usd:
          low: 60.0e6
          mode: 140.0e6
          high: 400.0e6
          source:
            grade: sourced
            detail: three comparable pilot lines
            reference: <your source, by name>
```

```bash
python -m sciscout.cli scenarios --file examples/sodium-battery-scenarios.yaml
```

Scenarios use **common random numbers**: each input draws from its own stream,
keyed by a stable hash of its name. Change one assumption and only that column
moves — every other draw is bit-identical across scenarios. This is not a
nicety. numpy's beta sampler uses rejection sampling and consumes a variable
number of underlying values, so on a single shared stream two scenarios differing
only in a stage cost come back with different success rates: a pure sampling
artefact, indistinguishable from a real effect. A worked three-scenario
comparison is in
[`examples/sodium-battery-comparison.md`](examples/sodium-battery-comparison.md).

## Quick start

```bash
pip install numpy pyyaml requests pytest

# Generate a synthetic demo corpus (no network required)
python scripts/make_demo_corpus.py

# Find research directions with no labels, route and rank them
python -m sciscout.cli discover --corpus data/demo_corpus.json --evaluate --rank

# Rank research directions
python -m sciscout.cli rank --corpus data/demo_corpus.json

# Model one direction's investment case
python -m sciscout.cli model --corpus data/demo_corpus.json \
    --track solid-state-sodium --tam 12e9 --check 5e6 --ownership 0.15

# Full markdown report
python -m sciscout.cli report --corpus data/demo_corpus.json --out report.md

# Compare scenarios for one direction
python -m sciscout.cli scenarios --file examples/sodium-battery-scenarios.yaml

# What commercialisation priors exist, and how well sourced they are
python -m sciscout.cli sectors
```

With outbound network access, harvest real data instead:

```bash
python -m sciscout.cli harvest --source openalex \
    --query "solid state sodium battery electrolyte" \
    --track solid-state-sodium --sector energy_hardware \
    --out data/sodium.json --limit 500
```

A worked example produced entirely from the synthetic corpus is in
[`data/example_report.md`](data/example_report.md).

## Honest limitations

These are the things most likely to mislead you. None are hidden in the code.

1. **The shipped priors are mostly seed values, not findings.** Stage durations,
   costs and success rates in `sciscout/priors/sectors.yaml` are marked
   `assumed` where they are judgement calls and `sourced` where they come from a
   named reference. Even the sourced ones are flagged `needs_review` — the
   biopharma clinical transition rates are well-known headline figures, but a
   figure you have not checked against its original is one you are borrowing,
   not one you know. Any ROI number produced on the defaults will report itself
   as ~100% assumption-driven. That is the system working, not failing.

2. **The demo corpus is synthetic.** All 1,174 records in
   `data/demo_corpus.json` are machine-generated, with `synthetic:` identifiers and
   placeholder authors, spanning 16 planted directions across energy, biopharma,
   semiconductors, software and agriculture. It exercises the pipeline and gives
   discovery something measurable to be scored against. It tells you nothing about
   any real field.

3. **Clustering is lexical, and it scales only so far.** `discover` groups
   papers that use the same words, so one idea pursued in two vocabularies splits,
   and a shared methods vocabulary can pull unrelated work together. Dense
   clustering refuses above 12,000 works rather than dying on a multi-gigabyte
   allocation; `discover_by_discipline` partitions first, which scales further and
   avoids cross-field lexical collisions, at the cost of splitting genuinely
   cross-disciplinary directions. Neither failure is hidden — run both and compare.

4. **Foundationality is a lexical proxy.** It counts platform/method/enabling
   language in abstracts. Properly it is how much downstream work builds on a
   result, which needs citation-graph data the corpus does not carry. It is
   marked `assumed` for this reason.

5. **Stages are modelled as sequential and non-overlapping.** Real programmes
   overlap them, so timelines here are biased long.

6. **Market models are placeholders until you supply real ones.** `MarketModel`
   and `DealTerms` ship with clearly-labelled defaults so a track can be modelled
   end to end the moment it is ingested. They are scaffolding.

7. **Mean NPV can rank an early-stage programme above a late-stage one**, because
   failing at discovery is cheap and failing in Phase III is not. This is real,
   not a bug, and it is why the platform reports NPV and MOIC side by side rather
   than collapsing to one figure of merit. There is a test pinning this
   behaviour so nobody later "fixes" it into agreeing with intuition.

## Layout

```
sciscout/
  provenance.py          Grade / Provenance / Estimate / ProvenanceLedger
  models.py              Work and Track
  pipeline.py            corpus -> ranked investment cases
  discovery.py           unsupervised clustering into candidate directions
  classify.py            routing to a commercialisation model, and coverage gaps
  scenarios.py           named scenarios, provenance-enforced config, comparison
  report.py              markdown rendering
  cli.py                 discover / harvest / rank / model / report / scenarios / sectors
  sources/               arxiv, openalex, local JSON corpus
  scoring/               citation baseline, importance and urgency, TRL inference
  commercial/            TRL -> stages, costs, timelines, constraints
  invest/                distributions, investment case, Monte Carlo engine
  priors/sectors.yaml    commercialisation priors, graded by provenance
examples/                worked scenario file and its comparison output
scripts/                 synthetic corpus generator
tests/                   78 tests
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite asserts the modelling claims directly, not just that the code runs:
abandoned programmes accrue no downstream cost, capital-hungry programmes dilute
early investors, sensitivity analysis recovers a driver that was deliberately
planted, missing citation data is scored as absent rather than zero, a two-year
publication history is refused as a trend, a scenario file cannot introduce a
number without saying where it came from, and discovery recovers 16 of 16 planted
directions on a corpus built with adjacent topic pairs.


# airadar: the AI app radar

A second tool in this repo, built on the same provenance discipline. It collects
signals on AI applications every day, ranks the fastest-growing ones, groups
them into directions, generates innovation hypotheses, and **grades its own
forecasts and tunes itself on the result**.

```
collect             ->  score growth   ->  directions,     ->  forecast and    ->  tune one knob
(live APIs, inbox       (measured,         emergent           score matured       by backtest,
 from agent web         stated,            groupings,         forecasts           if the evidence
 research)              reported)          hypotheses                             clears the bar)
```

**Growth is derived, never stored.** Observations are append-only and dated;
every score is recomputed from them with an `asof` date that hides the future.
That is what makes backtesting honest: replaying a past day sees exactly what
was known that day.

**Three grades of growth evidence.** A *delta* between two of the radar's own
snapshots is measured. A *period* gain ("+4,774 stars this week") is stated by a
source. A *reported* growth rate ("ARR tripled") is a company's chosen framing.
Each carries a trust weight, and every number reaches the report labelled with
how it was obtained.

**Comparable across kinds by percentile.** Stars and revenue cannot be compared
directly. Relative growth (log-growth per day) is ranked across everything;
absolute velocity is ranked only within its own metric family. Thin evidence is
shrunk toward the median, so one page-read claim cannot top the table. A gain of
more than half an app's total in a week is flagged as possible extraction error
or star farming, and discounted rather than dropped.

**Self-iteration is measured, not asserted.** Every day's ranking is recorded as
a forecast. Once its 7-day horizon passes, it is scored against the growth that
followed: Spearman rank correlation and precision@k. The tuner nudges each
knob one step up and down, replays every matured day with each candidate, and
adopts the best only if it beats the current settings by 0.02 on mean Spearman,
across at least 3 days, winning on two-thirds of them. The yardstick is computed
with fixed default parameters, so the tuner can never move its own target. A
test plants a noisy source in a synthetic world and checks the tuner learns to
discount it, and that it holds still when nothing is wrong.

**Collection works around a sandbox.** Live collectors (GitHub search, Hugging
Face, Hacker News, npm, PyPI) run wherever outbound network is open, such as the
included GitHub Actions workflow. A daily Claude Code session collects through
web tools into a strictly validated inbox (`airadar/INBOX.md`): every
observation needs a source URL and a method, and a malformed file is rejected
whole, not partially loaded. The session then writes the prose analysis and makes
one tested improvement to the system. The procedure and its guardrails are in
`airadar/DAILY_AGENT.md`.

```bash
python -m airadar.cli collect                          # live APIs (needs network)
python -m airadar.cli validate data/radar/inbox/*.json # check inbox files
python -m airadar.cli daily                            # the full daily cycle
python -m airadar.cli recheck                          # apps to re-observe today
```

Reports land in `data/radar/reports/`: `DATE.md` is the radar's own output,
`DATE-analysis.md` the agent's researched commentary on it.

### Limitations

1. **The first week has no measured growth.** Until apps are re-observed, every
   signal is a stated period gain or a reported rate, graded `sourced`. The
   report says so. The forecast scorecard and the tuner need about 10 days of
   history before either says anything.
2. **Page reads go through a model.** Numbers read off rendered pages by the
   agent's fetch tool can be mangled; they are graded `sourced`, flagged
   `needs_review`, and re-observation corrects them over time.
3. **Coverage follows the sources.** Open-source repos are easy to measure daily;
   commercial apps are mostly seen through what companies choose to report. A
   direction with no open-source footprint is under-represented.
4. **Opportunities are heuristics.** "Fast and uncrowded" or "open source ahead
   of product" are patterns in the radar's own table, graded `assumed`. They
   generate research questions; the agent's analysis is where they are tested.
