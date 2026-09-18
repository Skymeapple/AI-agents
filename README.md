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

## Quick start

```bash
pip install numpy pyyaml requests pytest

# Generate a synthetic demo corpus (no network required)
python scripts/make_demo_corpus.py

# Rank research directions
python -m sciscout.cli rank --corpus data/demo_corpus.json

# Model one direction's investment case
python -m sciscout.cli model --corpus data/demo_corpus.json \
    --track solid-state-sodium --tam 12e9 --check 5e6 --ownership 0.15

# Full markdown report
python -m sciscout.cli report --corpus data/demo_corpus.json --out report.md

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

2. **The demo corpus is synthetic.** Every record in `data/demo_corpus.json` is
   machine-generated, with `synthetic:` identifiers and placeholder authors. It
   exercises the pipeline. It tells you nothing about any real field.

3. **Clustering into research directions is by explicit label.** Proper
   clustering needs embeddings or citation-graph community detection. Grouping by
   a human-assigned label is a real limitation and also a defensible default: the
   boundary of a "research direction" is a judgement a clustering algorithm makes
   silently and usually badly.

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
  report.py              markdown rendering
  cli.py                 harvest / rank / model / report / sectors
  sources/               arxiv, openalex, local JSON corpus
  scoring/               citation baseline, importance and urgency, TRL inference
  commercial/            TRL -> stages, costs, timelines, constraints
  invest/                distributions, investment case, Monte Carlo engine
  priors/sectors.yaml    commercialisation priors, graded by provenance
scripts/                 synthetic corpus generator
tests/                   42 tests
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite asserts the modelling claims directly, not just that the code runs:
abandoned programmes accrue no downstream cost, capital-hungry programmes dilute
early investors, sensitivity analysis recovers a driver that was deliberately
planted, missing citation data is scored as absent rather than zero, and a
two-year publication history is refused as a trend.
