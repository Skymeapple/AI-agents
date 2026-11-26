# What drove each breakthrough

This page takes every measured gain in the programme and asks which ingredient
produced it: new information, more data, a better algorithm, a theory borrowed
from another field, or more reasoning per decision. It then looks at how the
ingredients combine.

It complements [`SYNTHESIS.md`](SYNTHESIS.md), which reviews the projects one by
one. All numbers are from the experiments in this repository on public data. None
is scored on HFM's private tests.

## 1. The five ingredients

| Ingredient | Meaning here | Example |
| --- | --- | --- |
| **New information** | An input or measurement the method was not given before | A neighbouring buoy's wave record; a galaxy's size and shape |
| **More data of the same kind** | More examples, more runs, or pooling across related tasks | All training sequences instead of 40,000; other steels as donors |
| **A better algorithm** | A different way to search or decide, on the same inputs | Local search; planning against scenarios |
| **Theory from another field** | A known physical or mathematical form built into the model | Arrhenius kinetics; the pressure equation of a fluid |
| **More reasoning per decision** | Simulating ahead, sampling futures, or simply searching longer | Look-ahead dispatching; ten times more search rounds |

## 2. How much each ingredient delivered

Gains are relative to the best practical method without that ingredient.

### New information

| Project | What was added | Gain |
| --- | --- | --- |
| Wave energy | A neighbouring buoy's waves | Error in annual resource 29% → 19% (−34%) |
| Optical constants | Another measurement of the same material | Skill +0.09 → +0.35 |
| Steel | One extra cyclic test (±2% strain) | Error −27%; 71% of the remaining gap closed |
| Bike sharing | A demand forecast | Failed trips −16% |
| Weather | Forecasts from other model families | Error −8.4% |
| Wind farms | The wind-direction offset of the site | Gain at the unseen farm 1.4% → 11.9% |
| Galaxy redshift | Extinction, size, shape, apertures, errors | Error −5.8% |

- **Seven of nine attempts worked.** The two that failed tried to substitute
  for the real thing: a look-alike material instead of the same material
  (optical constants), and tenant or batch identity instead of the lifetime
  itself (cloud placement).
- **Typical size: 6% to 34%, median about 16%.** This is the largest and most
  reliable ingredient.

### More data of the same kind

| Project | What was added | Gain |
| --- | --- | --- |
| Steel | A library of other steels | Error −73% against using the tension test alone |
| Regulatory DNA | All training sequences, not 40,000 per cell line | Correlation 0.593 → 0.657 |
| Regulatory DNA | Pooling three cell lines in one model | +0.039 (+0.076 on the smallest dataset) |
| Convection reconstruction | Ten fitting runs instead of five | Error −13.5%; ceiling of the method 0.34 → 0.24 |
| Weather | Five model families instead of three | Blend gain 4.6% → 8.4% |

- **Worked every time it was tried,** with the largest effect where the task
  had the least data of its own.
- **It does not help when the information is absent.** Three model families
  with unlimited wind data all stopped at 30% error on wave energy.

### A better algorithm

| Project | Change | Gain |
| --- | --- | --- |
| Grid scheduling | Plan against spread wind scenarios, not a reserve rule | Excess cost −23% |
| Optical coatings | Choose designs by their score under variation | Expected reflectance −14.5% |
| Job scheduling | Local search on top of the dispatch rule | Lateness −10% |
| Rail | Look-ahead instead of a priority rule | Delay −9.9% |
| Cloud placement | First fit instead of best fit | Server time −3.2% |
| Table questions (small open model) | Choose between a code route and a direct route with a simple rule | Correct answers 28.0% → 41.2% on fresh questions; one harness detail was worth 26 points on the code route |
| TOPv2 parsing (small open model) | Constrain output to well-formed trees with known labels | Exact match 26.7% → 45.7%; no gain on unseen tree shapes |
| Lung cells | A small network averaged with nearest neighbours, instead of a linear classifier | Macro-F1 0.684 → 0.786, in 35 of 35 studies |

- **Nine of sixteen attempts worked,** counting part-way trading of a fast
  portfolio signal (return per unit of risk 0.50 → 0.60). Failures: joint
  spill planning across eight dams (1.4% at best), drift-triggered portfolio
  rebalancing, headway-based bus holding (worse by 4–12%), random-scenario scheduling (unstable), lifetime classes for cloud
  placement, single-scenario look-ahead for bikes, and adding an untrained small
  model's entities to a dictionary tagger (F1 83.2 → 75.7).
- **Typical size: 3% to 23%, median about 10%.**
- **Algorithms fail when their premise is wrong.** Bus holding assumed bunching
  was costly; on these routes it was not.

### Theory from another field

| Project | Theory | Gain |
| --- | --- | --- |
| Cold battery | Arrhenius temperature law (chemistry) | Error −23% against a fitted slope, on two cells |
| Convection reconstruction | Pressure equation of the fluid | Pressure error −36%; total −10.8% |
| Wind farms | Wake physics as the base of a learned model | Best overall when combined with learning |
| Regulatory DNA | Strand symmetry of DNA | +0.02 correlation |

| Did not help | Theory |
| --- | --- |
| Optical constants | Causality (Kramers–Kronig) alone |
| Wave energy | Separate physics for swell and wind-sea |
| Lung cells | The cell-type hierarchy |
| Steel | A statistical prior conditioned on the tension test |
| Galaxy redshift | Predicting a full probability distribution |
| Air-quality sensors | Hygroscopic growth law for humidity |
| Dam spill | Saturating gas-from-spill form |

- **Four of eleven worked.** The lowest hit rate of the five ingredients.
- **When it works it is cheap and it transfers.** The Arrhenius result held on
  an independent cell with no change to the code.
- **The pattern in the failures:** theory helps when it supplies a relationship
  the data cannot show (how resistance behaves colder than anything measured;
  a field that is never measured). It does not help when it only re-arranges
  information the model already has.

### More reasoning per decision

| Project | Extra reasoning | Gain |
| --- | --- | --- |
| Rail | Simulate each option to the end of the day | Delay −9.9% |
| Bike sharing | Look-ahead on one past day | +0.7% (no gain) |
| Bike sharing | Look-ahead averaged over four past days | −2.7%, better on 13 of 17 days |
| Rail | Look-ahead over four randomly sampled futures (four times the compute) | −8.8%, against −9.7% without sampling |
| Rail | Look-ahead told each train's delay upstream (information, not reasoning) | −14.8% |
| Bus holding | Look-ahead with perfect foresight | Under 1% |
| Job scheduling | Ten times more search rounds, twice | 7.1% → 9.7% → 10.1% |
| Grid scheduling | 20 scenarios instead of 10 | 23% → 24% |
| Weather | Blend weights by latitude band | 8.4% → 8.5% |
| Looped transformer (toy scale) | Apply two shared layers three times: twice the compute, same parameters | Closes 63% of the gap to a model twice as deep; loses to it at equal compute |
| HFM's agents | Hundreds of further tuning experiments | Median gain per accepted step 0.15% |

- **Returns fall fast.** The first step of look-ahead or search is worth
  several percent. Each further tenfold increase is worth about a point or
  less.
- **Its value is capped by what it reasons about.** Rail look-ahead works
  because the timetable is known. Bike look-ahead on one remembered day does
  nothing, on four days it captures a quarter of what a perfect forecast would
  give. Bus look-ahead has nothing to find.

## 3. Ranking the ingredients

| Ingredient | Attempts that worked | Typical gain when it worked | Cost |
| --- | --- | --- | --- |
| New information | 7 of 9 | 6–34% | Needs a data source or a measurement |
| More data of the same kind | 5 of 5 | 6–73% | Needs the data to exist |
| A better algorithm | 9 of 16 | 3–23% | Engineering time |
| Theory from another field | 4 of 11 | 3–36% | Nearly free once known |
| More reasoning per decision | 3 of 8 | 0–10%, falling fast | Compute |

The counts are small and come from one programme. The ordering is the finding,
not the exact numbers: **what the method knows mattered more than how cleverly
it used it.**

## 4. How the ingredients combine

The largest total gains came from stacking different ingredients, each fixing a
different deficit.

| Project | Step by step | Total |
| --- | --- | --- |
| **Steel** | Tension test alone 0.545 → library of other steels 0.145 (data) → one extra cyclic test 0.104 (information). Best possible for the model: 0.087. | −81% |
| **Wind farms** | No wake model → physics, default settings −1.4% (theory) → site direction calibrated −11.9% (information) → learned residual added −29.8% (data and algorithm) | −29.8% |
| **Regulatory DNA** | Separate models 0.535 → strand averaging 0.552 (theory) → one multi-task model 0.593 (pooled data) → all training data 0.657 (more data) → five-seed ensemble 0.672 (more compute) | +0.137 |
| **Weather** | One model → bias-corrected −0.5% (algorithm) → blend of three −4.6% (information) → blend of five −8.4% (more of it) | −8.4% |
| **Convection** | Gappy POD 0.381 → pressure from the flow equation 0.340 (theory) → twice the fitting data, a further −13.5% at the one setting tested (data) | about −20% |
| **Cold battery** | Hold warm values → fitted slope +0.15 → Arrhenius form +0.34 (theory). Refinements beyond that did not carry to a second cell. | +0.34 |
| **Galaxy redshift** | Five groups of extra inputs worth 1–2% each, 5.8% together (information) | −5.8% |

Four regularities in how they combine:

1. **Different ingredients add; the same ingredient saturates.** In the DNA,
   wind and redshift projects each new kind of ingredient added its full
   amount on top of the others. Repeating one kind gave less each time: search
   rounds, scenario counts, tuning steps.
2. **Information sets the ceiling; algorithms and reasoning decide how close
   you get.** Cloud placement is the clearest case. With true lifetimes a
   simple rule saves 13.8%. Without them, five different algorithms save
   nothing beyond first fit.
3. **Order matters.** Calibrating the site's wind direction first made the
   physics model useful, which in turn gave the learned part a good base. A
   learned residual on an uncalibrated model would have been fitting an
   error that one number removes.
4. **Theory and data substitute for each other at the edges.** Where data is
   thin or absent (colder than calibration, an unmeasured field), theory
   carries the result. Where data is plentiful, a plain learned model matches
   or beats the structured one (wave energy, lung cells, DNA language models
   against supervised baselines in the literature).

## 5. What this means for a comprehensive breakthrough

No single ingredient produced a comprehensive result. The projects that moved
furthest followed the same sequence:

1. **Find what the method does not know.** An oracle test (give the method the
   suspected missing fact and measure the gap) located it in steel, cloud
   placement, bikes and rail.
2. **Supply it, by measurement, by a library or by a second source.** This
   step produced the largest single gains.
3. **Pool every related dataset.** Other steels, other cell lines, other
   models' forecasts, other runs.
4. **Put known theory where the data cannot reach,** and only there.
5. **Then improve the algorithm,** and stop adding reasoning once another
   tenfold costs more than it returns.
6. **Judge every step on data held out along the real shift, with noise
   measured.** About half the ideas here failed that test, including several
   that looked right in a smaller trial.

HFM's agents, left alone, spent almost all their effort on the last half of
step 5. That is why their typical step was worth 0.15%, while single steps
from the first four were worth 5% to 70% here.

## 6. Where AI stands on each ingredient

| Ingredient | What AI contributed in this programme | What it could not do |
| --- | --- | --- |
| New information | Found public sources and used them correctly once pointed at the gap | Produce a measurement that does not exist; decide a task's rules allow a new source |
| More data | Pooled and scaled without difficulty | Create data where none is public or licensed |
| Algorithms | Implemented every one, including simulators, quickly | Know in advance which would work: 5 of 9 |
| Cross-field theory | Recalled and applied the right laws | Tell which applications would help: 4 of 9 |
| Extended reasoning | Scaled it freely | Avoid over-investing in it; this is where unsupervised agents get stuck |

## 7. Caveats

- **Small counts.** Between five and nine attempts per ingredient.
- **Classification is a judgement.** Blending weather models is counted as
  information; it could be called data or algorithm. Look-ahead appears under
  both algorithm and reasoning.
- **Gains are on different scales** (error, cost, correlation, skill) and are
  compared only loosely.
- **Selection.** The ideas tested were chosen by the same AI that then
  classified them.
