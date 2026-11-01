# Research protocols RP1 to RP6

Version 1.0, 1 November 2026. Expands section 5 of the study into runnable protocols. Sample sizes use a two-sided alpha of 0.05 and 80% power (computed in `results/`; script lines in this file's appendix). "Sim result" reports the toy simulation run in this repository, which tests the method under stated assumptions and is not evidence about the real world.

Shared rules for all six: pre-register hypothesis, primary metric and kill criterion before data collection (governance template G3); use the verifier service (`aisci/verifiers.py`) and the experiment store (`aisci/datastore.py`); log negative results; review at month 6 against the kill criterion.

---

## RP1. Research taste: predicting which open questions matter

**Primary hypothesis.** A model given only literature up to year T ranks open questions by their impact at T+10 with Spearman rho of at least 0.4.

**Design.** Retrospective, time-split. 5,000 open questions extracted from review articles published 2000 to 2014 (stated "open problem", "remains unknown" sentences), deduplicated by embedding clustering.

**Outcome (impact at T+10).** Composite of citations to papers resolving the question, inclusion in prize citations, and number of follow-up papers. Weights fixed before analysis.

**Arms.** (A) model alone; (B) 30 senior scientists, each ranking 200 questions from their own field, blind; (C) model plus scientist; (D) citation-count heuristic baseline.

**Sample size.** Detecting rho = 0.2 needs 194 questions per field; detecting 0.4 needs 47. With 5,000 questions across about 10 fields, each field has about 500, which is enough.

**Analysis.** Spearman rho per field with bootstrap CIs; top-decile recall; paired comparison of A against B and D.

**Prospective arm (month 12 to 18).** The model proposes 50 questions for 2027. Experts rate them now, and the impact tracker (task 5.4) scores them later.

**Kill criterion.** rho below 0.2, or not better than D, at month 9.

**Needs.** A full-text literature corpus with licences, 30 experts at about 10 hours each, and LLM compute. **Not runnable here:** no corpus or expert access.

---

## RP2. Discovery without a strong verifier

**Primary hypothesis.** An ensemble of weak judges (simulation, literature consistency, learned critic) halves the false-pass rate of the best single judge at a matched false-reject rate of 30%.

**Design.** 2,000 published biology hypotheses with known lab outcomes, including failed replications (sources: replication projects and retraction databases). Train on outcomes resolved before 2022 and test on those after.

**Sample size.** Detecting 20% to 10% false-pass needs 199 false hypotheses per arm. 2,000 total, at about a 70% false share, gives about 1,400.

**First measurement (new, from the simulation).** Before fitting any ensemble, estimate the correlation of errors between judges on 200 labelled cases. If the shared-error component is large, stop or add a judge of a different kind (for example a cheap wet assay).

**Sim result** (`results/rp2_ensemble.json`). With independent judge errors the ensemble cut false passes by 96 to 100% across 5 seeds. With a moderate shared error (weight 1.0) the reduction fell to 41%. With a large one (2.0) it fell to 4%, which would trigger the kill criterion. **Ensembles only help when judges fail in different ways.**

**Kill criterion.** Less than 15% improvement at month 12.

**Needs.** The curated hypothesis set, domain simulators and assay budget. **Runnable here:** the method and the simulation only.

---

## RP3. Reliable cross-domain synthesis

**Primary hypothesis.** Requiring every synthesis claim to cite at least two supporting claims, and checking each with a dedicated critic, raises expert-judged synthesis accuracy from 57.9% (the Kosmos baseline) to at least 75%.

**Design.** Two arms over identical agent runs: (A) free synthesis; (B) chain-of-evidence rule plus critic. The rule is already enforced in code: `WorldModel.add` rejects a synthesis claim with fewer than two supports.

**Sample size.** 57.9% to 75% needs 119 claims per arm. The planned 1,000 also allows detecting smaller gains (5 points needs 1,442 per arm, so the plan should be raised to 1,500 per arm if the goal is the kill threshold).

**Error taxonomy.** Unsupported leap, wrong scope (finding from one species applied to another), stale literature, statistical misread, and citation mismatch. Two annotators each, reporting kappa.

**Kill criterion.** Gain of less than 5 points after two design iterations.

**Needs.** LLM agent compute and 3 experts. **Runnable here:** the rule exists as code and is tested (`test_2_3`); the accuracy study is not runnable here.

---

## RP4. Recognizing anomalies

**Primary hypothesis.** Agents with a surprise score (replicate any result more than 3 standard deviations above prediction, then explore near it) find planted or historical anomalies at least twice as often as agents that drop outliers.

**Design.** (1) Replay 20 historical datasets that contained a famous anomaly. (2) Plant synthetic anomalies in live closed-loop campaigns. Arms: outlier-rejecting agent against surprise-pursuing agent, on matched seeds.

**Sample size.** Recall of 30% against 60% needs 42 anomalies per arm. 20 historical cases are underpowered on their own, so at least 25 planted anomalies must be added.

**Sim result** (`results/i3_rp4_campaign.json`, 12 seeds, 100 experiments each). Both optimisers found the planted high-property phase in 67% of runs, against 33% for random search. The surprise score added nothing, and it produced 0.08 false alarms per 100 experiments. In this toy landscape the optimiser explored enough to find the anomaly anyway. **The hypothesis is unsupported in simulation.** The real test needs exploit-heavy agents and historical data, where dropping outliers is known to have hidden discoveries.

**Kill criterion.** Recall below 30% at an affordable false-alarm rate, or no gain over outlier rejection.

**Needs.** Curated historical datasets and a live lab. **Runnable here:** the simulation only.

---

## RP5. Making predictions survive the bench

**Primary hypothesis.** Filtering candidates for synthesizability and characterization uncertainty before robot time doubles the confirmed-novel yield per 100 experiments.

**Design.** Matched campaigns in one inorganic materials system, with and without filters. Every product is characterized by two methods (XRD plus a second method), and external crystallographers judge novelty blind against ICSD.

**Sample size.** 10% to 20% yield needs 199 experiments per arm, so two campaigns of at least 100 robot runs per arm.

**Kill criterion.** Yield gain below 25% after two campaigns.

**Needs.** A robotic synthesis line, characterization access and ICSD access. **Not runnable here.**

---

## RP6. Keeping long runs honest

**Primary hypothesis.** Periodic audits of the world model keep claim accuracy flat from cycle 10 to cycle 200 and retain at least 10 points of accuracy, for at most 40% extra compute.

**Design.** 30 matched long campaigns (15 seeds per arm), with 10 false claims seeded at cycle 20, tracking catch time and accuracy curves.

**Sim result** (`results/rp6_longrun.json`; per-type error rates from Kosmos; inheritance rates assumed). Without audits, accuracy settled at about 61% by cycle 10 and stayed there. Compounding cost about 15 points against the 76% per-claim baseline, and it all happened early. Auditing 10 claims every 5 cycles (40% overhead) raised cycle-200 accuracy to 74% (+13 points) and caught about half the seeded errors, with a median catch time of 35 cycles. Auditing every cycle reached 89% at 2 times the compute. **The kill criterion is passed at 40% overhead.** One correction to the protocol: accuracy was flat in both arms, so "flat over cycles" is the wrong primary metric. The level reached early is what matters, so the metric changes to accuracy at cycles 10 and 200 against the per-claim baseline.

**Needs.** Real agent runs to measure inheritance rates. **Runnable here:** the simulation, plus the production retraction and suspect-flag logic (tested).

---

## Appendix: sample-size calls

Two-proportion z-test, alpha 0.05 two-sided, power 0.8; Fisher-z for correlations. Values: RP1 n=194 (rho 0.2), 47 (rho 0.4); RP2 199 (0.20 vs 0.10); RP3 119 (0.579 vs 0.75), 1,442 (0.579 vs 0.63); RP4 42 (0.30 vs 0.60); RP5 199 (0.10 vs 0.20).
