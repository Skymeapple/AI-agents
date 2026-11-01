# AI for autonomous discovery: execution prototypes

Companion to the study "AI for Autonomous Scientific Discovery: Study and Execution Plan".
Everything here runs on a laptop with Python 3.11+, numpy, scipy, sympy, jsonschema and pytest.
Labs are SIMULATED; results in `results/` test methods under stated assumptions, not real-world claims.

| Path | Study task | What it is |
| --- | --- | --- |
| aisci/verifiers.py | I1 1.2, 1.4 | Verifier service: unit tests, symbolic identity, matrix-multiplication tensor, optimum claim, dimensional analysis; soft judges and ensemble |
| aisci/calibration.py | I1 1.3 | False-pass / false-reject rates with Wilson 95% CIs |
| aisci/worldmodel.py | I2 2.1, 2.3 | World-model schema with enforced provenance, retraction and suspect propagation |
| aisci/lab.py | I3 3.2-3.4 | Protocol JSON schema, simulated thin-film lab, Lab API, safety interlocks |
| aisci/campaign.py | I3 3.5, RP4 | 100-experiment closed-loop campaigns: random vs BO vs BO+surprise score |
| aisci/datastore.py | I4 4.2, 4.3 | Experiment record standard v1, hash-chained store that logs negatives and blocks |
| benchmark/ | I5 5.1 | 30-discovery rediscovery benchmark spec |
| aisci/rediscovery.py | I5 5.1, 5.2 | Executable Kepler/Planck tasks, baseline law-finder, cutoff-aware novelty metric |
| aisci/screen.py | I6 6.2 | Dual-use screen (placeholder list) and 50-attempt red-team harness |
| aisci/rp2_ensemble.py | RP2 | Ensemble-verifier simulation with shared-error sweep |
| aisci/rp6_longrun.py | RP6 | 200-cycle compounding-error and audit simulation |
| docs/protocols/ | RP1-RP6 | Full research protocols with sample sizes and kill criteria |
| docs/governance/ | I6 | Authorship policy, screening rules, pre-registration template, replication fund |
| aisci/verifier_map.py, results/verifier_map/ | I1 1.1 | 20-domain verifier map, two independent raters, agreement stats |
| aisci/agent.py | I2 2.2-2.5, RP3 | Executable research agent with world model, critic, auditor, interventional tests, on a known causal world |
| aisci/domain_model.py | I4 4.4 | Surrogate model trained on the experiment store vs baselines |
| aisci/evaluation_tools.py | I5 5.3, 5.4, RP1 | Blinded review kit, impact tracker, time-split question-impact pipeline |
| docs/data_gap_map.md | I4 4.1 | Open-dataset audit and ranked gaps |
| docs/ASSUMPTIONS_AND_CHECKPOINT.md | all | Claim labels, assumptions, blockers, exact next actions |
| pyproject.toml, CHANGELOG.md, RELEASE_CHECKLIST.md | I1 1.5 | Release candidate (built and install-tested, NOT published) |
| tests/ | all | One test per "done when" criterion: `python -m pytest -q tests` |

Reproduce results: `python -m aisci.calibration`, `python -m aisci.rp2_ensemble`, `python -m aisci.rp6_longrun`,
`python -m aisci.campaign` (about 3 minutes), `python -m aisci.rediscovery`, `python -m aisci.screen`, `python -m aisci.agent`, `python -m aisci.campaign --rp4`,
`python -m aisci.domain_model`, `python -m aisci.verifier_map`.
