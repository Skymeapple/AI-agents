# Assumptions, claim labels and checkpoint (second pass, 1 November 2026)

## Claim labels used throughout
- **Verified**: checked against real data or exact computation here (e.g. Kepler exponent from real orbits; Strassen tensor check; wheel installs and passes tests).
- **Simulated**: real code and statistics run against a synthetic world whose rules we wrote; shows a method works under stated assumptions only.
- **Hypothesis**: stated, not yet tested.
- **Unresolved / blocked**: needs access this environment does not have.

## Material assumptions
1. No LLM API key exists here, so agent roles are deterministic code (and model sub-agents for rating tasks). Results about "agents" describe this code, not LLM agents.
2. Verifier-map raters (task 1.1) are two independently prompted model instances, not human experts. They likely share training biases, so their agreement (kappa 0.92) overstates true inter-expert agreement. Readiness score = strength - cost_exp/4 - latency_exp/4 (our weighting).
3. Research-agent world (2.2-2.5, RP3): 20-variable linear-Gaussian causal model, edge probability 0.15, perfect interventions, literature error 17.9% (from Kosmos). Interventions here are free of confounds; real experiments are noisier, so the 99% synthesis accuracy is an upper bound.
4. RP6 inheritance rates (abstract simulation) are assumed; the executable agent (assumption 3) measures them instead.
5. Simulated lab: 3-component thin film plus anneal temperature, 3% failed runs, noise 0.05; planted anomaly width fixed.
6. Licence for the release candidate: Apache-2.0 (not yet confirmed by Liam).
7. Data gap map: figures as of pages opened 1 Nov 2026; items marked unconfirmed were not seen on a page.

## Blockers (documented, not worked around)
| Item | Blocked by | Exact next action |
| --- | --- | --- |
| 1.5 public release | Needs Liam's approval and venue | On approval: create repo, push `ai-science/`, tag v0.1.0; `python -m build && twine upload dist/*` if PyPI |
| RP1 on real literature | Network policy denies api.openalex.org and export.arxiv.org | Allow those hosts in the cloud environment, then run `taste_pipeline` on OpenAlex abstracts 2000-2014 |
| 5.3 blinded review | Needs human expert reviewers and human postdoc outputs | Recruit 6 reviewers; run `build_review_packets` on 20 agent + 20 human reports |
| 5.4 impact tracking | Needs a bibliographic API and 24 months | Wire `ImpactTracker.fetch` to OpenAlex once allowed; snapshot quarterly |
| 2.2 with a real LLM | Needs an API key | Replace the deterministic roles in `agent.py` with LLM calls behind the same WorldModel |
| RP5, 3.5 real campaign | Needs a robotic lab | Swap `SimulatedLab` for an instrument driver behind `LabAPI` |
| RP3 accuracy on real agent claims | Needs LLM agent runs and 3 experts | Collect 1,500 synthesis claims per arm per the protocol |

## Checkpoint
All feasible items are complete; 19 tests pass (`python -m pytest -q tests`). Nothing is mid-flight.
