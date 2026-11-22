# Code for "Verification Blind Spots Bound Error Recovery in Long-Running Agents"

Requirements: Python 3, numpy, scipy, matplotlib (no GPU, no LLM API).

```
cd code && ./run_all.sh      # ~12 minutes on one CPU core
```

| script | what it produces (in ../results/) |
|---|---|
| `theory.py` | closed forms: Thm 1 (interval invariance), Thm 2 ceiling, Prop 1 makespan, Prop 3 miss curves, Prop 4 T* |
| `simulator.py` | independent step-level Monte-Carlo simulator (errors, detectabilities, verifier calls, rollbacks) |
| `exp1_validate.py` | `e1_validation.json` — 96 configurations, theory vs simulation |
| `exp2_floor.py` | `e2_e3_floor_powerlaw.json`, `fig_floor_powerlaw.png` — invariance, ceiling, adaptive policies, power law, cost frontier |
| `exp2b_replicate.py` | `e2b_replication.json` — replication of one Monte-Carlo check |
| `exp4_age.py` | `e4_age_hazard.json`, `fig_age_hazard.png` — context-age hazard, T* scaling, fresh vs snapshot resumption |
| `exp5_mutation.py` | `e5_mutation.json`, `e5_detectabilities.npz`, `fig_mutation.png` — detectability measured on 102 AST mutants of 8 functions |
| `exp6_horizon.py` | `e6_horizon_multiplier.json` — horizon multipliers |

All random seeds are fixed in the scripts.
