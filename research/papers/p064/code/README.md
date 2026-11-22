# Code for "Rebuilding the Expertise Pipeline" (p064)

Everything is CPU-only (numpy, scipy, matplotlib). Total runtime is about 50 s on one core.

    bash run_all.sh        # writes all JSON/CSV/PNG/stdout files to ../results/

| file | what it does | outputs |
|---|---|---|
| `model.py` | analytic formulas (reproduction number, Lotka root and bounds, closed-form shortfall path, detection time, Hayes critical delay) and a deterministic cohort simulator | none |
| `bls_data.py` | 11 rows copied verbatim from the BLS Employment Projections table "Occupational separations and openings, projected 2025-35" | none |
| `common.py` | plotting style and the baseline calibration (software developers) | none |
| `exp1_theory_checks.py` | numerical checks of Propositions 1-3 | `fig1_theory_checks.png`, `exp1_theory_checks.json` |
| `exp2_calibration.py` | detection lags and locked-in deficits for 11 occupations, plus a Monte Carlo over the unobserved parameters | `fig2_detection_lag_by_occupation.png`, `exp2_calibration.{csv,json}` |
| `exp3_policies.py` | rebuilding pathways A-D and their combinations, started at years 0, 2, 5 and 8.1 | `fig3_rebuilding_pathways.png`, `exp3_policies.json` |
| `exp4_feedback.py` | shortage-reactive versus pipeline-targeting intake rules, and a stability map checked against Hayes (1950) | `fig4_feedback_stability.png`, `exp4_feedback.json` |
| `exp5_abm.py` | stochastic individual-level simulation with heterogeneous training times, and a comparison of three indicators for detecting shocks | `fig5_abm_indicators.png`, `exp5_abm.json` |

All random seeds are fixed. The parameters that are not observed are set in `common.py`: time-to-seniority tau = 5 years and completion probability p = 0.8. Shock sizes are also assumptions: the hiring cut x is 0.13, 0.16 or 0.19 and the learning loss d is 0 or 0.17. The paper explains where each value comes from and how sensitive the results are to it.
