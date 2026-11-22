# Code for "Assessment Design as the Binding Constraint on Learning with AI"

Requirements: Python 3, numpy, matplotlib (no GPU, no API keys).

Run from this directory, in order (total CPU time about 80 s on one core):

    python3 calibrate.py      # inverse calibration to Bastani et al. (PNAS 2025) moments -> results/calibration.csv, calibration_summary.json
    python3 experiments.py    # E1-E5 -> results/summary.json, fig1..fig5
    python3 sensitivity.py    # S1-S3 robustness -> results/sensitivity.json, sensitivity_table.csv
    python3 fig6_guardrail.py # Figure 6 (guardrail non-monotonicity) from sensitivity.json

Files
- `model.py`: skill dynamics, practice score, backward-induction solver for the (present-biased) student, simulator.
- `calibrate.py`: grid search over (offload share a, retained engagement kappa) matching the reported practice (+48%, +127%) and exam (-17%, ~0%) effects, swept over unobserved (s0, eta*T, q).
- `experiments.py`: E1 monotonicity check (Prop. 1), E2 informativeness closed form vs Monte Carlo (Prop. 2), E3 last-period rule and minimal proctored weight w_95, E4 fading theorem check (Prop. 4), E5 cohort policy simulation. `python3 experiments.py e3` reruns a single experiment and merges into summary.json.
- `sensitivity.py`: S1 closed-form last-period rule vs DP; S2 sensitivity of headline cohort effects to T, eta, kappa_base, prior-skill distribution; S3 guardrail non-monotonicity scan.

Random seeds are fixed (experiments.py: 20261122; sensitivity.py: 7). Logs of the reported run are in `../results/log_*.txt`.
