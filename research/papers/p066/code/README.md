# Code for "Offloading Traps in Adolescent Skill Formation" (p066)

Requirements: Python 3 with numpy, scipy, matplotlib (no GPU, no API keys).

```
cd code
bash run_all.sh        # ~15-20 min CPU total
```

| script | what it does | outputs (../results/) |
|---|---|---|
| `theory.py` | Reduced one-state model: analytic offloading threshold, bistability check, comparative statics, beta*eta check, stationary DP checks, tutor-retention analysis | `theory_results.json`, `fig_theory_threshold.png`, `fig_theory_sigma.png`, `fig_theory_sigma_q_map.png`, `theory_sigma_q_map.npy` |
| `model2.py` | Two-state (skill, persistence) model used by the next two scripts | - |
| `calibrate.py` | Identified-set search against the six arm-level moments of Bastani et al. (2025, PNAS) | `abc_accepted.npz`, `calibration_summary.json`, `calibrate_log.txt` |
| `project.py` | Long-run (age 12-20) projections across the identified set, scenario sweep, exposure-window experiment, PRCC sensitivity | `projection_results.json`, `fig_projection.png` |

Random seeds are fixed in each script. The only empirical inputs are the six published
arm means of Bastani et al. (2025) hard-coded in `calibrate.py` (TARGET).
