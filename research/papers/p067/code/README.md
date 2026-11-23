# p067 code: the deskilling trap (reliance-skill feedback model)

Run everything: `bash run_all.sh` (seeds fixed; ~2-3 min CPU). Outputs (JSON, logs, PNG) go to `../results/`.

| script | what it produces |
|---|---|
| `model.py` | model definition, closed forms (Props 1-4) |
| `exp1_bifurcation.py` | equilibria vs AI error, hysteresis sweep, team-error jump, relaxation rate (fig1, fig2) |
| `exp2_verify_props.py` | Monte-Carlo check of closed forms on random parameter draws; window vs beta; practice floor (fig3) |
| `exp3_planner.py` | myopic vs forward-looking (value iteration) reliance (fig4) |
| `exp4_abm.py` | heterogeneous stochastic population, null model, policies, early-warning signals (fig5, fig6) |
| `exp5_calibration.py` | consistency checks vs Budzyn et al. 2025, Shen & Tamkin 2026, Bastani et al. 2025 (fig7) |
| `exp6_sensitivity.py` | one-at-a-time sensitivity (fig8) |
| `exp7_beta_critical.py` | critical reliance sharpness for bistability |
