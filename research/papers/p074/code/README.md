# Reproducing "Logarithmic Leverage" (p074)

Requirements: Python 3, numpy, matplotlib (tested with numpy 2.5.3).

    cd code
    python3 run_all.py        # ~7 s on 4 CPU cores, seeded, deterministic

Outputs (in `../results/`):

| file | contents |
|---|---|
| `results.json` | every number quoted in the paper, keyed by experiment (E1-E7) |
| `run_log.txt` | console log of the run |
| `fig1_interventions.png` | change in US lead from prospective 2027 interventions (H-A) |
| `fig2_inversion.png` | implied follower advantage Phi (H-A) and implied recapture S (H-M) |
| `fig3_leakage.png` | leakage ceiling / convexity of enforcement value (Prop. 2) |
| `fig4_endogenous.png` | lag dynamics under endogenous follower R&D (Prop. 4) |
| `fig5_power.png` | power of lag-regression tests vs. policy-shock size (Prop. 6) |

Files:
- `params.py` — every input, labelled (S) sourced or (A) assumption.
- `model.py` — lag identity, the three structural hypotheses, stock-flow and catch-up dynamics.
- `run_all.py` — experiments E1 (rates), E2 (inversion), E3 (policy values), E4 (leakage),
  E5 (stock filter, incl. numerical check of the closed form), E6 (endogenous R&D), E7 (identification/power).
