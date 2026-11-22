# p072 — Layered compliance-partition model of cross-border AI regulatory coordination

Paper: `paper.md`.

## Reproduce (CPU only, ~4 minutes on 4 cores)

```bash
cd code
python3 run_experiments.py 20000   # main Monte Carlo + phase diagram -> results/summary.json, scenario_table.csv
python3 plots.py                   # figures fig1..fig4 + results/phase_check.json
python3 robustness.py              # prior-robustness grid -> results/robustness.csv/json
```
Dependencies: Python 3, numpy, scipy, matplotlib. Seeds are fixed (`SEED=20261122` main, `SEED=7` robustness).

## Files
- `code/obligations.py` — hand-coded obligation matrix (11 dimensions x 6 market segments) with
  provenance strings (article numbers), conflict hypotheses H0–H4, unverified cells.
- `code/model.py` — the model: layered (weights / runtime / process) partition of served segments into
  product variants; exact subset dynamic programming over 2^6 masks; export ("Brussels effect") events;
  chromatic lower bound check.
- `code/run_experiments.py` — scenarios (hypotheses and coordination instruments), Monte Carlo over the
  synthetic prior box, Spearman sensitivity, phase diagram for Proposition 1.
- `code/robustness.py` — re-runs headline metrics under rescaled priors.
- `results/` — all numbers quoted in the paper.

## Caveat
Cost parameters are synthetic priors in arbitrary units (1 = value of serving a USD 1tn-GDP market); they
are not estimates. Reported quantities are frequencies of structural outcomes over the prior box, plus
exact checks of the propositions.
