# p077 — The Significant Quantity of Compute

Reproduce all numbers and figures in `paper.md`:

    cd code && python3 run_all.py      # ~5 s on CPU; numpy, scipy, matplotlib

* `code/params.py` — every input with its source (or marked ASSUMPTION)
* `code/model.py`  — SQC, accountancy ratio, horizons, inspection capacity, HEM legacy model
* `code/run_all.py` — Monte Carlo (seed 20261122), PRCC sensitivity, MUF-test simulation, figures
* `results/results.json` — all numerical outputs quoted in the paper; `results/fig*.png` — figures
