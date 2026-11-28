# Code for "The Verification Ceiling"

Requirements: Python 3.13, numpy, scipy, matplotlib, networkx; the dependency graphs in
Experiment 2 are extracted from whichever versions of numpy, scipy, matplotlib, networkx,
sympy, pandas, setuptools, pip and the standard library are installed (versions used for the
paper are recorded in `results/exp2_chainref.json`). `pip install networkx matplotlib sympy pandas`.

Run everything (about 3 CPU-minutes, at most 2 processes):

    bash code/run_all.sh

| File | Purpose | Output in `results/` |
|---|---|---|
| `model.py` | closed forms (Lemma 1, Theorems 1-2, Propositions 3 and 5) | - |
| `exp1_validate.py` | Monte Carlo check of Lemma 1 / Theorem 1; ceiling table and Fig. 1 | `exp1_validate.json`, `fig1_ceiling.png` |
| `graphs.py`, `dagsim.py` | import-DAG extraction; project simulator; water-filling allocation | - |
| `exp2_dags.py` | real dependency DAGs: structure, Prop. 4 check, allocation policies, sweeps | `exp2_structure.json`, `exp2_sim.json`, `exp2_chainref.json` |
| `exp2_summarize.py`, `exp2_figure.py` | tables and Fig. 4 | `exp2_summary.json`, `fig4_dag_policies.png` |
| `exp3_hetero_diversity.py` | heterogeneity (Prop. 3) and verifier diversity (Prop. 5) | `exp3_hetero_diversity.json`, `fig2_heterogeneity.png`, `fig3_diversity.png` |
| `exp4_calibration.py` | calibration to published false-acceptance figures, sensitivity | `exp4_calibration.json` |

Seeds are fixed; Experiment 2 results change if the installed package versions change, because the graphs do.
`R` and `R_SWEEP` environment variables set the number of Monte Carlo replicates (defaults 4000 and 2000).
