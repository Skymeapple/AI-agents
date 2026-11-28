# p018 code: survival analysis of finite-universe-verified graph conjectures

Run everything: `bash run_all.sh` (about 6 CPU-minutes on 4 cores; ~0.7 GB of binary data in `../data/`).

| file | role |
|---|---|
| `invariants.c` | exact invariants (n, m, delta, Delta, alpha, omega, chi, gamma, nu, diam, rad, Havel-Hakimi residue, annihilation number, triangles) for graph6 input, n <= 20 |
| `validate_invariants.py` | cross-checks the C engine against independent brute-force/networkx code on all 995 connected graphs with n <= 7 plus 60 random graphs with 8-9 vertices (result: `results/validation_invariants.txt`) |
| `gen_large_sample.py` | falsification sample on 11-16 vertices: G(n,p) graphs plus structured families (paths, cycles, lollipops, circulants, trees, multipartite, ...) |
| `analysis.py` | enumerates 12,922 inequality templates, computes per-order maxima, survival, predictors, Dalmatian-like significance, known-theorem sanity checks |
| `analysis2.py` | hazard vs record age, stable-conjecture false acceptance, drift census, falsifier coverage, cross-validated hazard model, figures |
| `analysis3.py` | gap-closing (crossing-time) rule and tiny-graph tightness diagnostic |

Outputs land in `../results/` (`summary*.json`, `late_counterexamples*.json`, `templates.csv`, `conjectures_by_k.csv`, figures).
`geng` is nauty's generator (McKay & Piperno 2014), built from the sources bundled in the pynauty 2.8.8.1 sdist.
