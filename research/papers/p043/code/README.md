# Code for "Cue or Act?" (p043)

CPU-only. Requires Python 3 with `numpy scipy scikit-learn matplotlib jax optax`
(`pip install jax optax scikit-learn matplotlib`).

Run from this directory, in this order (later scripts load models saved by `run_main.py`):

```
python run_main.py --seeds 0 1 2 3 4 5 --variant dep     # ~1.5-3 min per seed on one core
python run_main.py --seeds 0 1 2 3 --variant indep       # ablation: truth-independent instructed lies
python run_controls.py --variant dep --seeds 0 1 2 3 4 5 # probes trained with cue-without-lie negatives
python run_controls.py --variant indep --seeds 0 1 2 3
python run_obfuscation.py --seeds 0 1 2 --lam 1.0        # fine-tune the model against frozen detectors
python theory_sim.py                                     # linear-Gaussian check of Props. 1, 3, 4
python make_figures.py                                   # writes results/summary.md and figures
```

Files:

- `world.py` - the synthetic "lying world" (three lie triggers, two cue-without-lie controls).
- `model.py` - 4-layer pre-LN transformer in JAX and its training loop.
- `detectors.py` - LR probe, difference-of-means, Mahalanobis, and the belief-output mismatch detector.
- `run_main.py` - trains one model per seed and evaluates all detectors under trigger shift.
- `run_controls.py` - supervised probes with control negatives (test of Prop. 3).
- `run_obfuscation.py` - adversarial fine-tuning against frozen detectors; frozen vs. retrained evaluation.
- `theory_sim.py` - numerical check of the closed forms.
- `make_figures.py` - aggregates JSON into `results/summary.md`, `fig_layers_*.png`, `fig_belief_*.png`, `fig_probit.png`.

All outputs go to `../results/`. Every number in `paper.md` is taken from `results/summary.md`
or the JSON files it is built from.
