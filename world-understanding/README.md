# How do learning machines come to understand a complex world?

A self-contained computational study. It maps eight explanatory frameworks for
how AI systems come to model the world, then tests each one in a small
**model world whose ground truth is known exactly**. That way every claim is
scored against an optimum or a theoretical prediction rather than intuition.
The output is an academic paper: [`paper/paper.pdf`](paper/paper.pdf)
(source in `paper/*.md`, HTML in `paper/paper.html`).

| | Framework | Model world | Headline result |
|---|---|---|---|
| E1 | Prediction = compression; emergent world models | Egocentric navigation in an aliased grid, exact Bayes filter | Transformer closes 85% of the n-gram→Bayes gap; hidden state decodable (53% vs ≤ 5% untrained); editing it steers predictions (89% vs 3% random) |
| E2 | Manifold hypothesis | Observations in ℝ⁶⁴ from d latent factors | Learning-curve exponent ≈ 4.9/d (R² 0.96), independent of ambient dimension |
| E3 | Scaling laws from a heavy-tailed world | Zipf-distributed facts | Data exponents match −(α−1)/α to within 0.02; capacity scaling is frequency-ordered but optimisation-limited |
| E4 | Superposition | Sparse features through a bottleneck | Up to 18.7 features in 5 dimensions; linear model never exceeds 5 |
| E5 | Algorithmic compression (grokking) | a + b mod 97 | See paper §4.5 |
| E6 | In-context Bayesian inference | Linear worlds with controlled task diversity | Below a diversity threshold: memorised-pool Bayes; above it: ridge-like generalisation |
| E7 | Limits: shortcut learning | Digits with a planted watermark | Reversed-cue accuracy 98% → 0% as correlation → 1; only decorrelation restores it |

## Running it

Python ≥ 3.11 with `jax`, `optax`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`
(and `pytest` for the tests). Building the PDF also needs `pandoc` and Node with
Playwright.

```bash
cd world-understanding
python -m pytest -q tests          # exact references and estimators
python -m wu.e1_world_model 0      # one module per experiment; --quick for a smoke test
python -m wu.plots                 # all figures -> paper/figures/
python -m wu.summary               # every number quoted in the paper
python paper/build.py              # paper.md -> paper.html -> paper.pdf
```

Everything runs on a 4-core CPU in a few hours. Raw results are in `results/*.json`.

## How much to trust it

The evidential standard is stated in the paper: pre-stated falsifiers, untrained,
random and linear controls, three seeds where affordable, and analytic references.
Its limits are stated there too, and are worth repeating here:

- These are **small models in small worlds**. They establish mechanisms, not magnitudes at scale.
- **Two results did not come out as the textbook says**, and the paper reports them as such.
  The capacity-scaling exponents in E3 do not track the Zipf exponent, because training is
  optimisation-limited. E2's 1/d law appears only once the optimisation floor is modelled;
  the naive fit does not show it.
- **E6 has one or two seeds per condition.** The memorise/generalise contrast replicates across
  seeds; the exact threshold is not resolved.
- Decisions made after seeing pilot data are listed in Appendix A of the paper.
