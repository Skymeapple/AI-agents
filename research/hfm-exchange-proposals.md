# Proposals to post in HFM's Research Exchange

HFM's site lets a signed-in person (Novix account) join each challenge's
"Research Exchange" conversation, propose new challenges, and offer review
time, data or compute. HFM's agents run the experiments and its private
evaluators score them. Posting these proposals is therefore the one route
by which the ideas in [`human-frontier-mission.md`](human-frontier-mission.md)
can be tested on HFM's real scoreboard.

Each proposal is written as a testable hypothesis in the style the agents
use. None has been posted. Posting needs your account and your decision.

| Challenge (current best) | Proposal to post |
| --- | --- |
| 24-Hour Weather with ArchesWeather (0.097486) | Fine-tuning ArchesWeather-M on all public years with a surface-weighted loss and a 4 × 6 h autoregressive rollout objective, at a larger budget class than a post-processing edit, will lower area-weighted normalised RMSE by at least 1% relative to the current post-processed recipe. The earlier fine-tune used only 2000–2009. |
| Photometric Redshift (0.02434) | Replacing point regression with a redshift probability distribution (classification over redshift bins, median as the estimate), with Galactic extinction and survey depth as inputs, will lower NRMSE on unseen sky regions by reducing catastrophic outliers. |
| Regulatory DNA with Evo 2 (0.7076) | One multi-task model with shared sequence layers and a head per cell line, with reverse-complement averaging, will raise assay macro-Spearman, with the largest gain on WTC11. |
| Lung Cell Atlas annotation (0.8536) | Adding a count-based second view (highly variable genes, scANVI-style) and decoding over the Cell Ontology hierarchy with parent back-off will raise study macro-F1 over the frozen-UCE pseudo-label recipe. |
| Crystal Generation (0.4395) | Generating 20–50 times the 64-structure budget and pre-screening with single-point MatterSim energies, symmetry and duplicate filters before spending the budget will raise the unique-novel-stable fraction more than exploring further formulas per parent. |
| Cold Battery Voltage (0.599) | Replacing fitted thermal slopes with Arrhenius-form resistance and diffusion terms (one activation energy per process) will raise skill on the coldest assessment cycles, because the functional form extrapolates below the calibration range. |
| Wave Energy (0.560) | Modelling wind-sea and swell separately, with swell as a persistent process whose statistics come from the campaign and not from local wind, will raise skill over single-regression sea-state models. |
| Steel Cyclic Transfer (0.668) and Limited Tests (0.545) | A hierarchical prior over Chaboche parameters learned from the donor steels and conditioned on monotonic-test features (yield, ultimate strength, plateau length), with posterior-mean prediction, will beat donor averaging on held-out steels and protocols. |
| Rayleigh–Bénard reconstruction (0.2342) | Computing pressure from the reconstructed velocity through the pressure-Poisson equation, in place of the regression for pressure, will lower field NRMSE; a learned Voronoi-input network for the measured fields should then lower it further. |
| Wind-Farm Wake Calibration (0.412) | Correcting per-turbine wind-direction bias and conditioning wake expansion on turbulence intensity before fitting any residual model will raise reward on the unseen farm more than per-turbine boosted residuals. |
| Single-Track Railway Dispatcher (0.398) | A rolling-horizon exact optimisation of meet and pass orders over a 10–20 minute look-ahead will outperform the pairwise veto rules. |
| Dam Spill Optimisation (0.720) | Jointly optimising spill at all eight dams over the fish travel horizon, with dissolved-gas limits as hard constraints, will outperform per-dam action selection. |
| Filing-Event Equity, A-Share Overlay, Order-Book Execution (all 1.0) | The reward is at its ceiling and can no longer rank methods. Request an uncapped net-of-cost score on a later out-of-sample window before further search. |

A caution from our own test of the optical-constants idea: on public data,
Kramers–Kronig consistency alone gave only a small, fragile gain. It is
left out of this list for that reason.
