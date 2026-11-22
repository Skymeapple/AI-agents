| Regime | seeds | i.i.d. acc | i.i.d. acc (MEC-identified) | i.i.d. acc (unidentified) | Twin acc (matched) | LR ceiling | Causal surplus |
|---|---|---|---|---|---|---|---|
| obs_p0 | 2 | 74.6 ± 0.5 | 85.5 ± 0.1 | 66.0 ± 0.8 | 50.2 ± 0.0 | 50.0 ± 0.0 | 0.004 |
| obs_p05 | 1 | 77.1 | 85.9 | 70.3 | 49.2 | 50.0 | -0.015 |
| obs_p09 | 1 | 89.5 | 92.0 | 87.6 | 49.9 | 50.0 | -0.002 |
| obs_p1 | 2 | 92.5 ± 0.2 | 93.0 ± 0.0 | 92.1 ± 0.3 | 50.0 ± 0.1 | 50.0 ± 0.0 | 0.001 |
| laplace_p0 | 2 | 91.9 ± 0.2 | 93.0 ± 0.8 | 91.0 ± 0.3 | 92.9 ± 0.4 | 99.8 ± 0.1 | 0.857 |
| graph_p0 | 1 | 100.0 | 100.0 | 100.0 | 100.0 | 50.0 | 1.000 |
| int32_p0 | 2 | 82.5 ± 0.5 | 90.2 ± 0.5 | 76.2 ± 1.3 | 65.3 ± 2.0 | 74.8 ± 0.4 | 0.305 |
| int128_p0 | 1 | 85.8 | 91.9 | 80.6 | 71.2 | 74.9 | 0.423 |

Cross-regime probes (twin accuracy, %):

| Model | probe | twin acc | LR ceiling | extra |
|---|---|---|---|---|
| obs_p0 | twins_laplace | 49.2 ± 0.1 | 99.9 ± 0.0 |  |
| obs_p05 | twins_laplace | 48.7 | 100.0 |  |
| obs_p09 | twins_laplace | 49.3 | 99.8 |  |
| obs_p1 | twins_laplace | 50.0 ± 0.1 | 99.9 ± 0.0 |  |
| laplace_p0 | twins_gauss | 49.5 ± 0.0 | 50.0 ± 0.0 |  |
| graph_p0 | twins_corrupt25 | 75.0 | 50.0 |  |
| graph_p0 | twins_conflict | 0.0 | 50.0 | follows description: 100.0 |

Oracle reference accuracies on the i.i.d. test set (%):

| Regime | frac. identified | MEC-uniform oracle | MEC+label-order oracle |
|---|---|---|---|
| obs_p0 | 43.9 ± 0.0 | 80.6 ± 0.6 | 79.0 ± 0.2 |
| obs_p05 | 43.7 | 79.7 | 89.2 |
| obs_p09 | 45.1 | 80.6 | 97.8 |
| obs_p1 | 45.7 ± 0.4 | 81.2 ± 0.1 | 100.0 ± 0.0 |

Within-pair statistics (matched twins):

| Regime | mean P(yes) on M | mean P(yes) on M' | mean abs within-pair diff | both-correct pairs |
|---|---|---|---|---|
| obs_p0 | 0.668 | 0.670 | 0.038 | 2.0 ± 0.1 |
| obs_p05 | 0.597 | 0.599 | 0.057 | 3.9 |
| obs_p09 | 0.517 | 0.520 | 0.036 | 1.5 |
| obs_p1 | 0.466 | 0.465 | 0.021 | 1.1 ± 0.1 |
| laplace_p0 | 0.895 | 0.110 | 0.796 | 86.5 ± 0.9 |
| graph_p0 | 1.000 | 0.000 | 1.000 | 100.0 |
| int32_p0 | 0.768 | 0.503 | 0.307 | 33.0 ± 2.8 |
| int128_p0 | 0.794 | 0.406 | 0.420 | 44.5 |

Ceiling capture kappa = (2 acc - 1) / (2 LR_ceiling - 1), per seed then mean:

| Model | probe | kappa per seed | mean |
|---|---|---|---|
| obs_p0 | twins_laplace | -0.017, -0.014 | -0.015 |
| obs_p05 | twins_laplace | -0.027 | -0.027 |
| obs_p09 | twins_laplace | -0.014 | -0.014 |
| obs_p1 | twins_laplace | 0.002, 0.000 | 0.001 |
| laplace_p0 | twins_matched | 0.868, 0.852 | 0.860 |
| int32_p0 | twins_matched | 0.705, 0.527 | 0.616 |
| int128_p0 | twins_matched | 0.849 | 0.849 |
