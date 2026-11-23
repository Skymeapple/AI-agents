| Regime | seeds | i.i.d. acc | i.i.d. acc (MEC-identified) | i.i.d. acc (unidentified) | Twin acc (matched) | LR ceiling | Causal surplus |
|---|---|---|---|---|---|---|---|
| obs_p0 | 2 | 74.6 ± 0.5 | 85.5 ± 0.1 | 66.0 ± 0.8 | 50.2 ± 0.0 | 50.0 ± 0.0 | 0.004 |
| obs_p05 | 1 | 77.1 | 85.9 | 70.3 | 49.2 | 50.0 | -0.015 |
| obs_p09 | 2 | 89.8 ± 0.3 | 92.2 ± 0.3 | 87.9 ± 0.3 | 49.9 ± 0.0 | 50.0 ± 0.0 | -0.003 |
| obs_p1 | 3 | 92.8 ± 0.4 | 93.1 ± 0.1 | 92.5 ± 0.6 | 50.0 ± 0.1 | 50.0 ± 0.0 | 0.000 |
| laplace_p0 | 2 | 91.9 ± 0.2 | 93.0 ± 0.8 | 91.0 ± 0.3 | 92.9 ± 0.4 | 99.8 ± 0.1 | 0.857 |
| graph_p0 | 1 | 100.0 | 100.0 | 100.0 | 100.0 | 50.0 | 1.000 |
| int32_p0 | 3 | 82.8 ± 0.6 | 90.4 ± 0.5 | 76.6 ± 1.2 | 64.1 ± 2.3 | 74.3 ± 0.7 | 0.282 |
| int128_p0 | 3 | 83.7 ± 2.0 | 89.5 ± 2.5 | 78.9 ± 1.7 | 68.5 ± 3.5 | 74.8 ± 0.1 | 0.371 |

Cross-regime probes (twin accuracy, %):

| Model | probe | twin acc | LR ceiling | extra |
|---|---|---|---|---|
| obs_p0 | twins_laplace | 49.2 ± 0.1 | 99.9 ± 0.0 |  |
| obs_p05 | twins_laplace | 48.7 | 100.0 |  |
| obs_p09 | twins_laplace | 49.3 ± 0.0 | 99.9 ± 0.1 |  |
| obs_p1 | twins_laplace | 50.0 ± 0.0 | 99.9 ± 0.0 |  |
| laplace_p0 | twins_gauss | 49.5 ± 0.0 | 50.0 ± 0.0 |  |
| graph_p0 | twins_corrupt25 | 75.0 | 50.0 |  |
| graph_p0 | twins_conflict | 0.0 | 50.0 | follows description: 100.0 |

Oracle reference accuracies on the i.i.d. test set (%):

| Regime | frac. identified | MEC-uniform oracle | MEC+label-order oracle |
|---|---|---|---|
| obs_p0 | 43.9 ± 0.0 | 80.6 ± 0.6 | 79.0 ± 0.2 |
| obs_p05 | 43.7 | 79.7 | 89.2 |
| obs_p09 | 44.8 ± 0.3 | 80.8 ± 0.2 | 97.9 ± 0.1 |
| obs_p1 | 45.3 ± 0.7 | 81.3 ± 0.2 | 100.0 ± 0.0 |

Within-pair statistics (matched twins):

| Regime | mean P(yes) on M | mean P(yes) on M' | mean abs within-pair diff | both-correct pairs |
|---|---|---|---|---|
| obs_p0 | 0.668 | 0.670 | 0.038 | 2.0 ± 0.1 |
| obs_p05 | 0.597 | 0.599 | 0.057 | 3.9 |
| obs_p09 | 0.528 | 0.530 | 0.036 | 1.2 ± 0.2 |
| obs_p1 | 0.463 | 0.463 | 0.020 | 0.9 ± 0.2 |
| laplace_p0 | 0.895 | 0.110 | 0.796 | 86.5 ± 0.9 |
| graph_p0 | 1.000 | 0.000 | 1.000 | 100.0 |
| int32_p0 | 0.766 | 0.520 | 0.292 | 31.2 ± 3.5 |
| int128_p0 | 0.782 | 0.443 | 0.375 | 39.4 ± 6.4 |

Ceiling capture kappa = (2 acc - 1) / (2 LR_ceiling - 1), per seed then mean:

| Model | probe | kappa per seed | mean |
|---|---|---|---|
| obs_p0 | twins_laplace | -0.017, -0.014 | -0.015 |
| obs_p05 | twins_laplace | -0.027 | -0.027 |
| obs_p09 | twins_laplace | -0.014, -0.015 | -0.015 |
| obs_p1 | twins_laplace | 0.002, 0.000, 0.001 | 0.001 |
| laplace_p0 | twins_matched | 0.868, 0.852 | 0.860 |
| int32_p0 | twins_matched | 0.705, 0.527, 0.501 | 0.578 |
| int128_p0 | twins_matched | 0.849, 0.849, 0.543 | 0.747 |

## Revision round 1

Bayes-optimal observational references on the seed-0 i.i.d. test items (%; results/bayes_oracles.json):

| Regime | MEC, label-free | MEC, labels | corr. R, label-free | corr. R, labels | Sigma, label-free | net (all seeds) |
|---|---|---|---|---|---|---|
| obs_p0 | 86.3 | 86.3 | 90.6 | 90.6 | 98.9 | 74.6 ± 0.5 |
| obs_p05 | 85.9 | 90.8 | 90.1 | 93.5 | 98.6 | 77.1 |
| obs_p09 | 86.5 | 98.0 | 90.2 | 98.3 | 98.8 | 89.8 ± 0.3 |
| obs_p1 | 86.4 | 100.0 | 90.6 | 100.0 | 98.7 | 92.8 ± 0.4 |

Same, restricted to MEC-unidentified queries (%):

| Regime | MEC, label-free | corr. R, label-free | corr. R, labels | Sigma, label-free | net unidentified (all seeds) |
|---|---|---|---|---|---|
| obs_p0 | 75.5 | 83.2 | 83.2 | 98.0 | 66.0 ± 0.8 |
| obs_p05 | 75.0 | 82.4 | 88.4 | 97.4 | 70.3 |
| obs_p09 | 75.4 | 82.2 | 96.9 | 97.8 | 87.9 ± 0.3 |
| obs_p1 | 75.2 | 82.8 | 100.0 | 97.7 | 92.5 ± 0.6 |

Per-seed i.i.d. accuracy (%):

| Regime | per seed |
|---|---|
| obs_p0 | s0: 74.1, s1: 75.0 |
| obs_p05 | s0: 77.1 |
| obs_p09 | s0: 89.5, s1: 90.1 |
| obs_p1 | s0: 92.3, s1: 92.7, s2: 93.2 |
| laplace_p0 | s0: 91.7, s1: 92.1 |
| graph_p0 | s0: 100.0 |
| int32_p0 | s0: 82.9, s1: 82.0, s2: 83.4 |
| int128_p0 | s0: 85.8, s1: 84.4, s2: 81.0 |

Revision twin probes (informative-evidence models; per seed; kappa with paired-bootstrap 95% CI):

| Model | seed | probe | twin acc | LR ceiling | kappa [95% CI] | adjacent acc / ceil | non-adjacent acc / ceil | acc when in-prior member is 'yes' / 'no' |
|---|---|---|---|---|---|---|---|---|
| laplace_p0 | 0 | rev_matched_redraw | 93.7 | 99.9 | 0.874 [0.855, 0.892] | - | - | - |
| laplace_p0 | 0 | rev_cov_all | 92.3 | 99.9 | 0.848 [0.829, 0.866] | 92.3 / 99.9 (n=1500) | - | 92.0 / 92.6 |
| laplace_p0 | 0 | rev_cov_inprior | 98.7 | 100.0 | 0.974 [0.966, 0.982] | 98.7 / 100.0 (n=1500) | - | 98.9 / 98.5 |
| laplace_p0 | 0 | rev_mec_all | 90.0 | 100.0 | 0.801 [0.780, 0.822] | 91.9 / 100.0 (n=1209) | 82.5 / 100.0 (n=291) | 92.6 / 87.5 |
| laplace_p0 | 0 | rev_mec_inprior | 97.3 | 100.0 | 0.946 [0.933, 0.957] | 98.9 / 100.0 (n=1209) | 90.5 / 100.0 (n=291) | 98.2 / 96.4 |
| laplace_p0 | 1 | rev_matched_redraw | 93.0 | 99.9 | 0.862 [0.844, 0.880] | - | - | - |
| laplace_p0 | 1 | rev_cov_all | 92.5 | 99.9 | 0.852 [0.832, 0.871] | 92.5 / 99.9 (n=1500) | - | 93.2 / 91.9 |
| laplace_p0 | 1 | rev_cov_inprior | 97.7 | 100.0 | 0.954 [0.943, 0.966] | 97.7 / 100.0 (n=1500) | - | 97.8 / 97.5 |
| laplace_p0 | 1 | rev_mec_all | 91.1 | 99.9 | 0.824 [0.804, 0.844] | 92.3 / 99.9 (n=1193) | 86.5 / 100.0 (n=307) | 91.6 / 90.7 |
| laplace_p0 | 1 | rev_mec_inprior | 96.7 | 100.0 | 0.933 [0.919, 0.945] | 98.8 / 100.0 (n=1243) | 86.4 / 100.0 (n=257) | 97.9 / 95.4 |
| int32_p0 | 0 | rev_matched_redraw | 65.5 | 73.6 | 0.655 [0.612, 0.698] | - | - | - |
| int32_p0 | 0 | rev_cov_all | 66.6 | 73.6 | 0.703 [0.659, 0.745] | 66.6 / 73.6 (n=1500) | - | 66.2 / 66.9 |
| int32_p0 | 0 | rev_cov_inprior | 69.3 | 74.4 | 0.792 [0.760, 0.824] | 69.3 / 74.4 (n=1500) | - | 68.9 / 69.8 |
| int32_p0 | 0 | rev_mec_all | 66.3 | 80.4 | 0.536 [0.495, 0.574] | 67.5 / 79.6 (n=1233) | 60.9 / 84.1 (n=267) | 65.7 / 66.8 |
| int32_p0 | 0 | rev_mec_inprior | 68.2 | 76.1 | 0.699 [0.660, 0.737] | 70.4 / 76.3 (n=1219) | 58.9 / 75.1 (n=281) | 68.8 / 67.7 |
| int32_p0 | 1 | rev_matched_redraw | 63.1 | 73.5 | 0.558 [0.512, 0.605] | - | - | - |
| int32_p0 | 1 | rev_cov_all | 63.4 | 73.9 | 0.558 [0.514, 0.602] | 63.4 / 73.9 (n=1500) | - | 63.7 / 63.0 |
| int32_p0 | 1 | rev_cov_inprior | 62.1 | 74.4 | 0.496 [0.449, 0.543] | 62.1 / 74.4 (n=1500) | - | 62.3 / 61.9 |
| int32_p0 | 1 | rev_mec_all | 62.8 | 80.4 | 0.421 [0.379, 0.460] | 63.4 / 80.1 (n=1208) | 60.4 / 81.8 (n=292) | 63.4 / 62.3 |
| int32_p0 | 1 | rev_mec_inprior | 61.0 | 75.1 | 0.438 [0.393, 0.484] | 61.2 / 74.8 (n=1210) | 60.3 / 76.6 (n=290) | 61.1 / 60.9 |
| int32_p0 | 2 | rev_matched_redraw | 62.6 | 73.6 | 0.532 [0.484, 0.575] | - | - | - |
| int32_p0 | 2 | rev_cov_all | 62.4 | 75.3 | 0.488 [0.443, 0.532] | 62.4 / 75.3 (n=1500) | - | 62.6 / 62.1 |
| int32_p0 | 2 | rev_cov_inprior | 61.7 | 73.9 | 0.489 [0.444, 0.534] | 61.7 / 73.9 (n=1500) | - | 61.4 / 62.0 |
| int32_p0 | 2 | rev_mec_all | 63.3 | 81.6 | 0.422 [0.384, 0.463] | 64.4 / 81.4 (n=1207) | 59.0 / 82.6 (n=293) | 65.0 / 61.8 |
| int32_p0 | 2 | rev_mec_inprior | 60.6 | 75.3 | 0.421 [0.377, 0.466] | 61.4 / 74.8 (n=1199) | 57.5 / 77.2 (n=301) | 61.5 / 59.8 |
| int128_p0 | 0 | rev_matched_redraw | 71.6 | 75.1 | 0.863 [0.831, 0.894] | - | - | - |
| int128_p0 | 0 | rev_cov_all | 71.0 | 75.4 | 0.825 [0.793, 0.858] | 71.0 / 75.4 (n=1500) | - | 72.0 / 69.8 |
| int128_p0 | 0 | rev_cov_inprior | 72.8 | 75.4 | 0.898 [0.869, 0.925] | 72.8 / 75.4 (n=1500) | - | 73.3 / 72.4 |
| int128_p0 | 0 | rev_mec_all | 69.1 | 81.2 | 0.612 [0.573, 0.651] | 70.8 / 80.8 (n=1193) | 62.4 / 82.7 (n=307) | 70.9 / 67.3 |
| int128_p0 | 0 | rev_mec_inprior | 72.1 | 76.3 | 0.840 [0.809, 0.871] | 73.7 / 75.9 (n=1259) | 63.7 / 78.2 (n=241) | 72.5 / 71.7 |
| int128_p0 | 1 | rev_matched_redraw | 71.3 | 74.9 | 0.854 [0.820, 0.887] | - | - | - |
| int128_p0 | 1 | rev_cov_all | 71.1 | 74.8 | 0.849 [0.817, 0.882] | 71.1 / 74.8 (n=1500) | - | 71.5 / 70.7 |
| int128_p0 | 1 | rev_cov_inprior | 73.1 | 75.3 | 0.916 [0.888, 0.941] | 73.1 / 75.3 (n=1500) | - | 72.9 / 73.3 |
| int128_p0 | 1 | rev_mec_all | 67.9 | 80.6 | 0.586 [0.549, 0.625] | 68.6 / 79.6 (n=1202) | 65.3 / 84.6 (n=298) | 68.8 / 67.0 |
| int128_p0 | 1 | rev_mec_inprior | 69.7 | 75.5 | 0.775 [0.739, 0.811] | 72.1 / 75.4 (n=1206) | 60.0 / 75.7 (n=294) | 70.8 / 68.7 |
| int128_p0 | 2 | rev_matched_redraw | 63.8 | 74.9 | 0.556 [0.511, 0.600] | - | - | - |
| int128_p0 | 2 | rev_cov_all | 63.4 | 75.3 | 0.530 [0.485, 0.574] | 63.4 / 75.3 (n=1500) | - | 64.9 / 61.9 |
| int128_p0 | 2 | rev_cov_inprior | 63.3 | 75.9 | 0.515 [0.472, 0.558] | 63.3 / 75.9 (n=1500) | - | 62.2 / 64.5 |
| int128_p0 | 2 | rev_mec_all | 62.0 | 80.7 | 0.392 [0.352, 0.434] | 63.3 / 80.1 (n=1215) | 56.7 / 83.0 (n=285) | 64.4 / 59.8 |
| int128_p0 | 2 | rev_mec_inprior | 62.9 | 76.3 | 0.492 [0.450, 0.533] | 64.4 / 76.6 (n=1224) | 56.5 / 74.8 (n=276) | 62.7 / 63.2 |

Kappa aggregated over seeds (mean ± s.d. of per-seed kappa):

| Model | probe | seeds | kappa |
|---|---|---|---|
| laplace_p0 | twins_matched | 2 | 0.860 ± 0.008 |
| laplace_p0 | rev_cov_all | 2 | 0.850 ± 0.002 |
| laplace_p0 | rev_cov_inprior | 2 | 0.964 ± 0.010 |
| laplace_p0 | rev_mec_all | 2 | 0.812 ± 0.012 |
| laplace_p0 | rev_mec_inprior | 2 | 0.940 ± 0.006 |
| int32_p0 | twins_matched | 3 | 0.578 ± 0.091 |
| int32_p0 | rev_cov_all | 3 | 0.583 ± 0.089 |
| int32_p0 | rev_cov_inprior | 3 | 0.592 ± 0.141 |
| int32_p0 | rev_mec_all | 3 | 0.459 ± 0.054 |
| int32_p0 | rev_mec_inprior | 3 | 0.520 ± 0.127 |
| int128_p0 | twins_matched | 3 | 0.747 ± 0.144 |
| int128_p0 | rev_cov_all | 3 | 0.735 ± 0.145 |
| int128_p0 | rev_cov_inprior | 3 | 0.776 ± 0.185 |
| int128_p0 | rev_mec_all | 3 | 0.530 ± 0.098 |
| int128_p0 | rev_mec_inprior | 3 | 0.702 ± 0.151 |

Prior-agnostic classical reasoners on covered-edge twins (results/rev_baselines.json):

| Setting | twin acc | LR ceiling | kappa [95% CI] |
|---|---|---|---|
| laplace_all | 99.3 | 99.9 | 0.989 [0.984, 0.995] |
| laplace_inprior | 99.9 | 100.0 | 0.998 [0.995, 1.000] |
| int32_all | 71.6 | 73.9 | 0.904 [0.878, 0.929] |
| int32_inprior | 73.2 | 74.5 | 0.948 [0.930, 0.966] |
| int128_all | 73.9 | 75.2 | 0.948 [0.932, 0.964] |
| int128_inprior | 73.0 | 74.5 | 0.941 [0.924, 0.958] |

Cue-removal probe for saved label-order models (results/cue_removal.json):

| Model | i.i.d. acc, random labels | identified | unidentified | i.i.d. acc, reversed labels |
|---|---|---|---|---|
| obs_p09_s1 | 59.9 | 72.5 | 49.9 | 26.8 |
| obs_p1_s2 | 59.7 | 72.4 | 49.7 | 26.0 |
