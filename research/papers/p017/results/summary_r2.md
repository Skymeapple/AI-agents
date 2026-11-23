# Round-2 summary (all numbers computed by code/aggregate_r2.py from raw outputs)

## 1. Exact finite-time ceiling (review M4)

| world | seed | Acc*_1 | Acc*_2 | Acc*_4 | Acc*_8 | Acc*_12 | Acc*_16 | Acc*_24 | max truncation excess (all encoders, k) |
|---|---|---|---|---|---|---|---|---|---|
| unique | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.0000 |
| unique | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.0000 |
| unique | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +0.0000 |
| balanced2 | 0 | 0.090 | 0.181 | 0.474 | 0.857 | 0.972 | 0.996 | 1.000 | -0.0018 |
| balanced2 | 1 | 0.090 | 0.180 | 0.482 | 0.884 | 0.982 | 0.998 | 1.000 | -0.0048 |
| balanced2 | 2 | 0.090 | 0.181 | 0.481 | 0.874 | 0.980 | 0.998 | 1.000 | -0.0039 |
| landmark | 0 | 0.090 | 0.138 | 0.222 | 0.438 | 0.704 | 0.892 | 1.000 | -0.0030 |
| landmark | 1 | 0.091 | 0.177 | 0.301 | 0.506 | 0.714 | 0.879 | 1.000 | +0.0022 |
| landmark | 2 | 0.088 | 0.149 | 0.246 | 0.427 | 0.642 | 0.842 | 1.000 | +0.0002 |

All truncation cells: n = 324, max excess of truncated-probe accuracy over exact Acc*_k = +0.0022, number of cells with excess > 0.005: 0

k_eff against the exact ceiling (main-table models):

- unique_s0: gru_trained: 1, gru_random: 1, esn_l0.9_h64: 1, esn_l0.99_h512: 1, tf_trained: 1, tf_random: 1
- unique_s1: gru_trained: 1, gru_random: 1, esn_l0.9_h64: 1, esn_l0.99_h512: 1
- unique_s2: gru_trained: 1, gru_random: 1, esn_l0.9_h64: 1, esn_l0.99_h512: 1
- balanced2_s0: gru_trained: 20, gru_random: 2, esn_l0.9_h64: 2, esn_l0.99_h512: 2, tf_trained: 8, tf_random: 1
- balanced2_s1: gru_trained: 5, gru_random: 2, esn_l0.9_h64: 2, esn_l0.99_h512: 2
- balanced2_s2: gru_trained: 8, gru_random: 2, esn_l0.9_h64: 2, esn_l0.99_h512: 2
- landmark_s0: gru_trained: 5, gru_random: 2, esn_l0.9_h64: 2, esn_l0.99_h512: 2, tf_trained: 10, tf_random: 1
- landmark_s1: gru_trained: 22, gru_random: 2, esn_l0.9_h64: 2, esn_l0.99_h512: 2
- landmark_s2: gru_trained: 15, gru_random: 2, esn_l0.9_h64: 2, esn_l0.99_h512: 2

Training-dynamics checkpoints (epoch, probe acc, exact k_eff):

- balanced2_s0: ep1: 0.208/3; ep2: 0.431/4; ep5: 0.786/7; ep10: 1.000/21; ep20: 1.000/23; ep30: 1.000/23; ep40: 1.000/23
- landmark_s0: ep1: 0.133/2; ep2: 0.158/3; ep5: 0.338/7; ep10: 0.403/8; ep20: 0.997/22; ep30: 0.999/23; ep40: 1.000/23
- landmark_s1: ep1: 0.206/3; ep2: 0.240/3; ep5: 0.763/13; ep10: 0.998/24; ep20: 1.000/24; ep30: 1.000/24; ep40: 1.000/24
- landmark_s2: ep1: 0.167/3; ep2: 0.205/4; ep5: 0.362/7; ep10: 0.999/24; ep20: 0.999/24; ep30: 0.999/24; ep40: 0.999/24

## 2. Trained GRU, 10 seeds per world and budget (one-cycle schedule; review M5)

| world | epochs | n | next-obs acc | probe acc | k_eff (per seed) | #seeds k_eff ≥ 12 | CRA probe | CRA random | CRA full | CRA none |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced2 | 10 | 10 | 0.961 ± 0.042 | 0.840 ± 0.151 | 20, 5, 8, 12, 6, 11, 11, 14, 11, 8 | 3 | 0.650 ± 0.090 | 0.386 ± 0.047 | 0.908 ± 0.091 | 0.093 ± 0.093 |
| balanced2 | 30 | 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 23, 20, 21, 23, 21, 24, 22, 23, 23, 24 | 10 | 0.847 ± 0.053 | 0.337 ± 0.050 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| landmark | 10 | 10 | 0.959 ± 0.018 | 0.537 ± 0.196 | 5, 22, 15, 10, 7, 8, 9, 10, 11, 10 | 2 | 0.502 ± 0.088 | 0.411 ± 0.040 | 0.801 ± 0.106 | 0.191 ± 0.099 |
| landmark | 30 | 10 | 0.996 ± 0.006 | 0.960 ± 0.061 | 15, 24, 23, 17, 24, 23, 24, 17, 24, 23 | 10 | 0.770 ± 0.098 | 0.332 ± 0.072 | 0.978 ± 0.042 | 0.022 ± 0.041 |

Per-seed (probe acc / k_eff / CRA probe / CRA random / next-obs / landmark recall):

- balanced2 e10 s0: 1.000 / 20 / 0.864 / 0.377 / 1.000 / nan
- balanced2 e10 s1: 0.587 / 5 / 0.510 / 0.424 / 0.873 / nan
- balanced2 e10 s2: 0.825 / 8 / 0.633 / 0.426 / 0.954 / nan
- balanced2 e10 s3: 0.967 / 12 / 0.711 / 0.373 / 0.987 / nan
- balanced2 e10 s4: 0.601 / 6 / 0.551 / 0.460 / 0.889 / nan
- balanced2 e10 s5: 0.707 / 11 / 0.623 / 0.430 / 0.976 / nan
- balanced2 e10 s6: 0.942 / 11 / 0.679 / 0.358 / 0.984 / nan
- balanced2 e10 s7: 0.986 / 14 / 0.646 / 0.298 / 0.995 / nan
- balanced2 e10 s8: 0.963 / 11 / 0.655 / 0.328 / 0.987 / nan
- balanced2 e10 s9: 0.818 / 8 / 0.624 / 0.385 / 0.965 / nan
- balanced2 e30 s0: 1.000 / 23 / 0.927 / 0.337 / 1.000 / nan
- balanced2 e30 s1: 1.000 / 20 / 0.777 / 0.363 / 1.000 / nan
- balanced2 e30 s2: 1.000 / 21 / 0.800 / 0.406 / 1.000 / nan
- balanced2 e30 s3: 1.000 / 23 / 0.854 / 0.335 / 1.000 / nan
- balanced2 e30 s4: 1.000 / 21 / 0.830 / 0.433 / 1.000 / nan
- balanced2 e30 s5: 1.000 / 24 / 0.919 / 0.333 / 1.000 / nan
- balanced2 e30 s6: 1.000 / 22 / 0.801 / 0.285 / 1.000 / nan
- balanced2 e30 s7: 1.000 / 23 / 0.920 / 0.320 / 1.000 / nan
- balanced2 e30 s8: 1.000 / 23 / 0.825 / 0.278 / 1.000 / nan
- balanced2 e30 s9: 1.000 / 24 / 0.818 / 0.278 / 1.000 / nan
- landmark e10 s0: 0.250 / 5 / 0.378 / 0.475 / 0.937 / 0.607
- landmark e10 s1: 0.987 / 22 / 0.660 / 0.331 / 0.998 / 0.995
- landmark e10 s2: 0.754 / 15 / 0.534 / 0.359 / 0.982 / 0.938
- landmark e10 s3: 0.538 / 10 / 0.514 / 0.415 / 0.957 / 0.763
- landmark e10 s4: 0.373 / 7 / 0.416 / 0.445 / 0.935 / 0.565
- landmark e10 s5: 0.394 / 8 / 0.439 / 0.427 / 0.962 / 0.815
- landmark e10 s6: 0.475 / 9 / 0.474 / 0.407 / 0.951 / 0.751
- landmark e10 s7: 0.548 / 10 / 0.601 / 0.391 / 0.952 / 0.733
- landmark e10 s8: 0.519 / 11 / 0.585 / 0.426 / 0.964 / 0.891
- landmark e10 s9: 0.530 / 10 / 0.417 / 0.438 / 0.947 / 0.692
- landmark e30 s0: 0.822 / 15 / 0.649 / 0.410 / 0.979 / 0.847
- landmark e30 s1: 1.000 / 24 / 0.782 / 0.336 / 1.000 / 1.000
- landmark e30 s2: 0.994 / 23 / 0.682 / 0.452 / 1.000 / 0.998
- landmark e30 s3: 0.895 / 17 / 0.656 / 0.389 / 0.993 / 0.967
- landmark e30 s4: 1.000 / 24 / 0.908 / 0.262 / 1.000 / 1.000
- landmark e30 s5: 0.998 / 23 / 0.817 / 0.368 / 1.000 / 1.000
- landmark e30 s6: 0.999 / 24 / 0.816 / 0.207 / 1.000 / 1.000
- landmark e30 s7: 0.897 / 17 / 0.659 / 0.339 / 0.992 / 0.958
- landmark e30 s8: 0.997 / 24 / 0.923 / 0.270 / 1.000 / 0.999
- landmark e30 s9: 0.998 / 23 / 0.809 / 0.287 / 1.000 / 1.000

Independent runs (n = 40; each a separately trained GRU): Pearson r(probe acc, CRA gain) = 0.880, 95% bootstrap CI [0.805, 0.928], Spearman ρ = 0.902. Runs with probe < 0.4: 3, of which gain ≤ 0: 2. Runs with probe ≥ 0.75: 29, of which gain ≥ 0.17: 29; min gain among them: 0.175.
CRA(probe) range over runs with k_eff ≥ 12: 0.534–0.927; CRA(random) over the same runs: 0.207–0.452
