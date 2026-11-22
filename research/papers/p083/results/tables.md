
### Table: carbon-only shifting, phi=0.3, 24h window, tower

| Site (BA) | on-site share | mean within-day rho(C,W) | rho(C,W_on) | rho(C,W_off) | dC % | dW total % | dW on-site (% of on-site) | water-neutral knee dC % | carbon retained |
|---|---|---|---|---|---|---|---|---|---|
| Ashburn VA | 0.48 | 0.19 | 0.31 | -0.30 | -0.90 | -0.20 | -0.7 | -0.90 | 1.000 |
| Dallas TX | 0.63 | 0.69 | 0.02 | 0.91 | -2.85 | -0.79 | -0.1 | -2.85 | 1.000 |
| San Jose CA | 0.69 | 0.65 | -0.63 | 0.94 | -4.73 | -0.75 | +1.1 | -4.73 | 1.000 |
| Phoenix AZ | 0.56 | 0.61 | -0.65 | 0.96 | -2.42 | -0.65 | +0.9 | -2.42 | 1.000 |
| The Dalles OR | 0.78 | 0.29 | -0.15 | 0.91 | -2.34 | -0.26 | +0.4 | -2.34 | 1.000 |
| Council Bluffs IA | 0.50 | 0.66 | 0.31 | 0.77 | -1.59 | -0.79 | -0.8 | -1.59 | 1.000 |
| Pryor OK | 0.58 | 0.85 | 0.37 | 0.98 | -3.83 | -1.72 | -0.8 | -3.83 | 1.000 |
| Atlanta GA | 0.52 | 0.29 | -0.05 | 0.29 | -1.02 | -0.16 | -0.0 | -1.02 | 1.000 |
| Memphis TN | 0.47 | 0.25 | 0.02 | 0.26 | -1.10 | -0.25 | -0.1 | -1.10 | 1.000 |
| Las Vegas NV | 0.64 | 0.84 | -0.63 | 0.98 | -4.87 | -1.00 | +0.9 | -4.87 | 1.000 |
| Salt Lake UT | 0.49 | 0.89 | -0.54 | 1.00 | -3.50 | -1.41 | +0.9 | -3.50 | 1.000 |
| Charlotte NC | 0.43 | -0.12 | 0.15 | -0.25 | -2.77 | +0.14 | -0.3 | -2.72 | 0.985 |
| Boston MA | 0.54 | -0.29 | 0.15 | -0.48 | -1.30 | +0.21 | -0.2 | -1.24 | 0.958 |

### Table: carbon-only shifting, phi=0.3, 24h window, adiab

| Site (BA) | on-site share | mean within-day rho(C,W) | rho(C,W_on) | rho(C,W_off) | dC % | dW total % | dW on-site (% of on-site) | water-neutral knee dC % | carbon retained |
|---|---|---|---|---|---|---|---|---|---|
| Ashburn VA | 0.05 | 0.05 | 0.20 | -0.30 | -0.90 | -0.70 | -19.2 | -0.90 | 1.000 |
| Dallas TX | 0.22 | 0.69 | 0.12 | 0.91 | -2.85 | -2.76 | -5.6 | -2.85 | 1.000 |
| San Jose CA | 0.05 | 0.73 | -0.19 | 0.94 | -4.73 | -3.35 | +24.9 | -4.73 | 1.000 |
| Phoenix AZ | 0.31 | 0.07 | -0.48 | 0.96 | -2.42 | +1.08 | +9.4 | -2.29 | 0.946 |
| The Dalles OR | 0.19 | 0.52 | -0.13 | 0.91 | -2.34 | +1.29 | +17.6 | -2.31 | 0.989 |
| Council Bluffs IA | 0.06 | 0.79 | 0.21 | 0.77 | -1.59 | -1.85 | -19.9 | -1.59 | 1.000 |
| Pryor OK | 0.12 | 0.91 | 0.28 | 0.98 | -3.83 | -4.54 | -15.4 | -3.83 | 1.000 |
| Atlanta GA | 0.07 | 0.32 | 0.04 | 0.29 | -1.02 | -0.58 | -4.0 | -1.02 | 1.000 |
| Memphis TN | 0.07 | 0.34 | 0.03 | 0.26 | -1.10 | -0.60 | -3.2 | -1.10 | 1.000 |
| Las Vegas NV | 0.29 | 0.43 | -0.30 | 0.98 | -4.87 | -0.29 | +9.8 | -4.87 | 1.000 |
| Salt Lake UT | 0.08 | 0.75 | -0.19 | 1.00 | -3.50 | -2.27 | +14.5 | -3.50 | 1.000 |
| Charlotte NC | 0.04 | 0.12 | 0.23 | -0.25 | -2.77 | -0.36 | -18.1 | -2.77 | 1.000 |
| Boston MA | 0.02 | -0.35 | 0.07 | -0.48 | -1.30 | +0.37 | -15.3 | -1.25 | 0.966 |

Exchange rates at penalty sites (pp water recovered per pp carbon given up):
              site   arch  xrate
45    Charlotte NC  tower   3.29
49       Boston MA  tower   3.76
65      Phoenix AZ  adiab   8.57
69   The Dalles OR  adiab  53.27
101      Boston MA  adiab   8.84

### MC summary

| Site | arch | P(dW>0) | median dW % | 5-95% dW | P(dW_on>0) | median dW_on % | median retained |
|---|---|---|---|---|---|---|---|
| Ashburn VA | adiab | 0.00 | -1.84 | [-5.69, -0.30] | 0.00 | -18.2 | 1.000 |
| Ashburn VA | tower | 0.00 | -0.94 | [-3.68, -0.08] | 0.00 | -0.7 | 1.000 |
| Atlanta GA | adiab | 0.00 | -1.02 | [-2.38, -0.29] | 0.00 | -3.2 | 1.000 |
| Atlanta GA | tower | 0.01 | -0.50 | [-1.61, -0.08] | 0.41 | -0.0 | 1.000 |
| Boston MA | adiab | 0.86 | +0.29 | [-0.46, +0.81] | 0.00 | -14.0 | 0.993 |
| Boston MA | tower | 0.89 | +0.21 | [-0.12, +0.53] | 0.00 | -0.2 | 0.991 |
| Charlotte NC | adiab | 0.06 | -2.65 | [-8.98, +0.02] | 0.00 | -16.8 | 1.000 |
| Charlotte NC | tower | 0.18 | -1.63 | [-6.85, +0.32] | 0.02 | -0.2 | 1.000 |
| Council Bluffs IA | adiab | 0.00 | -2.39 | [-4.33, -0.87] | 0.00 | -19.7 | 1.000 |
| Council Bluffs IA | tower | 0.00 | -1.14 | [-2.52, -0.41] | 0.00 | -0.7 | 1.000 |
| Dallas TX | adiab | 0.00 | -3.07 | [-4.93, -1.21] | 0.00 | -5.2 | 1.000 |
| Dallas TX | tower | 0.00 | -0.89 | [-1.67, -0.35] | 0.42 | -0.0 | 1.000 |
| Las Vegas NV | adiab | 0.14 | -1.74 | [-6.51, +0.46] | 1.00 | +9.8 | 1.000 |
| Las Vegas NV | tower | 0.00 | -1.74 | [-5.03, -0.48] | 1.00 | +0.9 | 1.000 |
| Memphis TN | adiab | 0.67 | +1.63 | [-1.02, +4.81] | 0.01 | -2.5 | 0.934 |
| Memphis TN | tower | 0.67 | +1.28 | [-0.40, +4.16] | 0.31 | -0.0 | 0.928 |
| Phoenix AZ | adiab | 0.85 | +0.50 | [-0.28, +2.40] | 1.00 | +7.2 | 0.980 |
| Phoenix AZ | tower | 0.00 | -0.49 | [-1.35, -0.11] | 1.00 | +0.7 | 1.000 |
| Pryor OK | adiab | 0.00 | -4.73 | [-7.59, -1.72] | 0.00 | -14.6 | 1.000 |
| Pryor OK | tower | 0.00 | -2.17 | [-3.89, -0.79] | 0.00 | -0.5 | 1.000 |
| Salt Lake UT | adiab | 0.00 | -1.97 | [-4.06, -0.62] | 1.00 | +14.4 | 1.000 |
| Salt Lake UT | tower | 0.00 | -1.37 | [-2.62, -0.47] | 1.00 | +0.9 | 1.000 |
| San Jose CA | adiab | 0.00 | -5.46 | [-11.17, -1.49] | 1.00 | +25.8 | 1.000 |
| San Jose CA | tower | 0.01 | -3.74 | [-9.42, -0.31] | 1.00 | +1.2 | 1.000 |
| The Dalles OR | adiab | 0.96 | +0.70 | [+0.11, +2.67] | 1.00 | +15.0 | 0.960 |
| The Dalles OR | tower | 0.67 | +0.35 | [-0.38, +1.09] | 1.00 | +0.3 | 0.975 |

### Accounting bias

| Site | arch | carbon % | on-site water % | total water % |
|---|---|---|---|---|
| Ashburn VA | tower | -0.42 | -1.20 | -0.33 |
| Dallas TX | tower | -0.32 | -0.91 | -0.44 |
| San Jose CA | tower | +2.43 | -1.22 | -0.01 |
| Phoenix AZ | tower | +1.64 | -0.89 | +0.21 |
| The Dalles OR | tower | +0.83 | -1.18 | -0.67 |
| Council Bluffs IA | tower | -0.75 | -1.19 | -0.65 |
| Pryor OK | tower | -1.83 | -1.12 | -1.19 |
| Atlanta GA | tower | +0.15 | -0.98 | -0.04 |
| Memphis TN | tower | +0.09 | -0.92 | +0.20 |
| Las Vegas NV | tower | +2.72 | -0.95 | +0.31 |
| Salt Lake UT | tower | +1.84 | -1.06 | +0.43 |
| Charlotte NC | tower | -0.64 | -1.12 | -0.09 |
| Boston MA | tower | -0.43 | -0.63 | +0.01 |
| Ashburn VA | adiab | -0.42 | -16.62 | -0.55 |
| Dallas TX | adiab | -0.32 | -9.61 | -2.00 |
| San Jose CA | adiab | +2.43 | -19.90 | +1.30 |
| Phoenix AZ | adiab | +1.64 | -7.85 | -1.46 |
| The Dalles OR | adiab | +0.83 | -17.96 | -3.07 |
| Council Bluffs IA | adiab | -0.75 | -16.85 | -1.21 |
| Pryor OK | adiab | -1.83 | -13.33 | -2.91 |
| Atlanta GA | adiab | +0.15 | -14.68 | -0.34 |
| Memphis TN | adiab | +0.09 | -13.39 | +0.05 |
| Las Vegas NV | adiab | +2.72 | -9.59 | -1.23 |
| Salt Lake UT | adiab | +1.84 | -14.78 | +0.41 |
| Charlotte NC | adiab | -0.64 | -16.26 | -0.21 |
| Boston MA | adiab | -0.43 | -15.95 | +0.34 |

hours per site 8754 8754
days: 363 364
