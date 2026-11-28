| model / floor | all | empty | occupied | age 0 | age 1-3 | age 4-7 | age 8-15 | age 16+ |
|---|---|---|---|---|---|---|---|---|
| trained (6000 steps), linear probe L4, legal top-1 0.924 | 0.812 | 0.998 | 0.654 | 0.919 | 0.614 | 0.631 | 0.630 | 0.631 |
| trained (6000 steps), MLP probe (last layer) | 0.835 | 0.999 | 0.694 | 0.921 | 0.664 | 0.677 | 0.674 | 0.666 |
| untrained twin, linear probe L4, legal top-1 0.148 | 0.698 | 0.926 | 0.504 | 0.951 | 0.474 | 0.450 | 0.447 | 0.459 |
| untrained twin, MLP probe (last layer) | 0.736 | 0.959 | 0.545 | 0.967 | 0.510 | 0.499 | 0.495 | 0.502 |
| floor: t only | 0.621 | 0.885 | 0.396 | 0.649 | 0.257 | 0.384 | 0.400 | 0.439 |
| floor: last 1 moves + t | 0.651 | 0.885 | 0.451 | 0.932 | 0.405 | 0.382 | 0.393 | 0.429 |
| floor: last 4 moves + t | 0.719 | 0.890 | 0.573 | 0.870 | 0.843 | 0.427 | 0.432 | 0.469 |
| floor: last 8 moves + t | 0.767 | 0.905 | 0.650 | 0.845 | 0.800 | 0.808 | 0.443 | 0.482 |
| floor: last 16 moves + t | 0.818 | 0.934 | 0.719 | 0.833 | 0.786 | 0.779 | 0.765 | 0.475 |
| floor: placement + t | 0.879 | 1.000 | 0.776 | 0.790 | 0.733 | 0.764 | 0.786 | 0.818 |

Counts (square-positions): all: 7678912, empty: 3540000, occupied: 4138912, age 0: 388349, age 1-3: 936742, age 4-7: 901561, age 8-15: 1050404, age 16+: 861856
