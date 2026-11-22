"""Monte-Carlo sensitivity of the fleet results to the uncertain parameters.
Uniform draws over data.RANGES; outputs results/montecarlo.csv and montecarlo_summary.json
"""
import json, os
import numpy as np
from scipy.stats import spearmanr
import data as D
from fleet import run_fleet

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
N = 2000


def main(seed=20261122):
    rng = np.random.default_rng(seed)
    keys = list(D.RANGES)
    draws, outs = [], []
    for i in range(N):
        p = dict(D.BASE)
        for k in keys:
            lo, hi = D.RANGES[k]; p[k] = rng.uniform(lo, hi)
        try:
            rows_p, srv = run_fleet(p, 'plateau', years_out=[2026, 2028, 2030])
            rows_b, _ = run_fleet(p, 'boom', years_out=[2028])
        except ValueError:
            continue
        r26, r28, r30 = rows_p
        outs.append([srv.T, r26['gap'], r26['gap'] / r26['dep_acc'], r26['value_gap'], r26['req_rent'],
                     r28['gap'], r30['gap'], rows_b[0]['gap'], r30['dep_acc'] / r26['dep_acc']])
        draws.append([p[k] for k in keys])
    draws, outs = np.array(draws), np.array(outs)
    names = ['econ_life', 'gap2026', 'gap_ratio2026', 'value_gap2026', 'req_rent2026',
             'gap2028_plateau', 'gap2030_plateau', 'gap2028_boom', 'depacc_2030_over_2026_plateau']
    hdr = ','.join(keys + names)
    np.savetxt(os.path.join(OUT, 'montecarlo.csv'), np.hstack([draws, outs]), delimiter=',',
               header=hdr, comments='', fmt='%.5f')
    summ = {'N_valid': int(len(outs))}
    for j, nm in enumerate(names):
        x = outs[:, j]
        summ[nm] = dict(p05=float(np.percentile(x, 5)), p50=float(np.percentile(x, 50)),
                        p95=float(np.percentile(x, 95)), mean=float(x.mean()),
                        prob_positive=float((x > 0).mean()))
    summ['spearman_gap2026'] = {k: float(spearmanr(draws[:, i], outs[:, 1])[0]) for i, k in enumerate(keys)}
    summ['spearman_value_gap2026'] = {k: float(spearmanr(draws[:, i], outs[:, 3])[0]) for i, k in enumerate(keys)}
    summ['spearman_req_rent2026'] = {k: float(spearmanr(draws[:, i], outs[:, 4])[0]) for i, k in enumerate(keys)}
    json.dump(summ, open(os.path.join(OUT, 'montecarlo_summary.json'), 'w'), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
