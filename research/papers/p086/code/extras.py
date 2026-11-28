"""Break-even obsolescence rates and a (g, L) heat-map of the 2026 depreciation gap.
Outputs results/extras.json, results/heatmap_gap_ratio_2026.csv, results/gstar_curve.csv
"""
import json, os
import numpy as np
from scipy.optimize import brentq
import data as D
from fleet import run_fleet

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')


def gstar_closed(n, L):
    """Corollary 2 (c=0, no Tmax): Phi(n)=1  <=>  g = n k/(1-k), k=(1-e^{-nL})/(nL)."""
    if n < 1e-9:
        return 2.0 / L
    k = (1 - np.exp(-n * L)) / (n * L)
    return n * k / (1 - k)


def main():
    res = {}
    # closed-form break-even curve
    rows = ['n,L,gstar']
    for L in (5.0, 5.5, 6.0):
        for n in np.linspace(0, 1.2, 61):
            rows.append(f'{n:.3f},{L},{gstar_closed(n, L):.5f}')
    open(os.path.join(OUT, 'gstar_curve.csv'), 'w').write('\n'.join(rows) + '\n')
    gs = [gstar_closed(n, 5.75) for n in np.linspace(0, 1.2, 121)]
    res['gstar_monotone_decreasing_L5.75'] = bool(np.all(np.diff(gs) < 0))
    cap = {y: sum(D.CAPEX[k][y] for k in D.CAPEX) for y in range(2021, 2026)}
    cap[2026] = sum(D.GUIDE_2026.values())
    res['capex_growth_log'] = {str(y): float(np.log(cap[y] / cap[y - 1])) for y in range(2022, 2027)}
    res['gstar_closed_L5.75'] = {f'n={n}': gstar_closed(n, 5.75) for n in (0.0, 0.2, 0.5, 0.71, 1.0)}

    # fleet break-even g for 2026 (base params, plateau)
    def gap26(g):
        p = dict(D.BASE); p['g'] = g
        rows_, _ = run_fleet(p, 'plateau', years_out=[2026])
        return rows_[0]['gap']
    res['fleet_gstar_2026'] = brentq(gap26, 0.05, 0.35)
    def gap30(g):
        p = dict(D.BASE); p['g'] = g
        rows_, _ = run_fleet(p, 'plateau', years_out=[2030])
        return rows_[0]['gap']
    try:
        res['fleet_gstar_2030_plateau'] = brentq(gap30, 0.05, 0.6)
    except ValueError:
        res['fleet_gstar_2030_plateau'] = None

    # heat-map
    gs_ = np.round(np.linspace(0.10, 0.60, 11), 3); Ls = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    lines = ['g,' + ','.join(str(L) for L in Ls)]
    for g in gs_:
        vals = []
        for L in Ls:
            p = dict(D.BASE); p['g'] = g; p['L'] = L
            r, _ = run_fleet(p, 'plateau', years_out=[2026])
            vals.append(r[0]['gap'] / r[0]['dep_acc'])
        lines.append(f'{g},' + ','.join(f'{v:.4f}' for v in vals))
    open(os.path.join(OUT, 'heatmap_gap_ratio_2026.csv'), 'w').write('\n'.join(lines) + '\n')
    # life L at which the 2026 gap vanishes, for g in {0.25,0.35,0.45}
    res['L_breakeven_2026'] = {}
    for g in (0.25, 0.35, 0.45):
        def f(L):
            p = dict(D.BASE); p['g'] = g; p['L'] = L
            r, _ = run_fleet(p, 'plateau', years_out=[2026]); return r[0]['gap']
        try:
            res['L_breakeven_2026'][str(g)] = brentq(f, 1.0, 6.5)
        except ValueError:
            res['L_breakeven_2026'][str(g)] = None
    json.dump(res, open(os.path.join(OUT, 'extras.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()


def prop3():
    """Prop 3: zero-growth steady-state stock gap per $1/yr of server capex = L/2 - E_delta[a]."""
    from scipy.integrate import quad
    from vintage import solve_P, life, econ_dep, value
    out = {}
    for g in (0.15, 0.2, 0.35, 0.5):
        c, r, Tmax, L = D.BASE['c'], D.BASE['r'], D.BASE['Tmax'], D.BASE['L']
        P = solve_P(c, g, r, Tmax); T = life(P, c, g, Tmax)
        Ea = quad(lambda a: a * float(econ_dep(a, P, c, g, r, Tmax)), 0, T)[0]
        intV = quad(lambda a: float(value(a, P, c, g, r, Tmax)), 0, T)[0]
        out[str(g)] = dict(T=T, mean_age_econ_dep=Ea, int_V=intV, L_half=L / 2, stock_gap_per_unit=L / 2 - Ea)
    json.dump(out, open(os.path.join(OUT, 'prop3.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    prop3()
