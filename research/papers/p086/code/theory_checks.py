"""Numerical verification of Lemma 1 and Proposition 1/2 of the paper.

Lemma 1   : economic depreciation delta(a) is non-increasing in age on [0,T)
Prop. 1   : steady-state ratio Phi(n) = D_econ/D_acc equals 1 at n=0
Prop. 2   : if T <= L, Phi(n) is strictly increasing in the capex growth rate n>0
Corollary : closed form for c=0, Tmax=inf: Phi(n) = [g/(g+n)] / [(1-e^{-nL})/(nL)]
Outputs results/theory_checks.json and results/phi_curves.csv
"""
import json, itertools, os
import numpy as np
from scipy.integrate import quad
from vintage import solve_P, life, econ_dep, value

OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUT, exist_ok=True)


def phi(n, P, c, g, r, Tmax, L):
    T = life(P, c, g, Tmax)
    num = quad(lambda a: float(econ_dep(a, P, c, g, r, Tmax)) * np.exp(-n * a), 0, T, limit=200)[0]
    den = quad(lambda a: (1.0 / L) * np.exp(-n * a), 0, L)[0]
    return num / den


def main():
    grid = dict(g=[0.15, 0.25, 0.35, 0.5, 0.7], c=[0.0, 0.03, 0.06, 0.1, 0.15],
                r=[0.06, 0.09, 0.12], Tmax=[6.0, 8.0, 12.0], L=[5.0, 5.5, 6.0])
    ns = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 1.0])
    n_cases = 0; lemma_viol = 0; prop1_maxerr = 0.0
    prop2_cases = 0; prop2_viol = 0; prop2_TgtL_nonmono = 0; TgtL = 0
    worst = None
    for g, c, r, Tmax, L in itertools.product(*grid.values()):
        try:
            P = solve_P(c, g, r, Tmax)
        except ValueError:
            continue
        T = life(P, c, g, Tmax)
        n_cases += 1
        aa = np.linspace(0, T * 0.999, 400)
        d = econ_dep(aa, P, c, g, r, Tmax)
        if np.any(np.diff(d) > 1e-9):
            lemma_viol += 1
            worst = dict(g=g, c=c, r=r, Tmax=Tmax, L=L)
        ph = np.array([phi(n, P, c, g, r, Tmax, L) for n in ns])
        # Prop 1: Phi(0)=1 (both depreciation schedules integrate to 1)
        prop1_maxerr = max(prop1_maxerr, abs(ph[0] - 1))
        mono = np.all(np.diff(ph) > 0)
        if T <= L:
            prop2_cases += 1
            if not mono:
                prop2_viol += 1
        else:
            TgtL += 1
            if not mono:
                prop2_TgtL_nonmono += 1
    # corollary check
    g, L = 0.35, 5.75
    cor = []
    for n in [0.1, 0.3, 0.65]:
        closed = (g / (g + n)) / ((1 - np.exp(-n * L)) / (n * L))
        P = solve_P(0.0, g, 0.09, 200.0)
        numer = phi(n, P, 0.0, g, 0.09, 200.0, L)
        cor.append(dict(n=n, closed_form=closed, numerical=numer))
    res = dict(cases=n_cases, lemma1_violations=lemma_viol, lemma1_counterexample=worst,
               prop1_max_abs_error=prop1_maxerr, prop2_cases_T_le_L=prop2_cases,
               prop2_violations=prop2_viol, cases_T_gt_L=TgtL,
               cases_T_gt_L_nonmonotone=prop2_TgtL_nonmono, corollary=cor)
    json.dump(res, open(os.path.join(OUT, 'theory_checks.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))

    # Phi curves for the figure: base params, several g
    rows = ['g,c,r,Tmax,L,T,n,phi']
    for g in [0.2, 0.35, 0.5]:
        c, r, Tmax, L = 0.06, 0.09, 8.0, 5.75
        P = solve_P(c, g, r, Tmax); T = life(P, c, g, Tmax)
        for n in np.linspace(0, 1.0, 41):
            rows.append(f'{g},{c},{r},{Tmax},{L},{T:.4f},{n:.3f},{phi(n,P,c,g,r,Tmax,L):.6f}')
    open(os.path.join(OUT, 'phi_curves.csv'), 'w').write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
