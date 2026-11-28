"""E1b: fine sweep of the hierarchy-violation rate rho at fixed magnitudes (cv=0): first-order transition
between full absorption (t2=45deg) and a slightly tilted near-faithful dictionary; compare with the
closed-form threshold rho_dagger (absorbed-vs-faithful comparison, Prop. 3)."""
import json, numpy as np, theory as T
rng = np.random.default_rng(5)
out = []
for lam in [0.05, 0.1, 0.2]:
    a = c = 1.0
    gain = lam * (a + c - np.hypot(a, c)) - lam ** 2 / 4
    th = np.arctan2(c, a)
    cost = (c - lam / 2) ** 2 - max(c * np.sin(th) - lam / 2, 0) ** 2
    rho_dag = gain / cost
    rows = []
    for rho in np.linspace(0, 0.26, 27):
        X, lab = T.sample_data(20000, 0.0, 0.0, 0.5, rho, np.random.default_rng(5))  # common random numbers
        t1, t2, L = T.optimum(X, lam)
        st = T.readout_stats(X, lab, T.dict_from_angles(t1, t2), lam)
        rows.append(dict(rho=float(rho), t2_deg=float(np.degrees(t2)), recall_pc=float(st['recall_pc']),
                         rel_act_pc=float(st['rel_act_pc'])))
        print(lam, rho, rows[-1])
    out.append(dict(lam=lam, rho_dagger=float(rho_dag), rows=rows))
json.dump(out, open('../results/e1b_rho_fine.json', 'w'), indent=1)
