"""E1: exact optimum of the sparse-coding objective in the two-feature hierarchy model.
Outputs results/e1_theory.json"""
import json, numpy as np, theory as T, time

rng = np.random.default_rng(1)
N = 12000
out = {'grid': [], 'q_sweep': [], 'rho_sweep': [], 'fixed_closed_form': [], 'derivative_check': []}
t0 = time.time()
CVS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
LAMS = [0.02, 0.05, 0.1, 0.2, 0.4]
for cv in CVS:
    for lam in LAMS:
        X, lab = T.sample_data(N, cv, cv, 0.5, 0.0, rng)
        t1, t2, L = T.optimum(X, lam)
        st = T.readout_stats(X, lab, T.dict_from_angles(t1, t2), lam)
        _, Lf = T.nn_lasso2(X, T.dict_from_angles(0, np.pi / 2), lam)
        out['grid'].append(dict(cv=cv, lam=lam, t1_deg=float(np.degrees(t1)), t2_deg=float(np.degrees(t2)),
                                loss_opt=float(L), loss_faithful=float(Lf[np.any(X != 0, 1)].mean()),
                                **{k: float(v) for k, v in st.items()}))
print('grid done', time.time() - t0)
for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
    X, lab = T.sample_data(N, 0.2, 0.2, q, 0.0, rng)
    t1, t2, L = T.optimum(X, 0.1)
    st = T.readout_stats(X, lab, T.dict_from_angles(t1, t2), 0.1)
    out['q_sweep'].append(dict(q=q, cv=0.2, lam=0.1, t1_deg=float(np.degrees(t1)), t2_deg=float(np.degrees(t2)),
                               **{k: float(v) for k, v in st.items()}))
for rho in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    for cv in [0.0, 0.2]:
        X, lab = T.sample_data(N, cv, cv, 0.5, rho, rng)
        t1, t2, L = T.optimum(X, 0.1)
        st = T.readout_stats(X, lab, T.dict_from_angles(t1, t2), 0.1)
        out['rho_sweep'].append(dict(rho=rho, cv=cv, lam=0.1, t1_deg=float(np.degrees(t1)), t2_deg=float(np.degrees(t2)),
                                     **{k: float(v) for k, v in st.items()}))
print('sweeps done', time.time() - t0)
# Prop 1 closed form vs numeric (fixed magnitudes a, c)
for a, c in [(1, 1), (1, 0.5), (1, 2)]:
    for lam in [0.05, 0.2]:
        f, ab = T.closed_form_fixed(a, c, lam)
        x = np.array([[a, c]], float)
        _, nf = T.nn_lasso2(x, T.dict_from_angles(0, np.pi / 2), lam)
        th = np.arctan2(c, a)
        _, na = T.nn_lasso2(x, T.dict_from_angles(0, th), lam)
        out['fixed_closed_form'].append(dict(a=a, c=c, lam=lam, faithful_closed=f, faithful_num=float(nf[0]),
                                             absorbed_closed=ab, absorbed_num=float(na[0])))
# Prop 2: derivative of expected loss wrt t2 at the faithful dictionary vs closed form
for cv, rho, lam in [(0.0, 0.0, 0.1), (0.2, 0.0, 0.1), (0.4, 0.5, 0.2), (0.2, 2.0, 0.05)]:
    X, lab = T.sample_data(200000, cv, cv, 0.5, rho, rng)
    h = 1e-4
    _, Lp = T.nn_lasso2(X, T.dict_from_angles(0, np.pi / 2), lam)
    _, Lm = T.nn_lasso2(X, T.dict_from_angles(0, np.pi / 2 - h), lam)
    fd = (Lp.mean() - Lm.mean()) / h
    a, c = X[:, 0], X[:, 1]
    pc = lab == 2
    z2 = np.clip(c - lam / 2, 0, None)
    w = np.where(a > lam / 2, lam, 2 * a)
    closed = np.mean(np.where(pc, w * z2, 0.0))
    out['derivative_check'].append(dict(cv=cv, rho=rho, lam=lam, finite_diff=float(fd), closed_form=float(closed)))
print(json.dumps(out['derivative_check'], indent=1))
print(json.dumps(out['fixed_closed_form'], indent=1))
json.dump(out, open('../results/e1_theory.json', 'w'), indent=1)
print('total', time.time() - t0)
