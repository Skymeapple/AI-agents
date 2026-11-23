"""E2: train ReLU SAEs (2 latents) on the two-feature hierarchy embedded in R^8 and compare with the
exact optimum from E1. Outputs results/e2_sae.json"""
import json, time, numpy as np, theory as T
from sae import SAE, train

D_AMB = 8
rng0 = np.random.default_rng(123)
Q, _ = np.linalg.qr(rng0.standard_normal((D_AMB, D_AMB)))
eP, eC = Q[:, 0], Q[:, 1]
E = np.stack([eP, eC])                 # (2, d)


def make_sampler(cv, q, rho):
    def s(n, rng):
        X2, _ = T.sample_data(n, cv, cv, q, rho, rng)
        return X2 @ E
    return s


def analyse(sae, cv, q, rho, lam, rng):
    X2, lab = T.sample_data(40000, cv, cv, q, rho, rng)
    X = X2 @ E
    P = sae.Wd @ E.T                   # decoder coords in (eP, eC) plane, (2,2)
    i_par = int(np.argmax(P[:, 0] / np.linalg.norm(sae.Wd, axis=1)))
    i_ch = 1 - i_par
    Z = sae.encode(X)
    fire = Z[:, i_par] > 0.1
    parent = (lab == 1) | (lab == 2)
    res = dict(t1_deg=float(np.degrees(np.arctan2(P[i_par, 1], P[i_par, 0]))),
               t2_deg=float(np.degrees(np.arctan2(P[i_ch, 1], P[i_ch, 0]))),
               offplane=float(1 - np.sum(P ** 2, 1).min()),
               recall_pc=float(fire[lab == 2].mean()),
               recall_ponly=float(fire[lab == 1].mean()),
               precision=float(parent[fire].mean()) if fire.any() else float('nan'),
               rel_act_pc=float((Z[lab == 2, i_par] / X2[lab == 2, 0]).mean()),
               child_recall_pc=float((Z[lab == 2, i_ch] > 0.1).mean()),
               loss=float(sae.loss(X, lam)))
    # exact-code loss with the SAE's learned dictionary (amortisation gap)
    Dp = P / np.linalg.norm(P, axis=1, keepdims=True)
    _, f = T.nn_lasso2(X2, Dp, lam)
    res['loss_exact_codes_same_dict'] = float(f.mean())
    return res


def run(cv, q, rho, lam, seeds=3, steps=5000):
    best = None
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        sae = SAE(D_AMB, 2, rng)
        train(sae, make_sampler(cv, q, rho), lam, steps=steps, rng=rng)
        r = analyse(sae, cv, q, rho, lam, np.random.default_rng(7))
        r['seed'] = s
        if best is None or r['loss'] < best['loss']:
            best = r
    best.update(cv=cv, q=q, rho=rho, lam=lam)
    return best


if __name__ == '__main__':
    t0 = time.time()
    out = {'grid': [], 'q_sweep': [], 'rho_sweep': []}
    for cv in [0.0, 0.1, 0.2, 0.4]:
        for lam in [0.05, 0.1, 0.2]:
            out['grid'].append(run(cv, 0.5, 0.0, lam)); print(out['grid'][-1], time.time() - t0, flush=True)
    for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
        out['q_sweep'].append(run(0.2, q, 0.0, 0.1)); print(out['q_sweep'][-1], flush=True)
    for cv in [0.0, 0.2]:
        for rho in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            out['rho_sweep'].append(run(cv, 0.5, rho, 0.1)); print(out['rho_sweep'][-1], flush=True)
    json.dump(out, open('../results/e2_sae.json', 'w'), indent=1)
    print('total', time.time() - t0)
