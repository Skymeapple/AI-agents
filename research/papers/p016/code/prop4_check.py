"""Check Proposition 4 (momentum overshoot) in the rank-one model (theta=45deg, a=b=c=1): heavy ball, eta=1/lambda_2."""
import numpy as np, json, os
from scipy.linalg import eigh
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
th = np.pi/4; u = np.array([1., 0]); v = np.array([np.cos(th), np.sin(th)]); e0 = np.array([0., 1.])
rows = []
for rho in [1e-3, 1e-2, 0.1]:
    H = (1-rho)*np.outer(v, v) + rho*np.outer(u, u); lam, Q = eigh(H); eta = 1/lam[1]
    Finf = 0.5*np.sin(th)**2*np.cos(th)**2
    for beta in [0.0, 0.5, 0.9]:
        e_prev = e0.copy(); e = e0.copy(); L = [0.0]
        for k in range(int(50/(eta*lam[0])*(1-beta)) + 200):
            e_new = e - eta*H@e + beta*(e - e_prev); e_prev, e = e, e_new; L.append(0.5*(u@e)**2)
        L = np.array(L); T = int(np.where(L >= 0.1*Finf)[0][-1]) + 1
        rows.append(dict(rho=rho, beta=beta, G_over_Finf=float(L.max()/Finf), L2step_over_Finf=float(L[2]/Finf),
                         pred_limit=(1+beta)**2, T=T))
        print(rows[-1])
json.dump(rows, open(os.path.join(R, "prop4_check.json"), "w"), indent=1)
