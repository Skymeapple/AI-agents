"""Monte-Carlo check of Propositions 1-2 (linear-Gaussian representation model).

h = A z + B n,  z=(m,c) ~ N(0, [[1,rho],[rho,1]]),  n ~ N(0, I_k) nuisance.
Report logit  l_r = beta_m m + beta_c c ;  action logit l_a = gamma m  (exact linear readouts).
Emotion vector p_rho = E[h | l_r>0] - E[h | l_r<0]   (diff-of-means, estimated at corr rho).
Prediction:  CCI(rho) = dl_a/dl_r = gamma (beta_m + rho beta_c) / (b' S b)
             GI = [CCI(0)] * [dl_r(p_m)/dl_a(p_m)] = beta_m^2 / (beta_m^2 + beta_c^2)
"""
import json, os
import numpy as np

rng = np.random.default_rng(0)
D, K, N = 64, 6, 200000
A = rng.normal(size=(D, 2)); B = rng.normal(size=(D, K))
M = np.concatenate([A, B], 1); P = np.linalg.pinv(M)        # rows: dual readouts
gamma = 1.3
out = []
for latent in ("gaussian", "binary"):
    for bm, bc in [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.0, 0.5), (1.0, 0.0)]:
        w_r = P[0] * bm + P[1] * bc
        w_a = P[0] * gamma
        # situation (mood) pattern at rho=0
        def sample(rho):
            if latent == "gaussian":
                z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=N)
            else:
                m = rng.choice([-1.0, 1.0], N)
                c = np.where(rng.random(N) < (1 + rho) / 2, m, -m)
                z = np.stack([m, c], 1)
            n = rng.normal(size=(N, K))
            return z, z @ A.T + n @ B.T
        z0, h0 = sample(0.0)
        pm = h0[z0[:, 0] > 0].mean(0) - h0[z0[:, 0] < 0].mean(0)
        for rho in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
            z, h = sample(rho)
            lr = h @ w_r
            if latent == "binary" and bm == bc:   # break ties for binary latents
                lr = lr + 1e-3 * rng.normal(size=N)
            y = lr > 0
            p = h[y].mean(0) - h[~y].mean(0)
            cci = (w_a @ p) / (w_r @ p)
            pred = gamma * (bm + rho * bc) / (bm ** 2 + bc ** 2 + 2 * rho * bm * bc)
            rec = dict(latent=latent, beta_m=bm, beta_c=bc, rho=rho, cci_emp=float(cci), cci_pred=float(pred))
            if rho == 0.0:
                gi = cci * (w_r @ pm) / (w_a @ pm)
                rec.update(GI_emp=float(gi), GI_pred=bm ** 2 / (bm ** 2 + bc ** 2))
            out.append(rec)
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", "results", "theory_check.json"), "w") as f:
    json.dump(out, f, indent=1)
for r in out:
    print(r)
