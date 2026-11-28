"""Exp 5: do cheap pairwise statistics measured on held-out items predict a team's vote gain?
For 600 random 5-agent teams from the 46-agent pool: measure, on a 'calibration' half of the items,
mean member accuracy, best member accuracy, mean correctness correlation rho_bar, and mean wrong-answer
agreement; then predict plurality-vote accuracy on the other half. Reports cross-validated R^2."""
import json
import numpy as np
from common import load, plurality

d = load()
A, y, cat = d["A"], d["y"], d["cat"]
n, m = A.shape
rng = np.random.default_rng(5)
cal = np.zeros(n, bool)
for c in np.unique(cat):
    idx = np.nonzero(cat == c)[0]
    cal[rng.choice(idx, len(idx) // 2, replace=False)] = True
ev = ~cal
C = A == y[:, None]
Wr = (~C) & (A >= 0)
accc = C[cal].mean(0)
rho = np.corrcoef(C[cal].T.astype(float))
agree = np.zeros((m, m))
for i in range(m):
    for j in range(i + 1, m):
        b = cal & Wr[:, i] & Wr[:, j]
        agree[i, j] = agree[j, i] = (A[b, i] == A[b, j]).mean() if b.sum() > 20 else np.nan

rows = []
for t in range(600):
    team = rng.choice(m, 5, replace=False)
    iu = np.triu_indices(5, 1)
    rows.append([accc[team].mean(), accc[team].max(), rho[np.ix_(team, team)][iu].mean(),
                 np.nanmean(agree[np.ix_(team, team)][iu]), (plurality(A[ev][:, team], rng) == y[ev]).mean()])
R = np.array(rows)
Y = R[:, 4] - R[:, 0]          # vote gain over mean member (evaluation half)


def cv_r2(X, Y, folds=10):
    idx = np.arange(len(Y)); rng2 = np.random.default_rng(0); rng2.shuffle(idx)
    pred = np.zeros_like(Y)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        Xt = np.column_stack([np.ones(len(tr)), X[tr]]); b = np.linalg.lstsq(Xt, Y[tr], rcond=None)[0]
        pred[te] = np.column_stack([np.ones(len(te)), X[te]]) @ b
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum())


specs = {"mean_acc": [0], "mean_acc+best_acc": [0, 1], "mean_acc+rho_bar": [0, 2], "mean_acc+wrong_agree": [0, 3],
         "mean_acc+best+rho_bar+wrong_agree": [0, 1, 2, 3], "rho_bar+wrong_agree": [2, 3]}
res = {k: cv_r2(R[:, v], Y) for k, v in specs.items()}
Xf = np.column_stack([np.ones(len(Y)), (R[:, :4] - R[:, :4].mean(0)) / R[:, :4].std(0)])
beta = np.linalg.lstsq(Xf, Y, rcond=None)[0]
out = dict(cv_r2_vote_gain_over_mean_member=res, standardized_coefs_full_model=dict(
    intercept=beta[0], mean_acc=beta[1], best_acc=beta[2], rho_bar=beta[3], wrong_agree=beta[4]),
    gain_mean=float(Y.mean()), gain_sd=float(Y.std()),
    corr_gain_rho=float(np.corrcoef(R[:, 2], Y)[0, 1]), corr_gain_agree=float(np.corrcoef(R[:, 3], Y)[0, 1]))
out = json.loads(json.dumps(out, default=float))
json.dump(out, open("../results/exp5_predict.json", "w"), indent=2)
print(json.dumps(out, indent=2))
