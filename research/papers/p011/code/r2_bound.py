"""Check of the quantitative Prop. 3(b') bound S_train <= p*eps/((p-1)*eta) and N_train <= eps/delta, where eps
is the model's answer cross-entropy on the training distribution (exact expectation over the noise:
clean, all p noisy tokens, MASK). Applied to round-2 checkpoints with eta>0. Output: results/r2_bound.json"""
import os, glob, json, re
import numpy as np, torch, torch.nn.functional as F
import cotlab as C
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); P = 23
rows = []
files = sorted(glob.glob(os.path.join(R, "runs_r2", "*_step*.pt"))) + sorted(glob.glob(os.path.join(R, "runs", "add_free_e0.*.pt")))
for f in files:
    b = os.path.basename(f)
    m = re.match(r"r2_e([\d.]+)_d([\d.]+)_s(\d+)_step(\d+)\.pt", b) or re.match(r"add_free_e([\d.]+)_d([\d.]+)_s(\d+)(?:_long)?\.pt", b)
    eta, delta, seed = float(m[1]), float(m[2]), int(m[3])
    if eta == 0: continue
    net = C.TinyTF(P, d=64, L=2); net.load_state_dict(torch.load(f)); net.eval()
    Dtr, _ = C.make_data(P, "add", seed)
    a, bb, c, s, y = Dtr.T; B = len(a)
    def ce_and_probs(st):
        X = torch.tensor(np.stack([a, bb, c, np.full(B, P), st, np.full(B, P)], 1))
        with torch.no_grad():
            lo = net(X)[:, 5]
        return F.cross_entropy(lo, torch.tensor(y), reduction="none").numpy(), lo.softmax(-1).numpy()
    ce_clean, _ = ce_and_probs(s)
    ce_noise = np.zeros(B); follow = np.zeros(B)
    for v in range(P):
        ce_v, pr = ce_and_probs(np.full(B, v))
        ce_noise += ce_v / P
        wrong = v != s
        follow += np.where(wrong, pr[np.arange(B), (v + c) % P], 0) / (P - 1)
    ce_mask, prm = ce_and_probs(np.full(B, P + 1))
    eps = (1 - eta - delta) * ce_clean.mean() + eta * ce_noise.mean() + delta * ce_mask.mean()
    rows.append(dict(model=b[:-3], eta=eta, delta=delta, eps=float(eps), soft_follow_train=float(follow.mean()),
                     bound_S=float(P * eps / ((P - 1) * eta)),
                     soft_trunc_fail_train=float(1 - prm[np.arange(B), y].mean()),
                     bound_N=(float(eps / delta) if delta > 0 else None)))
    print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(R, "r2_bound.json"), "w"), indent=1)
ok = all(r["soft_follow_train"] <= r["bound_S"] + 1e-9 for r in rows)
print("bound_S holds for all:", ok, " n=", len(rows))
