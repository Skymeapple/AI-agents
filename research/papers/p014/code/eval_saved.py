"""Cue-removal probe for saved label-order models (revision): evaluate a model trained
with p_order on i.i.d. items whose labels are uniformly random (p_order=0) and on items
whose labels follow the REVERSE causal order. Output: results/cue_removal.json"""
import glob, json, os
import numpy as np
import torch
import scm as S
import twins as T

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def batch_reverse(rng, K):
    """Items with labels in reverse causal order (anti-cue)."""
    F, G, O, Y = [], [], [], []
    for _ in range(K):
        m = S.random_scm(rng, p_order=1.0)
        P = np.eye(S.N_NODES)[::-1]
        m = {"W": P.T @ m["W"] @ P, "s": m["s"][::-1].copy()}; m["A"] = (m["W"] != 0).astype(int)
        X, I = S.sample(rng, m, T.N, "gauss", 0, 0)
        i, j, y = T.pick_query(rng, m["A"])
        f, g, o = T.encode(X, I, np.zeros((4, 4), int), i, j)
        F.append(f); G.append(g); O.append(o); Y.append(y)
    return (torch.tensor(np.stack(F)), torch.tensor(np.stack(G)), torch.tensor(np.array(O))[:, None], np.array(Y))


out = {}
mect = S.MECTable()
for path in sorted(glob.glob(os.path.join(R, "models", "obs_p*_s[0-9].pt"))):
    key = os.path.basename(path)[:-3]
    net = T.SetNet(); net.load_state_dict(torch.load(path)); net.eval()
    cfg0 = dict(p_order=0.0, family="gauss", n_int=0, graph=False, corrupt=0.0)
    r = T.iid_metrics(net, np.random.default_rng(50_000), cfg0, mect, K=4000)
    F, G, O, Y = batch_reverse(np.random.default_rng(60_000), 4000)
    p = T.predict(net, F, G, O)
    out[key] = {"iid_acc_random_labels": r["iid_acc"], "acc_identified_random_labels": r["iid_acc_identified"],
                "acc_unidentified_random_labels": r["iid_acc_unidentified"],
                "iid_acc_reversed_labels": float(((p > 0.5).astype(int) == Y).mean())}
    print(key, out[key], flush=True)
json.dump(out, open(os.path.join(R, "cue_removal.json"), "w"), indent=1)
