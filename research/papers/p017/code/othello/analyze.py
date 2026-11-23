"""Probe the Othello GPT (trained) and its own initialisation (untrained twin):
legal-move accuracy, per-layer linear board probes (mine/theirs/empty), stratified by how long
ago each square last changed, and the attention-window truncation test (Corollary 2).
Probe train: data_lab3 (first 5000 games); test: data_lab4 (first 2000 games).
Output: analysis.json"""
import json, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import OUT, load_lab, valid_mask, stratified
from gpt import GPT, PAD
from engine import legal, TOK2SQ
torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))


def resid(m, toks, window=None):
    X = torch.as_tensor(np.where(toks < 0, PAD, toks))
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 500):
            lg, r = m(X[i:i + 500], window=window, return_resid=True)
            outs.append((lg, torch.stack(r, 0)))
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs], 1)  # (layers+1, N, L, d)


def fit_probe(Xtr, ytr, epochs=8, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    lin = nn.Linear(Xtr.shape[1], 64 * 3)
    opt = torch.optim.Adam(lin.parameters(), 3e-3)
    y = torch.as_tensor(ytr.astype(np.int64))
    Z = (Xtr - mu) / sd
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 2048):
            j = perm[i:i + 2048]
            loss = F.cross_entropy(lin(Z[j]).view(-1, 3), y[j].view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return lambda X: lin((X - mu) / sd).view(-1, 64, 3).argmax(-1).detach().numpy()


def legal_acc(logits, toks, labs, lens):
    """Fraction of positions t (0..len-2) where the argmax next-move is legal in the true board."""
    ok, tot = 0, 0
    pred = logits.argmax(-1).numpy()
    for g in range(len(toks)):
        for t in range(lens[g] - 1):
            lab = labs[g, t]
            me = sum(1 << s for s in np.nonzero(lab == 1)[0]); op = sum(1 << s for s in np.nonzero(lab == 2)[0])
            p = pred[g, t]
            ok += p < 60 and bool(legal(int(me), int(op)) >> int(TOK2SQ[p]) & 1); tot += 1
    return ok / tot


if __name__ == "__main__":
    ptr = load_lab(["lab3"]); pte = load_lab(["lab4"])
    ptr = tuple(x[:5000] for x in ptr); pte = tuple(x[:2000] for x in pte)
    vtr, vte = valid_mask(ptr[3]), valid_mask(pte[3])
    ytr, yte, ate = ptr[1][vtr], pte[1][vte], pte[2][vte]
    res = {}
    torch.manual_seed(0); twin = GPT()                      # exact initialisation of the trained model
    trained = GPT(); s = torch.load(os.path.join(OUT, os.environ.get("CKPT", "gpt_6k.pt"))); trained.load_state_dict(s["model"]); res["train_steps"] = s["step"]
    for name, m in [("trained", trained), ("untrained_twin", twin)]:
        m.eval(); t0 = time.time()
        lg_te, R_te = resid(m, pte[0]); _, R_tr = resid(m, ptr[0])
        r = {"legal_top1": legal_acc(lg_te[:500], pte[0][:500], pte[1][:500], pte[3][:500]), "layers": {}}
        best, bestacc = None, -1
        for l in range(R_te.shape[0]):
            f = fit_probe(R_tr[l][torch.as_tensor(vtr)], ytr)
            corr = f(R_te[l][torch.as_tensor(vte)]) == yte
            r["layers"][l] = stratified(corr, ate)
            if r["layers"][l]["all"][0] > bestacc: best, bestacc, bestf = l, r["layers"][l]["all"][0], f
        r["best_layer"] = best
        print(name, "legal", round(r["legal_top1"], 4), {l: round(v["all"][0], 4) for l, v in r["layers"].items()}, f"{time.time()-t0:.0f}s", flush=True)
        # attention-window truncation (Corollary 2): window w at every layer -> receptive field nl*(w-1)+1 tokens
        r["window"] = {}
        for w in [1, 2, 3, 4, 6, 8, 16]:
            lgw, Rw_te = resid(m, pte[0], window=w); _, Rw_tr = resid(m, ptr[0], window=w)
            same = bestf(Rw_te[best][torch.as_tensor(vte)]) == yte
            fw = fit_probe(Rw_tr[best][torch.as_tensor(vtr)], ytr)
            refit = fw(Rw_te[best][torch.as_tensor(vte)]) == yte
            r["window"][w] = {"receptive_field": len(m.blocks) * (w - 1) + 1, "same_probe": stratified(same, ate),
                              "refit_probe": stratified(refit, ate),
                              "legal_top1": legal_acc(lgw[:300], pte[0][:300], pte[1][:300], pte[3][:300])}
            print("  window", w, "rf", r["window"][w]["receptive_field"], "same", round(same.mean(), 4), "refit", round(refit.mean(), 4),
                  "legal", round(r["window"][w]["legal_top1"], 4), flush=True)
        res[name] = r
        json.dump(res, open(os.path.join(OUT, os.environ.get("AOUT", "analysis_6k.json")), "w"), indent=1)
