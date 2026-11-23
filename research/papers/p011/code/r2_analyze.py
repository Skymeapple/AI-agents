"""Round-2 mechanistic analyses applied to (i) the round-1 final models (results/runs/*.pt) and
(ii) every saved checkpoint of the pre-specified round-2 sweep (results/runs_r2/*_step*.pt).
Per model it computes:
  behavioural tests (hard argmax and soft probabilities), attention path cuts (analyze.path_cut_eval),
  truncation under four removal formats, a presence-gating test (embedding interpolation and
  activation patching at the CoT position), and interchange interventions for Definition 3 (D).
Output: results/r2_models.json  (usage: python3 r2_analyze.py [old|new|all])"""
import os, sys, glob, json, re
import numpy as np, torch
import cotlab as C
import analyze as A

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = 23


def fwd(model, X, emb4=None, patch=None):
    """Forward pass with optional override of the CoT-position token embedding (emb4: (d,) or (B,d))
    and optional residual patch: patch=(k, fn) applies hs=fn(hs) to the residual stream *before* block k
    (k=1 is the residual after layer 1). Returns ANS logits and the list of residual streams."""
    hs = model.tok(X) + model.pos(torch.arange(6))
    if emb4 is not None:
        hs = hs.clone(); hs[:, 4] = emb4 + model.pos.weight[4]
    resid = [hs]
    for k, b in enumerate(model.blocks):
        if patch is not None and patch[0] == k:
            hs = patch[1](hs); resid[-1] = hs
        hs = b(hs, model.mask)
        resid.append(hs)
    return model.head(model.lnf(hs))[:, 5], resid


def inputs(D, st):
    a, b, c = D[:, 0], D[:, 1], D[:, 2]
    B = len(a)
    return torch.tensor(np.stack([a, b, c, np.full(B, P), st, np.full(B, P)], 1))


@torch.no_grad()
def removal_formats(model, Dte):
    s, y = Dte[:, 3], Dte[:, 4]
    E = model.tok.weight
    out = {}
    X = inputs(Dte, np.full(len(s), P + 1))
    out["trunc_MASK"] = float((fwd(model, X)[0].argmax(-1).numpy() == y).mean())
    X = inputs(Dte, np.full(len(s), P))
    out["trunc_SEP"] = float((fwd(model, X)[0].argmax(-1).numpy() == y).mean())
    X = inputs(Dte, s)
    out["trunc_meanemb"] = float((fwd(model, X, emb4=E[:P].mean(0))[0].argmax(-1).numpy() == y).mean())
    out["trunc_zeroemb"] = float((fwd(model, X, emb4=torch.zeros(E.shape[1]))[0].argmax(-1).numpy() == y).mean())
    return out


@torch.no_grad()
def gating(model, Dte, seed=5):
    """(a) Interpolate the CoT-position token embedding from a corrupted token s' (lambda=0) to MASK
    (lambda=1); record follow/ignore. (b) Activation patching: with corrupted CoT input, replace the
    residual at the CoT position after layer 1 by its value in the MASK run (and the reverse)."""
    rng = np.random.default_rng(seed)
    s, c, y = Dte[:, 3], Dte[:, 2], Dte[:, 4]
    sp = (s + rng.integers(1, P, len(s))) % P
    yf = (sp + c) % P
    E = model.tok.weight
    X = inputs(Dte, sp)
    lam_curve = []
    for lam in np.linspace(0, 1, 11):
        e = (1 - lam) * E[torch.tensor(sp)] + lam * E[P + 1]
        pr = fwd(model, X, emb4=e)[0].argmax(-1).numpy()
        lam_curve.append([float(lam), float((pr == yf).mean()), float((pr == y).mean())])
    Xm = inputs(Dte, np.full(len(s), P + 1))
    _, rm = fwd(model, Xm)
    _, rc = fwd(model, X)
    out = {"interp": lam_curve}
    for k, name in [(1, "L1")]:
        def pm(hs, src=rm[k]):
            hs = hs.clone(); hs[:, 4] = src[:, 4]; return hs
        pr = fwd(model, X, patch=(k, pm))[0].argmax(-1).numpy()
        out[f"patch_mask_into_corrupt_{name}_follow"] = float((pr == yf).mean())
        out[f"patch_mask_into_corrupt_{name}_ignore"] = float((pr == y).mean())
        def pc(hs, src=rc[k]):
            hs = hs.clone(); hs[:, 4] = src[:, 4]; return hs
        pr = fwd(model, Xm, patch=(k, pc))[0].argmax(-1).numpy()
        out[f"patch_corrupt_into_mask_{name}_acc"] = float((pr == y).mean())
        out[f"patch_corrupt_into_mask_{name}_follow"] = float((pr == yf).mean())
    return out


def probe_basis(H, lab, k, lam=1e-1):
    """Orthonormal basis (d x k) of the top-k directions of a one-hot ridge probe for `lab`."""
    Hc = H - H.mean(0)
    Y = np.eye(P)[lab]; Y = Y - Y.mean(0)
    W = np.linalg.solve(Hc.T @ Hc + lam * np.eye(H.shape[1]), Hc.T @ Y)   # d x P
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    return U[:, :k]


@torch.no_grad()
def interchange(model, Dtr, Dte, cond, seed=9):
    """Definition 3 test. Base (a,b,c) and source (a2,b2,c2) with c2 != c and s2 != s; CoT token is a
    corrupted value (cond='corrupt') or MASK (cond='mask') in both. At a site (residual after layer 1, at
    position j) replace the component of the base activation in a subspace U by the source's.
    Outcomes: 's_aligned' y = s2 + c (the site carries the intermediate s, separately from c),
    'y_src' y = s2 + c2 (the site carries the whole answer), 'unchanged' (= unpatched base output)."""
    rng = np.random.default_rng(seed)
    def tok(D):
        if cond == "mask":
            return np.full(len(D), P + 1)
        return (D[:, 3] + rng.integers(1, P, len(D))) % P
    sub = Dtr[np.random.default_rng(3).permutation(len(Dtr))[:6000]]
    _, rtr = fwd(model, inputs(sub, tok(sub)))
    perm = rng.permutation(len(Dte))
    src = Dte[perm]
    keep = (src[:, 2] != Dte[:, 2]) & (src[:, 3] != Dte[:, 3])
    base, src = Dte[keep], src[keep]
    Xb, Xs = inputs(base, tok(base)), inputs(src, tok(src))
    lb, rb = fwd(model, Xb); _, rs = fwd(model, Xs)
    yb = lb.argmax(-1).numpy()
    s_al = (src[:, 3] + base[:, 2]) % P
    y_src = (src[:, 3] + src[:, 2]) % P
    out = {"base_acc": float((yb == base[:, 4]).mean()), "n": int(len(base))}
    # Sites (residual stream after layer 1). Under the causal mask, positions 0,1,3 carry information
    # about (a,b) only (no c); positions 2,4,5 may already combine a,b with c. A full patch of an
    # (a,b)-only site set is an interchange intervention aligning the high-level variable s with that
    # site set; its interchange-intervention accuracy (IIA) is the 's_aligned' rate.
    def full(js):
        def fn(hs):
            hs = hs.clone()
            for j in js:
                hs[:, j] = rs[1][:, j]
            return hs
        return fn
    def subsp(j, U):
        Ut = torch.tensor(U, dtype=torch.float32)
        def fn(hs):
            hs = hs.clone(); dlt = rs[1][:, j] - hs[:, j]
            hs[:, j] = hs[:, j] + (dlt @ Ut) @ Ut.T; return hs
        return fn
    d = rb[1].shape[-1]
    Rnd = np.linalg.qr(np.random.default_rng(1).standard_normal((d, 22)))[0]
    H3 = rtr[1][:, 3].numpy()
    sites = {"p1": full([1]), "p3": full([3]), "p13": full([1, 3]), "p013": full([0, 1, 3]),
             "p2": full([2]), "p5": full([5]),
             "p3_s8": subsp(3, probe_basis(H3, sub[:, 3], 8)), "p3_s22": subsp(3, probe_basis(H3, sub[:, 3], 22)),
             "p3_rand22": subsp(3, Rnd)}
    for name, fn in sites.items():
        pr = fwd(model, Xb, patch=(1, fn))[0].argmax(-1).numpy()
        out[name] = {"s_aligned": float((pr == s_al).mean()), "y_src": float((pr == y_src).mean()),
                     "unchanged": float((pr == yb).mean())}
    return out


def evaluate_model(model, Dtr, Dte):
    row = {}
    row.update(C.evaluate(model, Dte, P, "add"))
    row.update(A.path_cut_eval(model, Dte, P, "add"))
    row.update(A.soft_eval(model, Dte, P))
    row.update(removal_formats(model, Dte))
    row["gating"] = gating(model, Dte)
    row["D_corrupt"] = interchange(model, Dtr, Dte, "corrupt")
    row["D_mask"] = interchange(model, Dtr, Dte, "mask")
    return row


def main(which):
    rows = []
    if which in ("old", "all"):
        for f in sorted(glob.glob(os.path.join(R, "runs", "*.json"))):
            rec = json.load(open(f))
            m = C.TinyTF(P, d=64, L=2, regime=rec["regime"])
            m.load_state_dict(torch.load(os.path.join(R, "runs", rec["tag"] + ".pt"))); m.eval()
            Dtr, Dte = C.make_data(P, "add", rec["seed"])
            row = dict(set="r1", tag=rec["tag"], regime=rec["regime"], eta=rec["eta"], delta=rec["delta"],
                       seed=rec["seed"], step=rec["steps"])
            row.update(evaluate_model(m, Dtr, Dte)); rows.append(row)
            print(row["tag"], flush=True)
    if which in ("new", "all"):
        for f in sorted(glob.glob(os.path.join(R, "runs_r2", "*_step*.pt"))):
            mt = re.match(r"r2_e([\d.]+)_d([\d.]+)_s(\d+)_step(\d+)\.pt", os.path.basename(f))
            eta, delta, seed, step = float(mt[1]), float(mt[2]), int(mt[3]), int(mt[4])
            m = C.TinyTF(P, d=64, L=2, regime="free"); m.load_state_dict(torch.load(f)); m.eval()
            Dtr, Dte = C.make_data(P, "add", seed)
            row = dict(set="r2", tag=os.path.basename(f)[:-3], regime="free", eta=eta, delta=delta, seed=seed, step=step)
            row.update(evaluate_model(m, Dtr, Dte)); rows.append(row)
            print(row["tag"], flush=True)
    fn = os.path.join(R, f"r2_models_{which}.json")
    json.dump(rows, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
