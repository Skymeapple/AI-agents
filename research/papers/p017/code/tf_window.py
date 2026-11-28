"""Review M5 / question 3: transformers on the torus, 3 seeds, with the attention-window
truncation test that exercises Corollary 2. Each query attends only to the last w tokens at
every layer, so h_t depends on at most n_layers*(w-1)+1 tokens (and on t through the absolute
positional embedding). We apply the probe fitted on the full-context model ('same') and a probe
refitted on the windowed model ('refit'), and compare with the exact ceiling Acc*_k for
k = ceil(receptive_field / 2) (obs, move) pairs.
Usage: python tf_window.py --mode landmark --seeds 0 1 2 ; output ../results/tf/tf_<mode>_s<seed>.json"""
import argparse, json, os, math, time
import numpy as np, torch
import run_experiment as R
from world import TorusWorld, exact_curve
from models import TinyTransformer, train_lm, fit_readout

ap = argparse.ArgumentParser(); ap.add_argument("--mode"); ap.add_argument("--seeds", nargs="+", type=int)
a = ap.parse_args(); torch.set_num_threads(1)
od = os.path.join(R.RES, "tf"); os.makedirs(od, exist_ok=True)


def enc_window(m, toks, w):
    L = toks.shape[1]
    i = torch.arange(L)
    ok = (i[None, :] <= i[:, None]) & ((i[:, None] - i[None, :]) < w) if w else (i[None, :] <= i[:, None])
    mask = torch.zeros(L, L).masked_fill(~ok, float("-inf"))
    out = []
    with torch.no_grad():
        X = torch.as_tensor(toks)
        for j in range(0, len(X), 1000):
            x = m.emb(X[j:j + 1000]) + m.pos[:L]
            x = m.enc(x, mask=mask)
            out.append(m.ln(x))
    return torch.cat(out)


for seed in a.seeds:
    t0 = time.time()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    w = TorusWorld(5, a.mode, seed)
    tr, _, _, _ = w.sample(14000, R.T, rng); pr, pr_pos, _, _ = w.sample(6000, R.T, rng); te, te_pos, te_mv, te_ob = w.sample(2000, R.T, rng)
    ce = exact_curve(w, te_mv, te_ob, R.EVAL_T)
    rec = {"mode": a.mode, "seed": seed, "exact_curve": ce, "models": {}}
    tf = TinyTransformer(w.vocab, 2 * R.T + 1)
    train_lm(tf, tr, epochs=10, bs=128, lr=2e-3, seed=seed)
    torch.manual_seed(seed + 99)
    for name, m in [("tf_trained", tf), ("tf_random", TinyTransformer(w.vocab, 2 * R.T + 1))]:
        m.eval()
        Hp, Ht = enc_window(m, pr, None), enc_window(m, te, None)
        Xp = R.hid_at(Hp, R.EVAL_T).reshape(-1, 64); Xt = R.hid_at(Ht, R.EVAL_T).reshape(-1, 64)
        yp = pr_pos[:, R.EVAL_T].reshape(-1); yt = te_pos[:, R.EVAL_T].reshape(-1)
        W, b = fit_readout(Xp, yp, w.S)
        pa = float(((Xt @ W.T + b).argmax(1).numpy() == yt).mean())
        with torch.no_grad():
            oa = float((m.outp(Xt)[:, 4:].argmax(1).numpy() == te_ob[:, R.EVAL_T].reshape(-1)).mean())
        r = {"probe_acc": pa, "obs_acc_own_head": oa, "k_eff_exact": next((k for k, x in enumerate(ce) if x >= pa - 1e-12), None), "window": {}}
        for win in [1, 2, 3, 4, 6, 8, 12, 16]:
            rf = 2 * (win - 1) + 1
            kk = math.ceil(rf / 2)
            Hpw, Htw = enc_window(m, pr, win), enc_window(m, te, win)
            Xpw = R.hid_at(Hpw, R.EVAL_T).reshape(-1, 64); Xtw = R.hid_at(Htw, R.EVAL_T).reshape(-1, 64)
            same = float(((Xtw @ W.T + b).argmax(1).numpy() == yt).mean())
            Wr, br = fit_readout(Xpw, yp, w.S)
            refit = float(((Xtw @ Wr.T + br).argmax(1).numpy() == yt).mean())
            r["window"][win] = {"receptive_field_tokens": rf, "k_pairs": kk, "acc_star": ce[min(kk, 24)], "same": same, "refit": refit}
        rec["models"][name] = r
        print(a.mode, seed, name, f"probe={pa:.3f} keff={r['k_eff_exact']} obs={oa:.3f}",
              {k: (v['k_pairs'], round(v['acc_star'], 3), round(v['same'], 3), round(v['refit'], 3)) for k, v in r["window"].items()}, flush=True)
    json.dump(rec, open(os.path.join(od, f"tf_{a.mode}_s{seed}.json"), "w"), indent=1)
    print("done", time.time() - t0, flush=True)
