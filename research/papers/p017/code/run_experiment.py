"""Main experiment: probe decodability vs. the k-suffix Bayes ceiling, truncation tests,
and counterfactual-rollout interventions, for trained and untrained sequence encoders.

Usage:  python run_experiment.py --modes unique balanced2 landmark --seeds 0 1 2
Outputs: ../results/raw_<mode>_s<seed>.json
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from world import TorusWorld
from models import GRULM, ESN, TinyTransformer, train_lm, fit_readout

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results" if not os.environ.get("SMOKE") else "smoke")
os.makedirs(RES, exist_ok=True)

T = 24
EVAL_T = list(range(12, T + 1))          # steps whose hidden states are probed
TRUNC_T = [16, 20, 24]
KS = [1, 2, 3, 4, 6, 8, 10, 12, 16]
T0, R = 14, 10
SMOKE = bool(os.environ.get("SMOKE"))
NTR, NPR, NTE, EP = (600, 400, 300, 1) if SMOKE else (14000, 6000, 2000, 10)                            # rollout start and horizon


def hid_at(hs, ts):
    # hidden after reading move m_t is at token index 2t-1
    return hs[:, [2 * t - 1 for t in ts]]


def encode(model, toks, kind):
    with torch.no_grad():
        X = torch.as_tensor(toks)
        out = []
        for i in range(0, X.shape[0], 1000):
            _, hs = model(X[i:i + 1000]) if kind != "tf" else model(X[i:i + 1000])
            out.append(hs)
        return torch.cat(out, 0)


def suffix_curve(world, mv, ob, pos, ts, kmax=24):
    """Acc*_k averaged over ts (expected max-posterior = Bayes accuracy of the best
    suffix-measurable predictor under the stationary prior)."""
    curve = []
    for k in range(0, kmax + 1):
        accs = []
        for t in ts:
            if k > t:
                kk = t
            else:
                kk = k
            p = world.suffix_bayes(mv, ob, t, kk)
            accs.append(p.max(1).mean())
        curve.append(float(np.mean(accs)))
    return curve


def run(mode, seed, log, with_tf):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    world = TorusWorld(5, mode, seed)
    V = world.vocab
    tr_toks, _, _, _ = world.sample(NTR, T, rng)
    pr_toks, pr_pos, pr_mv, pr_ob = world.sample(NPR, T, rng)
    te_toks, te_pos, te_mv, te_ob = world.sample(NTE, T, rng)
    res = {"mode": mode, "seed": seed, "models": {}}
    # ----- ceilings -----
    res["suffix_curve"] = suffix_curve(world, te_mv, te_ob, te_pos, EVAL_T)
    res["suffix_curve_trunc_t"] = suffix_curve(world, te_mv, te_ob, te_pos, TRUNC_T, kmax=16)
    log(f"[{mode} s{seed}] suffix curve k=1,4,8,16,40: "
        f"{[round(res['suffix_curve'][k], 3) for k in [1, 4, 8, 16, 24]]}")

    models = {}
    t0 = time.time()
    g = GRULM(V, 64)
    train_lm(g, tr_toks, epochs=EP, bs=128, lr=6e-3, seed=seed, log=None)
    models["gru_trained"] = (g, "rnn")
    log(f"  trained GRU in {time.time() - t0:.0f}s")
    torch.manual_seed(seed + 77)
    models["gru_random"] = (GRULM(V, 64), "rnn")
    models["esn_l0.9_h64"] = (ESN(V, 64, 0.9, seed), "esn")
    models["esn_l0.99_h512"] = (ESN(V, 512, 0.99, seed), "esn")
    if with_tf:
        t0 = time.time()
        tf = TinyTransformer(V, 2 * T + 1)
        train_lm(tf, tr_toks, epochs=EP, bs=128, lr=2e-3, seed=seed)
        models["tf_trained"] = (tf, "tf")
        log(f"  trained transformer in {time.time() - t0:.0f}s")
        torch.manual_seed(seed + 99)
        models["tf_random"] = (TinyTransformer(V, 2 * T + 1), "tf")

    for name, (m, kind) in models.items():
        m.eval()
        H_pr = encode(m, pr_toks, kind)
        H_te = encode(m, te_toks, kind)
        Xp = hid_at(H_pr, EVAL_T).reshape(-1, H_pr.shape[-1])
        Xt = hid_at(H_te, EVAL_T).reshape(-1, H_te.shape[-1])
        yp = pr_pos[:, EVAL_T].reshape(-1)
        yt = te_pos[:, EVAL_T].reshape(-1)
        op = pr_ob[:, EVAL_T].reshape(-1)
        ot = te_ob[:, EVAL_T].reshape(-1)
        r = {}
        # ---- next-observation readout ----
        trained = name.endswith("trained")
        if trained:
            with torch.no_grad():
                lg = m.out(Xt) if kind == "rnn" else m.outp(Xt)
            obs_logits = lg[:, 4:]
            Wo = bo = None
        else:
            Wo, bo = fit_readout(Xp, op, world.C)
            obs_logits = Xt @ Wo.T + bo
        pred_o = obs_logits.argmax(1).numpy()
        r["obs_acc"] = float((pred_o == ot).mean())
        r["obs_nll"] = float(F.cross_entropy(obs_logits, torch.as_tensor(ot)).item())
        if mode == "landmark":
            hit = ot == 1
            r["landmark_recall"] = float((pred_o[hit] == 1).mean())
            r["landmark_precision"] = float((ot[pred_o == 1] == 1).mean()) if (pred_o == 1).any() else 0.0
        # ---- state probe ----
        W, b = fit_readout(Xp, yp, world.S)
        logits = Xt @ W.T + b
        r["probe_acc"] = float((logits.argmax(1).numpy() == yt).mean())
        Wtr = Xp @ W.T + b
        r["probe_train_acc"] = float((Wtr.argmax(1).numpy() == yp).mean())
        # control task (Hewitt & Liang style): random relabelling of states per sequence-free
        # identity -> labels are a fixed random permutation of the state; not informative here,
        # so we instead use a *shuffled* control: labels permuted across samples.
        perm = np.random.default_rng(seed).permutation(len(yp))
        Wc, bc = fit_readout(Xp, yp[perm], world.S)
        r["probe_control_acc"] = float(((Xt @ Wc.T + bc).argmax(1).numpy() ==
                                        yt[np.random.default_rng(seed + 1).permutation(len(yt))]).mean())
        curve = res["suffix_curve"]
        keff = next((k for k, a in enumerate(curve) if a >= r["probe_acc"]), None)
        r["k_eff"] = keff
        r["excess_over_k1"] = r["probe_acc"] - curve[1]
        log(f"  {name:16s} obs_acc={r['obs_acc']:.3f} probe={r['probe_acc']:.3f} k_eff={keff}")

        # ---- truncation test & suffix bound (recurrent encoders only) ----
        if kind in ("rnn", "esn"):
            Wd = W.clone()
            trunc = {}
            for k in KS:
                accs, bound_terms, fullaccs, deltas, lam_terms = [], [], [], [], []
                for t in TRUNC_T:
                    sub = torch.as_tensor(te_toks[:, 2 * (t - k):2 * t])  # obs_{t-k} .. m_t
                    with torch.no_grad():
                        _, hk = m(sub)
                    hk = hk[:, -1]
                    hf = H_te[:, 2 * t - 1]
                    lk = hk @ W.T + b
                    lf = hf @ W.T + b
                    y = te_pos[:, t]
                    accs.append((lk.argmax(1).numpy() == y).mean())
                    fullaccs.append((lf.argmax(1).numpy() == y).mean())
                    delta = (hf - hk).norm(dim=1)
                    deltas.append(delta.mean().item())
                    top2 = lf.topk(2, dim=1)
                    margin = top2.values[:, 0] - top2.values[:, 1]
                    wtop = Wd[top2.indices[:, 0]]                     # (N, d)
                    Lc = (wtop[:, None, :] - Wd[None, :, :]).norm(dim=2).max(1).values
                    # prediction provably unchanged if margin > Lc * delta
                    bound_terms.append((margin <= Lc * delta).float().mean().item())
                    if kind == "esn":
                        # a-priori: delta <= lam^(2k) * sup||h_{t-k} - 0|| <= lam^(2k) sqrt(H)
                        dbound = (m.lam ** (2 * k)) * np.sqrt(m.H)
                        lam_terms.append((margin <= Lc * dbound).float().mean().item())
                ck = res["suffix_curve_trunc_t"][k]
                trunc[k] = {"trunc_probe_acc": float(np.mean(accs)), "full_probe_acc": float(np.mean(fullaccs)),
                            "mean_delta": float(np.mean(deltas)), "acc_star_k": ck,
                            "bound_data": ck + float(np.mean(bound_terms)),
                            "bound_apriori": (ck + float(np.mean(lam_terms))) if lam_terms else None}
            r["trunc"] = trunc

        # ---- counterfactual rollout interventions (recurrent encoders only) ----
        if kind in ("rnn", "esn"):
            r["rollout"] = rollout_test(m, kind, world, te_toks, te_pos, te_mv, H_te, W, Wo, bo, seed)
            log("    rollout " + " ".join(f"{k}:{v['cf_disc']:.2f}" for k, v in r["rollout"].items()))
        res["models"][name] = r
    return res


def rollout_test(m, kind, world, toks, pos, mv, H, Wprobe, Wo, bo, seed):
    rng = np.random.default_rng(seed + 5)
    N = toks.shape[0]
    src = rng.permutation(N)
    keep = pos[:, T0] != pos[src, T0]
    base_idx = np.nonzero(keep)[0]
    src_idx = src[keep]
    h = H[base_idx, 2 * T0 - 1]
    hs = H[src_idx, 2 * T0 - 1]
    d = h.shape[1]
    # probe subspace (row space of centred probe weights), rank <= S-1
    Wc = Wprobe - Wprobe.mean(0, keepdim=True)
    U, Sv, _ = torch.linalg.svd(Wc.T, full_matrices=False)
    rank = int((Sv > 1e-6 * Sv[0]).sum())
    P = U[:, :rank]
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, rank, generator=g))
    edits = {
        "none": h,
        "probe_subspace": h + (hs - h) @ P @ P.T,
        "random_subspace": h + (hs - h) @ Q @ Q.T,
        "full_swap": hs,
    }
    fut = mv[base_idx, T0:T0 + R]          # moves m_{T0+1} .. m_{T0+R}
    cf_cols = world.rollout_colors(pos[src_idx, T0], fut)
    f_cols = world.rollout_colors(pos[base_idx, T0], fut)
    disc = cf_cols != f_cols
    out = {"rank": rank}
    for name, h0 in edits.items():
        preds = []
        hcur = h0.clone()
        with torch.no_grad():
            for j in range(R + 1):
                if Wo is None:
                    lg = m.out(hcur)[:, 4:]
                else:
                    lg = hcur @ Wo.T + bo
                o = lg.argmax(1)
                preds.append(o.numpy())
                if j == R:
                    break
                hcur = m.step(hcur, o + 4)
                hcur = m.step(hcur, torch.as_tensor(fut[:, j]))
        preds = np.stack(preds, 1)
        out[name] = {
            "cf_disc": float((preds[disc] == cf_cols[disc]).mean()),
            "f_disc": float((preds[disc] == f_cols[disc]).mean()),
            "cf_acc_all": float((preds == cf_cols).mean()),
            "cf_disc_by_step": [float((preds[:, j][disc[:, j]] == cf_cols[:, j][disc[:, j]]).mean())
                                if disc[:, j].any() else None for j in range(R + 1)],
        }
    rank_out = out.pop("rank")
    out_named = {k: v for k, v in out.items()}
    out_named["_rank"] = {"cf_disc": float(rank_out)}
    return out_named


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["unique", "balanced2", "landmark"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tf_seeds", nargs="*", type=int, default=[0])
    a = ap.parse_args()
    logf = open(os.path.join(RES, f"log_{'_'.join(a.modes)}.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    for mode in a.modes:
        for seed in a.seeds:
            t0 = time.time()
            res = run(mode, seed, log, seed in a.tf_seeds)
            with open(os.path.join(RES, f"raw_{mode}_s{seed}.json"), "w") as f:
                json.dump(res, f, indent=1)
            log(f"[{mode} s{seed}] done in {time.time() - t0:.0f}s")
