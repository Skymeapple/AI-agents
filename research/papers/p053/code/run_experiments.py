"""Train all worlds x seeds and run the grounding analyses. Usage: python3 run_experiments.py"""
import json, os, sys, time
import numpy as np
import torch
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from emo import make_data, labels, train, T0, T

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
WORLDS = [("D", 0.0), ("X25", 0.25), ("X50", 0.5), ("X75", 0.75), ("G", 1.0), ("M", 0.0)]
SEEDS = [0, 1, 2, 3, 4]
RHOS = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
POS = [8, 12, 16, 20]           # readout positions used for analysis
N_EVAL = 3000


def states(net, d):
    with torch.no_grad():
        hs, lr, la = net(torch.from_numpy(d["X"]))
    return hs.numpy(), lr.numpy(), la.numpy()


def flat(a, pos=POS):
    return np.concatenate([a[:, p] for p in pos], 0)


def probe_acc(X, y, rng, steps=300):
    """Logistic-regression probe, 50/50 split, standardized features."""
    n = len(y); idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    tri = torch.tensor(tr)

    def closure():
        opt.zero_grad()
        z = Xt[tri] @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt[tri]) + 1e-3 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        pred = (Xt[torch.tensor(te)] @ w + b) > 0
    return float((pred.numpy() == (y[te] > 0.5)).mean())


def analyse(world, w, seed):
    torch.set_num_threads(1)
    t0 = time.time()
    net = train(world, w, seed)
    Wr = net.rep.weight.detach().numpy(); Wa = net.act.weight.detach().numpy()
    vr = Wr[0] - Wr[1]          # happy-minus-sad report logit direction
    va = Wa[1] - Wa[0]          # risky-minus-safe action logit direction
    res = dict(world=world, w=w, seed=seed)
    rng = np.random.default_rng(1000 + seed)

    # --- accuracy on rho=0.6 (training distribution) and rho=0 ---
    for rho in (0.6, 0.0):
        d = make_data(N_EVAL, rho, rng)
        rep, act = labels(d, w, world, rng)
        _, lr, la = states(net, d)
        # for mixtures the label is stochastic; report accuracy vs. the deterministic components
        res[f"act_acc_rho{rho}"] = float((la.argmax(-1)[:, T0:] == act[:, T0:]).mean())
        nh = d["hide"] == 0
        pr = lr[nh][:, T0:, :2].argmax(-1) == 0
        res[f"rep_agree_mood_rho{rho}"] = float((pr == (d["m"][nh][:, T0:] >= 0)).mean())
        res[f"rep_agree_cue_rho{rho}"] = float((pr == (d["c"][nh][:, T0:] > 0)).mean())
        res[f"neutral_under_hide_rho{rho}"] = float((lr[~nh][:, T0:].argmax(-1) == 2).mean())

    # --- behavioural ground truth: conflict trials at rho=0 (mood sign != cue sign) ---
    d0 = make_data(N_EVAL, 0.0, rng, hide_p=0.0)
    h0, lr0, la0 = states(net, d0)
    ms = d0["m"][:, T0:] >= 0; cs = d0["c"][:, T0:] > 0
    conf = ms != cs
    pr = lr0[:, T0:, 0] > lr0[:, T0:, 1]
    res["conflict_follow_mood"] = float((pr[conf] == ms[conf]).mean())

    # --- probes (rho = 0, no hide) ---
    Hf = flat(h0)
    res["probe_mood"] = probe_acc(Hf, flat(d0["m"]) >= 0, rng)
    res["probe_cue"] = probe_acc(Hf, flat(d0["c"]) > 0, rng)
    res["probe_resource"] = probe_acc(Hf, flat(d0["r"]) > 0, rng)
    res["probe_report"] = probe_acc(Hf, flat(lr0[:, :, 0] - lr0[:, :, 1]) > 0, rng)
    # concealment: mood decodable when display rule forces neutral report?
    dh = make_data(N_EVAL, 0.0, rng, hide_p=1.0)
    hh, lrh, lah = states(net, dh)
    res["probe_mood_hidden"] = probe_acc(flat(hh), flat(dh["m"]) >= 0, rng)
    acth = labels(dh, w, world, rng)[1]
    res["act_acc_hidden"] = float((lah.argmax(-1)[:, T0:] == acth[:, T0:]).mean())

    # --- reference "situation" direction: mood pattern at rho=0 ---
    lr_f = flat(lr0[:, :, 0] - lr0[:, :, 1]); la_f = flat(la0[:, :, 1] - la0[:, :, 0])
    sd_r, sd_a = lr_f.std(), la_f.std()
    mpos = flat(d0["m"]) >= 0
    p_m = Hf[mpos].mean(0) - Hf[~mpos].mean(0)
    res["dr_pm"] = float(vr @ p_m); res["da_pm"] = float(va @ p_m)

    # --- expression ("emotion vector") patterns estimated at different rho ---
    res["sweep"] = []
    for rho in RHOS:
        d = make_data(N_EVAL, rho, rng, hide_p=0.0)
        h, lr, la = states(net, d)
        Hr = flat(h); happy = flat(lr[:, :, 0] - lr[:, :, 1]) > 0
        p = Hr[happy].mean(0) - Hr[~happy].mean(0)
        u = p / np.linalg.norm(p)
        alpha = (Hr @ u).std()                       # natural scale of the direction
        dr, da = float(vr @ p), float(va @ p)
        # delayed effect: inject at t=12, read out at t=15 (propagation through recurrence)
        x = torch.from_numpy(d["X"])
        with torch.no_grad():
            hs_a, _ = net.gru(x[:, :13])
            h12 = hs_a[:, -1] + torch.tensor(alpha * u, dtype=torch.float32)
            hs_b, _ = net.gru(x[:, 13:16], h12[None].contiguous())
            hs_c, _ = net.gru(x[:, 13:16], hs_a[:, -1][None].contiguous())
            r_b, a_b = net.rep(hs_b[:, -1]).numpy(), net.act(hs_b[:, -1]).numpy()
            r_c, a_c = net.rep(hs_c[:, -1]).numpy(), net.act(hs_c[:, -1]).numpy()
        ddr = float(((r_b[:, 0] - r_b[:, 1]) - (r_c[:, 0] - r_c[:, 1])).mean())
        dda = float(((a_b[:, 1] - a_b[:, 0]) - (a_c[:, 1] - a_c[:, 0])).mean())
        res["sweep"].append(dict(
            rho=rho, dr=dr, da=da, cci=da / dr,
            dr_z=float(alpha * vr @ u / sd_r), da_z=float(alpha * va @ u / sd_a),
            delayed_dr_z=float(ddr / sd_r), delayed_da_z=float(dda / sd_a),
            cos_pm=float(u @ p_m / np.linalg.norm(p_m))))
    s0 = [s for s in res["sweep"] if s["rho"] == 0.0][0]
    res["GI"] = float((s0["da"] / s0["dr"]) * (res["dr_pm"] / res["da_pm"]))
    s6 = [s for s in res["sweep"] if s["rho"] == 0.6][0]
    res["GI_naive_rho0.6"] = float((s6["da"] / s6["dr"]) * (res["dr_pm"] / res["da_pm"]))
    res["secs"] = time.time() - t0
    print(world, seed, f"{res['secs']:.0f}s GI={res['GI']:.3f} follow_mood={res['conflict_follow_mood']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(wd, w, s) for wd, w in WORLDS for s in SEEDS]
    with Pool(4) as pool:
        out = pool.starmap(analyse, jobs)
    with open(os.path.join(OUT, "raw_results.json"), "w") as f:
        json.dump(out, f, indent=1)
