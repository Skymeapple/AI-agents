"""Run every experiment in the paper. Usage:  python3 run_all.py  (about 10 min CPU)
Outputs: ../results/results.json, ../results/*.png, ../results/log.txt
"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testbed import (ReferenceAgent, WinMLP, SplitMLP, GRUMimic, BottleMLP, sample_episodes,
                     train_mimic, run_agent, report_gradient_direction, LAM, G, B0, B1)

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES, exist_ok=True)
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 80
CLASSES = {"WinMLP": WinMLP, "SplitMLP": SplitMLP, "GRU": GRUMimic}
sig = lambda z: 1 / (1 + np.exp(-z))
PROBE_D, PROBE_R0 = 0.6, 0.3


def set_probe(x, t, delta):
    x[:, t, 1] = PROBE_R0 + delta; x[:, t, 2] = PROBE_D
    x[:, t, 3] = PROBE_R0; x[:, t, 4] = 0.0


def controlled_episodes(n, T, shock_times, rng):
    x = sample_episodes(n, T, rng, p_shock=0.0)
    for st in shock_times:
        x[:, st, 0] = 1.0
    return x


def bkl(pa, pm):
    e = 1e-7
    pa = np.clip(pa, e, 1 - e); pm = np.clip(pm, e, 1 - e)
    return pa * np.log(pa / pm) + (1 - pa) * np.log((1 - pa) / (1 - pm))


# ------------------------------------------------------------------ tests
def test_fidelity(agent, A, rng):
    """In-distribution fidelity and Proposition-1 check (L = 12)."""
    x = sample_episodes(4000, 12, rng)
    la, ra, _ = A.forward(x)
    lm, rm, _ = run_agent(agent, x)
    pa, pm = sig(la), sig(lm)
    eps = float(bkl(pa, pm).mean())
    # D-distributed trajectory tests with values in [0,1]
    hi1 = (x[..., 2] > x[..., 4])  # option 1 is the more noxious one
    pa_hi = np.where(hi1, pa, 1 - pa); pm_hi = np.where(hi1, pm, 1 - pm)
    T1a, T1m = pa_hi.mean(), pm_hi.mean()          # mean P(choose more-noxious)
    s = A.states(x)
    post = s > 0.2
    T2a = (pa_hi * post).sum() / post.sum(); T2m = (pm_hi * post).sum() / post.sum()
    # report-based test: mean clipped report (in [0,1])
    T3a, T3m = np.clip(ra, 0, 1).mean(), np.clip(rm, 0, 1).mean()
    corr_r = float(np.corrcoef(ra.ravel(), rm.ravel())[0, 1])
    return dict(eps=eps, tv_bound=float(np.sqrt(12 * eps / 2)),
                rep_mse=float(np.mean((ra - rm) ** 2)), rep_corr=corr_r,
                T_noxious_gap=float(abs(T1a - T1m)), T_postshock_gap=float(abs(T2a - T2m)),
                T_report_gap=float(abs(T3a - T3m)))


def test_tradeoff(agent, rng, n=600):
    """Elwood/Keeling-style trade-off: indifference reward premium for accepting
    the noxious option, after 0, 1 or 2 recent shocks (probe at t=11)."""
    grid = np.round(np.arange(-0.2, 1.6, 0.02), 3)
    out = {}
    for name, shocks in [("none", []), ("one", [9]), ("two", [8, 9])]:
        base = controlled_episodes(n, 12, shocks, rng)
        P, Rp = [], []
        for d in grid:
            x = base.copy(); set_probe(x, 11, d)
            l, r, _ = run_agent(agent, x)
            P.append(sig(l[:, 11]).mean()); Rp.append(r[:, 11].mean())
        P = np.array(P)
        idx = np.where(P >= 0.5)[0]
        if len(idx) == 0 or idx[0] == 0:
            ind = float("nan")
        else:
            i = idx[0]
            ind = float(grid[i - 1] + (0.5 - P[i - 1]) * (grid[i] - grid[i - 1]) / (P[i] - P[i - 1]))
        out[name] = dict(indiff=ind, report=float(np.mean(Rp)), curve=P.tolist())
    out["grid"] = grid.tolist()
    return out


LAGS = [0, 2, 5, 8, 11, 15, 20, 25, 30, 40, 50]


def test_persistence(agent, rng, n=400, delta=0.45, max_t=None):
    """Single shock at t=0, probe at lag l. Response = shock - no-shock difference."""
    res = {"lags": LAGS, "d_report": [], "d_avoid": []}
    for lag in LAGS:
        T = lag + 1
        b = sample_episodes(n, T, rng, p_shock=0.0)
        set_probe(b, lag, delta)
        xs = b.copy(); xs[:, 0, 0] = 1.0
        l0, r0, _ = run_agent(agent, b); l1, r1, _ = run_agent(agent, xs)
        res["d_report"].append(float((r1[:, lag] - r0[:, lag]).mean()))
        res["d_avoid"].append(float((sig(l0[:, lag]) - sig(l1[:, lag])).mean()))
    return res


def ridge(X, y, lam):
    Xc = X - X.mean(0); yc = y - y.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w


def test_steering(agent, A, rng, n=2000, delta=0.45, target=0.5):
    """Interventional coupling: steer activations along (i) a linear probe for the
    report and (ii) the report-gradient direction, so the report rises by `target`;
    compare the induced change in choice logit with the natural logit-report slope."""
    x = sample_episodes(n, 12, rng); set_probe(x, 11, delta)
    l, r, h = run_agent(agent, x)
    Ln, Rn, Hn = l[:, 11], r[:, 11], h[:, 11]
    nat_slope = float(np.polyfit(Rn, Ln, 1)[0])
    # probe on held-out half
    w = ridge(Hn[: n // 2], Rn[: n // 2], 1e-2 * n / 2)
    pred = (Hn[n // 2:] - Hn[: n // 2].mean(0)) @ w + Rn[: n // 2].mean()
    probe_r2 = float(1 - np.mean((pred - Rn[n // 2:]) ** 2) / np.var(Rn[n // 2:]))
    dirs = {"probe": w / np.linalg.norm(w),
            "gradient": None}
    g = report_gradient_direction(agent, x)
    dirs["gradient"] = g / (np.linalg.norm(g) + 1e-12)
    # steering evaluated on no-shock episodes
    xb = controlled_episodes(500, 12, [], rng); set_probe(xb, 11, delta)
    l0, r0, h0 = run_agent(agent, xb)
    out = {"natural_slope": nat_slope, "probe_r2": probe_r2}
    # matched-level (local) natural slope: natural trajectories whose report is near the
    # unsteered baseline vs near baseline + target (robust to logit saturation, Sec. 5.6)
    rb = float(r0[:, 11].mean()); hw = 0.08
    lo = np.abs(Rn - rb) < hw; hi = np.abs(Rn - (rb + target)) < hw
    if lo.sum() > 10 and hi.sum() > 10:
        loc_slope = float((Ln[hi].mean() - Ln[lo].mean()) / (Rn[hi].mean() - Rn[lo].mean()))
    else:
        loc_slope = float("nan")
    out["local_slope"] = loc_slope; out["n_local"] = [int(lo.sum()), int(hi.sum())]
    scale = float(np.sqrt((Hn.var(0)).sum()))
    for dn, u in dirs.items():
        found = None
        for a in np.linspace(0, 6, 241)[1:] * scale:
            st = np.zeros((xb.shape[0], 12, Hn.shape[1]), np.float32); st[:, 11] = a * u
            l1, r1, _ = run_agent(agent, xb, steer=st)
            dr = (r1[:, 11] - r0[:, 11]).mean()
            if dr >= target:
                dl = (l1[:, 11] - l0[:, 11]).mean()
                found = dict(alpha_over_scale=float(a / scale), d_report=float(dr),
                             d_logit=float(dl), slope=float(dl / dr),
                             rho=float((dl / dr) / nat_slope),
                             rho_local=float((dl / dr) / loc_slope))
                break
        out[dn] = found if found else dict(rho=float("nan"), rho_local=float("nan"), note="report never reached target")
    return out


def test_suppression(model, A, seed, rng):
    """Fine-tune the whole mimic to report 0 ('I am fine') while still cloning A's
    actions. Measure report, behaviour and internal encoding before/after."""
    def measure(m):
        x = sample_episodes(3000, 12, rng)
        la, ra, _ = A.forward(x); lm, rm, hm = run_agent(m, x)
        s = A.states(x)
        H = hm.reshape(-1, hm.shape[-1]); sv = s.ravel(); nn_ = len(sv) // 2
        w = ridge(H[:nn_], sv[:nn_], 1e-2 * nn_)
        pred = (H[nn_:] - H[:nn_].mean(0)) @ w + sv[:nn_].mean()
        r2 = float(1 - np.mean((pred - sv[nn_:]) ** 2) / np.var(sv[nn_:]))
        pers = test_persistence(m, rng, n=300)
        return dict(report_state_corr=float(np.corrcoef(rm.ravel(), sv)[0, 1]),
                    report_sd=float(rm.std()), action_kl=float(bkl(sig(la), sig(lm)).mean()),
                    d_avoid_lag2=pers["d_avoid"][1], d_report_lag2=pers["d_report"][1],
                    state_probe_r2=r2)
    before = measure(model)
    import copy
    m2 = copy.deepcopy(model); m2.train()
    m2 = train_mimic(type(model), A, seed + 100, epochs=30, lr=1e-3, report_target=0.0, init=m2)
    after = measure(m2)
    return dict(before=before, after=after)


def analytic_A():
    """Closed-form values for A (sanity check)."""
    s1 = G * LAM ** 2
    s2 = G * LAM ** 2 + G * LAM ** 3
    return dict(indiff_none=PROBE_D * B0, indiff_one=PROBE_D * (B0 + B1 * s1),
                indiff_two=PROBE_D * (B0 + B1 * s2))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    log = open(os.path.join(RES, "log.txt"), "w")
    def P(*a):
        s = " ".join(str(v) for v in a); print(s); log.write(s + "\n"); log.flush()
    A = ReferenceAgent(0)
    R = {"config": dict(LAM=LAM, G=G, B0=B0, B1=B1, seeds=SEEDS, epochs=EPOCHS),
         "analytic_A": analytic_A()}
    rng = np.random.default_rng(12345)
    R["A"] = dict(tradeoff=test_tradeoff(A, rng), persistence=test_persistence(A, rng),
                  steering=test_steering(A, A, rng))
    P("A done", R["A"]["tradeoff"]["none"]["indiff"], R["A"]["tradeoff"]["one"]["indiff"],
      R["A"]["steering"]["probe"], R["A"]["steering"]["gradient"])
    models = {}
    for name, cls in CLASSES.items():
        R[name] = []
        for sd in SEEDS:
            m = train_mimic(cls, A, sd, epochs=EPOCHS)
            models[(name, sd)] = m
            rr = np.random.default_rng(sd * 7 + 1)
            e = dict(seed=sd, fidelity=test_fidelity(m, A, rr), tradeoff=test_tradeoff(m, rr),
                     persistence=test_persistence(m, rr), steering=test_steering(m, A, rr))
            m.steer_layer = 1
            e["steering_layer1"] = test_steering(m, A, np.random.default_rng(sd * 7 + 2))
            m.steer_layer = 2
            R[name].append(e)
            P(name, sd, "eps=%.4f" % e["fidelity"]["eps"],
              "ind none/one=%.3f/%.3f" % (e["tradeoff"]["none"]["indiff"], e["tradeoff"]["one"]["indiff"]),
              "avoid lag30=%.3f" % e["persistence"]["d_avoid"][LAGS.index(30)],
              "rho probe=%.3f grad=%.3f" % (e["steering"]["probe"]["rho"], e["steering"]["gradient"]["rho"]),
              "L1 rho probe=%.3f grad=%.3f" % (e["steering_layer1"]["probe"]["rho"], e["steering_layer1"]["gradient"]["rho"]),
              "t=%.0fs" % (time.time() - t0))
    # suppression experiment
    R["suppression"] = {}
    for name in ["WinMLP", "GRU"]:
        R["suppression"][name] = []
        for sd in SEEDS[:3]:
            out = test_suppression(models[(name, sd)], A, sd, np.random.default_rng(sd + 500))
            R["suppression"][name].append(out)
            P("suppress", name, sd, out["before"]["report_state_corr"], "->", out["after"]["report_state_corr"],
              "probeR2", out["before"]["state_probe_r2"], "->", out["after"]["state_probe_r2"],
              "avoid", out["before"]["d_avoid_lag2"], "->", out["after"]["d_avoid_lag2"])
    # coverage sweep: training-horizon length
    R["coverage"] = {}
    for name, cls in [("GRU", GRUMimic), ("WinMLP", WinMLP)]:
        R["coverage"][name] = {}
        for Ttr in [12, 24, 48]:
            lst = []
            for sd in SEEDS[:3]:
                m = train_mimic(cls, A, 200 + sd, T_train=Ttr, epochs=EPOCHS, bs=max(16, 256 * 12 // Ttr))  # equal data, equal gradient steps
                rr = np.random.default_rng(sd + 900)
                x = sample_episodes(1500, 48, rr); la, ra, _ = A.forward(x); lm, rm, _ = run_agent(m, x)
                eps48 = float(bkl(sig(la), sig(lm)).mean())
                lst.append(dict(seed=sd, persistence=test_persistence(m, rr), eps_T48=eps48,
                                rep_mse_T48=float(np.mean((ra - rm) ** 2))))
            R["coverage"][name][str(Ttr)] = lst
            P("coverage", name, Ttr, [round(v["persistence"]["d_avoid"][LAGS.index(30)], 3) for v in lst],
              [round(v["eps_T48"], 4) for v in lst], "t=%.0fs" % (time.time() - t0))
    # bottleneck sweep: does a low-dimensional history code induce coupling?
    R["bottleneck"] = {}
    for b in [1, 2, 4, 16, 64]:
        lst = []
        for sd in SEEDS[:3]:
            torch.manual_seed(300 + sd)
            m = train_mimic(lambda b=b: BottleMLP(b=b), A, 300 + sd, epochs=EPOCHS)
            rr = np.random.default_rng(sd + 1300)
            lst.append(dict(seed=sd, fidelity=test_fidelity(m, A, rr), steering=test_steering(m, A, rr),
                            persistence=test_persistence(m, rr)))
        R["bottleneck"][str(b)] = lst
        P("bottleneck", b, [round(v["fidelity"]["eps"], 4) for v in lst],
          [round(v["steering"]["gradient"]["rho"], 3) for v in lst],
          [round(v["steering"]["probe"]["rho"], 3) for v in lst], "t=%.0fs" % (time.time() - t0))
    R["runtime_s"] = time.time() - t0
    json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1)
    P("total runtime", R["runtime_s"])


if __name__ == "__main__":
    main()
