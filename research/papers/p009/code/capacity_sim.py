"""Experiment A: ideal (untrained) fixed-state associative memories vs. closed-form theory.

Memories (state M in R^{d_v x d_k}):
  additive  : M += u_v k^T                         (linear attention / fast weights)
  decay     : M = g*M + u_v k^T                     (scalar-gated linear attention, Mamba-2-like decay)
  delta     : M += (u_v - M k) k^T  (beta=1)        (DeltaNet)
Keys: random unit vectors in R^{d_k}; value codes: random unit vectors u_1..u_V in R^{d_v}.
Read: r = M k_i, decode argmax_v <u_v, r>.

Theory (Proposition 1, Gaussian approximation):
  correct score ~ N(s, sc^2), each of V-1 competitors ~ N(0, sw^2), independent:
  acc = int phi(z) Phi((s + sc z)/sw)^(V-1) dz
  with G = sum of squared weights of the *other* stored items,
    sc^2 = G/(d_k d_v) + G/(V d_k),  sw^2 = s^2/d_v + G/(d_k d_v) + G/(V d_k).
Outputs: results/sim_capacity.json, results/fig_sim_capacity.png
"""
import json, os, numpy as np
from scipy.stats import norm

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def theory_acc(s, G, dk, dv, V):
    sc = np.sqrt(G / (dk * dv) + G / (V * dk)) + 1e-12
    sw = np.sqrt(s ** 2 / dv + G / (dk * dv) + G / (V * dk))
    z = np.linspace(-8, 8, 4001)
    integrand = norm.pdf(z) * np.exp((V - 1) * norm.logcdf((s + sc * z) / sw))
    return float(np.trapezoid(integrand, z))


def simulate(kind, N, dk, dv, V, trials=40, g=0.95):
    accs = []
    lag_correct = np.zeros(N); lag_count = np.zeros(N)
    for _ in range(trials):
        U = unit(rng.standard_normal((V, dv)))
        K = unit(rng.standard_normal((N, dk)))
        vals = rng.integers(0, V, N)
        M = np.zeros((dv, dk))
        for t in range(N):
            k = K[t]; u = U[vals[t]]
            if kind == "additive":
                M += np.outer(u, k)
            elif kind == "decay":
                M = g * M + np.outer(u, k)
            elif kind == "delta":
                M += np.outer(u - M @ k, k)
        R = K @ M.T                     # N x dv retrieved
        pred = np.argmax(R @ U.T, axis=1)
        ok = pred == vals
        accs.append(ok.mean())
        lags = N - 1 - np.arange(N)
        lag_correct[lags] += ok; lag_count[lags] += 1
    return float(np.mean(accs)), (lag_correct / np.maximum(lag_count, 1)).tolist()


def main():
    V = 512
    out = {"V": V, "additive": [], "decay": [], "delta": [], "dv_sweep": []}
    Ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # 1) additive & delta vs N for several dk (dv=64)
    for dk in [8, 16, 32, 64]:
        for N in Ns:
            for kind in ["additive", "delta"]:
                a, _ = simulate(kind, N, dk, 64, V, trials=20 if N >= 512 else 40)
                rec = {"dk": dk, "dv": 64, "N": N, "acc_sim": a}
                if kind == "additive":
                    rec["acc_theory"] = theory_acc(1.0, N - 1, dk, 64, V)
                out[kind].append(rec)
            print(dk, N, out["additive"][-1], out["delta"][-1]["acc_sim"], flush=True)
    # 2) dv sweep at dk=16 (additive): capacity should scale ~ dk*dv/(2 ln V) for dv << V
    for dv in [16, 32, 64, 128, 256]:
        for N in Ns:
            a, _ = simulate("additive", N, 16, dv, V, trials=20)
            out["dv_sweep"].append({"dk": 16, "dv": dv, "N": N, "acc_sim": a,
                                    "acc_theory": theory_acc(1.0, N - 1, 16, dv, V)})
    # 3) decay: accuracy vs lag (N=256, dk=32, dv=64), several gammas, theory per lag
    for g in [0.9, 0.97, 0.99, 1.0]:
        N, dk, dv = 256, 32, 64
        a, lag = simulate("decay", N, dk, dv, V, trials=40, g=g)
        th = []
        for tau in range(N):
            w2 = np.array([g ** (2 * j) for j in range(N)])
            G = w2.sum() - g ** (2 * tau)
            th.append(theory_acc(g ** tau, G, dk, dv, V))
        out["decay"].append({"gamma": g, "N": N, "dk": dk, "dv": dv, "acc_sim": a,
                             "acc_by_lag_sim": lag, "acc_by_lag_theory": th})
        print("decay", g, a, flush=True)
    # empirical capacity N*(acc=0.5) by log-interpolation
    def cap(rows, key):
        rows = sorted(rows, key=lambda r: r["N"])
        for r0, r1 in zip(rows, rows[1:]):
            if r0[key] >= 0.5 > r1[key]:
                f = (r0[key] - 0.5) / (r0[key] - r1[key])
                return float(np.exp(np.log(r0["N"]) + f * (np.log(r1["N"]) - np.log(r0["N"]))))
        return None
    caps = []
    for kind in ["additive", "delta"]:
        for dk in [8, 16, 32, 64]:
            rows = [r for r in out[kind] if r["dk"] == dk]
            c = {"kind": kind, "dk": dk, "dv": 64, "Nstar_sim": cap(rows, "acc_sim")}
            if kind == "additive":
                c["Nstar_theory"] = cap(rows, "acc_theory")
                c["Nstar_rule"] = dk / (2 * np.log(V) * (1 / 64 + 1 / V))
            caps.append(c)
    for dv in [16, 32, 64, 128, 256]:
        rows = [r for r in out["dv_sweep"] if r["dv"] == dv]
        caps.append({"kind": "additive_dvsweep", "dk": 16, "dv": dv, "Nstar_sim": cap(rows, "acc_sim"),
                     "Nstar_theory": cap(rows, "acc_theory"),
                     "Nstar_rule": 16 / (2 * np.log(V) * (1 / dv + 1 / V))})
    out["capacities"] = caps
    json.dump(out, open(os.path.join(RES, "sim_capacity.json"), "w"), indent=1)
    for c in caps:
        print(c)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {8: "C0", 16: "C1", 32: "C2", 64: "C3"}
    for dk in [8, 16, 32, 64]:
        ra = [r for r in out["additive"] if r["dk"] == dk]; rd = [r for r in out["delta"] if r["dk"] == dk]
        ax[0].plot([r["N"] for r in ra], [r["acc_sim"] for r in ra], "o", color=cols[dk])
        ax[0].plot([r["N"] for r in ra], [r["acc_theory"] for r in ra], "-", color=cols[dk], label=f"additive d_k={dk}")
        ax[0].plot([r["N"] for r in rd], [r["acc_sim"] for r in rd], "x--", color=cols[dk], alpha=.6)
    ax[0].set_xscale("log"); ax[0].set_xlabel("stored pairs N"); ax[0].set_ylabel("recall accuracy")
    ax[0].set_title("(a) additive (o sim, - theory) vs delta (x--), d_v=64"); ax[0].legend(fontsize=7)
    for i, dv in enumerate([16, 32, 64, 128, 256]):
        r = [x for x in out["dv_sweep"] if x["dv"] == dv]
        ax[1].plot([x["N"] for x in r], [x["acc_sim"] for x in r], "o", color=f"C{i}")
        ax[1].plot([x["N"] for x in r], [x["acc_theory"] for x in r], "-", color=f"C{i}", label=f"d_v={dv}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("stored pairs N"); ax[1].set_title("(b) additive, d_k=16, value-dim sweep"); ax[1].legend(fontsize=7)
    for i, d in enumerate(out["decay"]):
        lag = np.arange(d["N"])
        ax[2].plot(lag, d["acc_by_lag_sim"], ".", color=f"C{i}", ms=3)
        ax[2].plot(lag, d["acc_by_lag_theory"], "-", color=f"C{i}", label=f"gamma={d['gamma']} (mean {d['acc_sim']:.2f})")
    ax[2].set_xlabel("lag (pairs since write)"); ax[2].set_title("(c) decayed memory, N=256, d_k=32"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_sim_capacity.png"), dpi=130)


if __name__ == "__main__":
    main()
