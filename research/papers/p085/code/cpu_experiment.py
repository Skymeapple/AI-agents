"""CPU-scale mechanism experiment for the decode cost-floor model.

A toy decoder stack (numpy/OpenBLAS, fp32) runs one decode step for a batch of
b sequences, each with L tokens of cached keys/values.  We measure wall time per
step and per generated token, and compare against a roofline prediction built
from *independently* calibrated machine parameters (GEMM FLOP/s, GEMV bandwidth).

Tests (all predictions of the analytic model in cost_model.py):
  T1  cost/token falls with batch b but saturates at a floor proportional to L
  T2  the floor scales with KV bytes per token (n_kv heads), not with FLOPs
  T3  at long context the floor is insensitive to weight bytes (FFN width)
  T4  verifying tau tokens per pass (speculative/multi-token) divides the KV term
Outputs: results/cpu_*.json
"""
import json, time, os, sys
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(0)
DT = np.float32
BYTES = 4

def timeit(f, reps=5, warm=1):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.min(ts))

# ---------------- machine calibration ----------------
def calibrate():
    n = 2048
    A = rng.standard_normal((n, n), dtype=DT); Bm = rng.standard_normal((n, n), dtype=DT)
    tm, _ = timeit(lambda: A @ Bm, reps=5)
    F = 2 * n**3 / tm
    M = rng.standard_normal((8192, 16384), dtype=DT)  # 512 MB  > L3
    v = rng.standard_normal(16384, dtype=DT)
    tv, _ = timeit(lambda: M @ v, reps=7)
    Bw = M.nbytes / tv
    # small-row batched matmul (attention-like shape) bandwidth, reported only
    K = rng.standard_normal((32, 16, 64, 4096), dtype=DT)  # 512MB
    q = rng.standard_normal((32, 16, 1, 64), dtype=DT)
    ta, _ = timeit(lambda: q @ K, reps=5)
    Battn = K.nbytes / ta
    # full attention kernel (QK^T, softmax, PV) on a synthetic cache, n_kv=16, g=1
    Kc = np.full((16, 16, 64, 2048), 0.01, dtype=DT); Vc = np.full((16, 16, 2048, 64), 0.01, dtype=DT)
    def attn():
        sc = q[:16] @ Kc; sc -= sc.max(-1, keepdims=True); np.exp(sc, out=sc)
        sc /= sc.sum(-1, keepdims=True); return sc @ Vc
    tk, _ = timeit(attn, reps=7)
    Bkv = (Kc.nbytes + Vc.nbytes) / tk
    del A, Bm, M, K, Kc, Vc
    return dict(F_gemm=F, B_gemv=Bw, B_batched_rowvec=Battn, B_attn_kernel=Bkv)

# ---------------- toy model ----------------
class Toy:
    def __init__(s, d=1024, n_heads=16, d_h=64, n_layers=8, ffn=2816, n_kv=16):
        s.d, s.nh, s.dh, s.nl, s.ffn, s.nkv = d, n_heads, d_h, n_layers, ffn, n_kv
        s.g = n_heads // n_kv
        s.layers = []
        for _ in range(n_layers):
            s.layers.append(dict(
                wq=(rng.standard_normal((d, n_heads * d_h), dtype=DT) * 0.02),
                wkv=(rng.standard_normal((d, 2 * n_kv * d_h), dtype=DT) * 0.02),
                wo=(rng.standard_normal((n_heads * d_h, d), dtype=DT) * 0.02),
                w13=(rng.standard_normal((d, 2 * ffn), dtype=DT) * 0.02),
                w2=(rng.standard_normal((ffn, d), dtype=DT) * 0.02)))
    def weight_bytes(s):
        return sum(w.nbytes for l in s.layers for w in l.values())
    def kv_bytes_per_token(s):
        return 2 * s.nl * s.nkv * s.dh * BYTES
    def alloc_cache(s, b, L):
        s.K = [np.full((b, s.nkv, s.dh, L), 0.01, dtype=DT) for _ in range(s.nl)]
        s.V = [np.full((b, s.nkv, L, s.dh), 0.01, dtype=DT) for _ in range(s.nl)]
    def free_cache(s):
        s.K = s.V = None
    def step(s, x, tau=1):
        # x: (b*tau, d). One decode/verify pass; new KV is computed but the cache
        # write is omitted (it is O(b*tau*kappa), negligible vs reading b*L*kappa).
        b = x.shape[0] // tau
        for li, l in enumerate(s.layers):
            q = x @ l["wq"]; kv = x @ l["wkv"]
            q = q.reshape(b, tau, s.nkv, s.g, s.dh).transpose(0, 2, 1, 3, 4).reshape(b, s.nkv, tau * s.g, s.dh)
            sc = q @ s.K[li]                     # (b,nkv,tau*g,L)
            sc -= sc.max(-1, keepdims=True)
            np.exp(sc, out=sc); sc /= sc.sum(-1, keepdims=True)
            o = sc @ s.V[li]                     # (b,nkv,tau*g,dh)
            o = o.reshape(b, s.nkv, tau, s.g, s.dh).transpose(0, 2, 1, 3, 4).reshape(b * tau, s.nh * s.dh)
            x = x + o @ l["wo"]
            h = x @ l["w13"]; a, c = h[:, :s.ffn], h[:, s.ffn:]
            x = x + ((a / (1 + np.exp(-a))) * c) @ l["w2"]
        return x
    def flops(s, b, L, tau=1):
        n = b * tau
        per_layer_w = 2 * n * (s.d * s.nh * s.dh + s.d * 2 * s.nkv * s.dh + s.nh * s.dh * s.d + s.d * 2 * s.ffn + s.ffn * s.d)
        per_layer_attn = 2 * 2 * n * s.nh * s.dh * L
        return s.nl * (per_layer_w + per_layer_attn), s.nl * per_layer_w, s.nl * per_layer_attn

def predict(m, cal, b, L, tau=1):
    """Operator-level roofline: weight GEMMs (GEMV-calibrated bandwidth) and attention
    (attention-kernel-calibrated bandwidth), each max(compute, memory)."""
    F, B, Bkv = cal["F_gemm"], cal["B_gemv"], cal["B_attn_kernel"]
    _, fw, fa = m.flops(b, L, tau)
    W = m.weight_bytes(); kvb = b * L * m.kv_bytes_per_token()
    t_w = max(fw / F, W / B)
    t_a = max(fa / F, kvb / Bkv)
    return t_w + t_a, t_w, t_a

def run_point(m, cal, b, L, tau=1, reps=7):
    m.alloc_cache(b, L)
    x = rng.standard_normal((b * tau, m.d), dtype=DT) * 0.1
    tmed, tmin = timeit(lambda: m.step(x, tau), reps=reps)
    m.free_cache()
    pred, tw, ta = predict(m, cal, b, L, tau)
    return dict(b=b, L=L, tau=tau, n_kv=m.nkv, ffn=m.ffn, W_bytes=m.weight_bytes(),
                kappa=m.kv_bytes_per_token(), t_step=tmed, t_step_min=tmin,
                t_per_token=tmed / (b * tau), pred_step=pred, pred_w=tw, pred_attn=ta,
                pred_per_token=pred / (b * tau))

def main(rep=0):
    t0 = time.time()
    global rng
    rng = np.random.default_rng(rep)
    cal = calibrate(); print("calibration", cal, flush=True)
    json.dump(cal, open(os.path.join(RES, f"cpu_calibration_rep{rep}.json"), "w"), indent=1)
    rows = []
    MAXTOK = 32768
    Ls = [256, 512, 1024, 2048, 4096]
    bs = [1, 2, 4, 8, 16, 32, 64, 128]
    for nkv in [16, 4, 1]:
        m = Toy(n_kv=nkv)
        cap = MAXTOK * 16 // nkv if nkv > 1 else MAXTOK * 4  # equal-ish cache memory
        cap = min(cap, 4 * MAXTOK)
        # weight-stage time t_w(b): same model with a 1-token cache
        for b in bs:
            r = run_point(m, cal, b, 1); r["exp"] = "W"; rows.append(r)
        for L in Ls:
            for b in bs:
                if b * L > cap:
                    continue
                if L not in (1024, 4096) and b * 2 * L <= cap:
                    continue  # for slope fits we only need the largest batch at other L
                r = run_point(m, cal, b, L); r["exp"] = "T1T2"; rows.append(r)
                print(f"nkv={nkv} L={L} b={b} t/tok={r['t_per_token']*1e3:.2f}ms pred={r['pred_per_token']*1e3:.2f}ms", flush=True)
        del m
    for ffn in [1408, 2816, 5632]:
        m = Toy(n_kv=16, ffn=ffn)
        for L in [256, 4096]:
            for b in [1, 8]:
                r = run_point(m, cal, b, L); r["exp"] = "T3"; rows.append(r)
                print(f"ffn={ffn} L={L} b={b} t/tok={r['t_per_token']*1e3:.2f}ms", flush=True)
        del m
    m = Toy(n_kv=16)
    for L in [512, 4096]:
        for tau in [1, 2, 4, 8, 16]:
            r = run_point(m, cal, 8, L, tau=tau); r["exp"] = "T4"; rows.append(r)
            print(f"tau={tau} L={L} t/tok={r['t_per_token']*1e3:.2f}ms pred={r['pred_per_token']*1e3:.2f}", flush=True)
    del m
    json.dump(rows, open(os.path.join(RES, f"cpu_runs_rep{rep}.json"), "w"), indent=1)
    print("done in %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
