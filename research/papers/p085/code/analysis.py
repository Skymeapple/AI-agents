"""Applies the cost-floor model to sourced configurations: regime map, elasticities,
lever headroom (Monte Carlo), hardware-trend projection, and sensitivity analysis.
Outputs: results/analysis_*.json, results/fig_*.png"""
import json, os, sys
from dataclasses import replace
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from cost_model import *

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SLO = 0.05  # 50 ms per decode step (time-per-output-token target)
out = {}

CONFIGS = [(LLAMA70, H100_BF16), (DSV3, replace(H100_FP8, G=16, name="16xH100 (FP8)")), (DSV3, B200_FP8)]
Ls = [1024, 4096, 16384, 65536, 131072]

# ---- Table 1: floors and regimes ----
t1 = []
for m, h in CONFIGS:
    row = dict(model=m.name, hw=h.name, kappa_bytes=m.kappa(), attn_intensity=m.attn_flops() / m.kappa(),
               machine_balance=h.Fp() / h.Bp(), W_over_M=m.W() / h.Mt(), L_star=crossover_L(m, h), floors={})
    for L in Ls:
        c, co, me, b = floor_cost(m, h, L, slo=SLO)
        c0, _, _, b0 = floor_cost(m, h, L)
        row["floors"][L] = dict(usd_per_Mtok=c * 1e6, usd_per_Mtok_noSLO=c0 * 1e6, b_eff=b,
                                regime="KV/memory" if me > co else "compute",
                                kv_share=(L * m.kappa() / h.Bp()) / max(co, me))
    t1.append(row)
out["table1"] = t1

# ---- Table 2: elasticities ----
t2 = []
for m, h in CONFIGS:
    for L in [1024, 32768]:
        e = elasticities(m, h, L, slo=SLO)
        t2.append(dict(model=m.name, hw=h.name, context=L, **{k: round(float(v), 3) for k, v in e.items()}))
out["table2_elasticities"] = t2

# ---- Figure 1: floor vs context ----
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
Lgrid = np.logspace(np.log10(256), np.log10(262144), 120)
for ax, (m, h) in zip(axes, CONFIGS):
    c = np.array([floor_cost(m, h, L, slo=SLO)[0] for L in Lgrid]) * 1e6
    comp = np.array([h.P() * floor_cost(m, h, L, slo=SLO)[1] for L in Lgrid]) * 1e6
    kv = np.array([h.P() * L * m.kappa() / h.Bp() for L in Lgrid]) * 1e6
    ax.loglog(Lgrid, c, color="#1f4e79", lw=2.2, label="floor c*(L)")
    ax.loglog(Lgrid, comp, "--", color="#c55a11", label="compute term")
    ax.loglog(Lgrid, kv, ":", color="#2e7d32", lw=2, label=r"$P L\kappa/B'$ (unamortizable)")
    Ls_ = crossover_L(m, h)
    if np.isfinite(Ls_):
        ax.axvline(Ls_, color="grey", lw=0.8); ax.text(Ls_ * 1.1, c.min() * 1.3, f"L*={Ls_:.0f}", fontsize=8)
    ax.set_title(f"{m.name}\n{h.name}", fontsize=9); ax.set_xlabel("context length L (tokens)")
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("$ per 1M output tokens (floor)"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_floor_vs_context.png"), dpi=150); plt.close()

# ---- Lever headroom: deterministic ladder + Monte Carlo ----
def apply_levers(m, h, L, kvq=1.0, arch=1.0, sparse=1.0, accept=1.0, wq=1.0, fq=1.0, util=1.0):
    """kvq: KV bytes divisor from KV quantization; arch: KV divisor from latent/shared KV;
    sparse: effective-context divisor (sparse attention, reads L/sparse tokens);
    accept: tokens per verification pass; wq: weight-bytes divisor; fq: compute-throughput
    multiplier (lower-precision math); util: multiplier on achieved eta_F and eta_B."""
    m2 = replace(m, q_kv=m.q_kv / (kvq * arch), q_w=m.q_w / wq)
    h2 = replace(h, F=h.F * fq, eta_F=min(0.95, h.eta_F * util), eta_B=min(0.95, h.eta_B * util))
    Le = L / sparse
    # 'arch' divides kappa without changing attention FLOPs per context token (e.g. fewer KV
    # heads or cross-layer KV sharing). MLA proper also RAISES attention FLOPs; see DSV3 rows.
    c = floor_cost(m2, h2, Le, slo=SLO)
    if accept > 1:
        # verify tau positions per pass to emit `accept` tokens on average; 10% drafting
        # overhead.  A rational operator uses it only when it lowers cost.
        tau = int(np.ceil(accept * 1.6))
        cs = floor_cost(m2, h2, Le, accept=accept, tau=tau, draft_overhead=0.1, slo=SLO)
        if cs[0] < c[0]:
            c = cs
    return c

ladder = []
for L in [4096, 32768]:
    base = apply_levers(LLAMA70, H100_BF16, L)[0]
    steps = [("baseline (BF16 W, BF16 KV, GQA-8)", {}),
             ("+ weights FP8 (W/2)", dict(wq=2)),
             ("+ KV 4-bit (kappa/4)", dict(wq=2, kvq=4)),
             ("+ KV-architecture compression (kappa/3.56)", dict(wq=2, kvq=4, arch=3.56)),
             ("+ sparse attention (L_eff = L/4)", dict(wq=2, kvq=4, arch=3.56, sparse=4)),
             ("+ multi-token verify (2.5 tok/pass)", dict(wq=2, kvq=4, arch=3.56, sparse=4, accept=2.5)),
             ("+ FP8 compute (F x2)", dict(wq=2, kvq=4, arch=3.56, sparse=4, accept=2.5, fq=2)),
             ("+ utilization x1.15", dict(wq=2, kvq=4, arch=3.56, sparse=4, accept=2.5, fq=2, util=1.15))]
    for name, kw in steps:
        c, co, me, b = apply_levers(LLAMA70, H100_BF16, L, **kw)
        ladder.append(dict(L=L, step=name, usd_per_Mtok=c * 1e6, cumulative_factor=base / c,
                           regime="KV/memory" if me > co else "compute", b_eff=b))
out["ladder"] = ladder

single = []
for L in [1024, 4096, 32768, 131072]:
    base = apply_levers(LLAMA70, H100_BF16, L)[0]
    row = dict(L=L, baseline_usd_per_Mtok=base * 1e6)
    for name, kw in [("weights 4-bit (W/4)", dict(wq=4)), ("KV 4-bit (kappa/4)", dict(kvq=4)),
                     ("KV-arch compression (kappa/3.56)", dict(arch=3.56)), ("sparse attn (L/4)", dict(sparse=4)),
                     ("multi-token verify (2.5/pass)", dict(accept=2.5)), ("FP8 compute (F x2)", dict(fq=2)),
                     ("2x cheaper bandwidth+FLOPs (hardware gen at equal price)", dict(fq=2, util=1.0))]:
        if name.startswith("2x"):
            h2 = replace(H100_BF16, F=H100_BF16.F * 2, B=H100_BF16.B * 2, M=H100_BF16.M * 2)
            c = floor_cost(LLAMA70, h2, L, slo=SLO)[0]
        else:
            c = apply_levers(LLAMA70, H100_BF16, L, **kw)[0]
        row[name] = base / c
    single.append(row)
out["single_levers"] = single
# MLA model: how much KV-side headroom is left once attention intensity is near machine balance?
dsv = []
for L in [4096, 32768, 131072]:
    base = floor_cost(DSV3, B200_FP8, L, slo=SLO)
    r = dict(L=L, base_usd_per_Mtok=base[0] * 1e6, regime="KV/memory" if base[2] > base[1] else "compute")
    for div in [2, 4, 8]:
        c = floor_cost(replace(DSV3, q_kv=DSV3.q_kv / div), B200_FP8, L, slo=SLO)
        r[f"kappa/{div}"] = base[0] / c[0]
        r[f"kappa/{div}_regime"] = "KV/memory" if c[2] > c[1] else "compute"
    c = floor_cost(replace(DSV3, q_w=0.5), B200_FP8, L, slo=SLO)
    r["weights FP4"] = base[0] / c[0]
    dsv.append(r)
out["dsv3_b200_kv_levers"] = dsv

rng = np.random.default_rng(42)
def logu(lo, hi, n): return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
N = 20000
mc = {}
for L in [4096, 32768, 131072]:
    base = apply_levers(LLAMA70, H100_BF16, L)[0]
    s = dict(kvq=logu(1.5, 8, N), arch=logu(1, 4, N), sparse=logu(1, 8, N), accept=logu(1.2, 3.5, N),
             wq=logu(1, 4, N), fq=logu(1, 4, N), util=logu(1, 1.2, N))
    fac, kvbound = np.empty(N), np.empty(N, bool)
    for i in range(N):
        c, co, me, _ = apply_levers(LLAMA70, H100_BF16, L, **{k: v[i] for k, v in s.items()})
        fac[i] = base / c; kvbound[i] = me > co
    # rank correlation of each lever with total factor (sensitivity)
    from scipy.stats import spearmanr
    sens = {k: float(spearmanr(np.log(v), np.log(fac))[0]) for k, v in s.items()}
    mc[L] = dict(median=float(np.median(fac)), p05=float(np.percentile(fac, 5)), p95=float(np.percentile(fac, 95)),
                 frac_still_kv_bound=float(kvbound.mean()), spearman=sens)
    if L == 32768:
        fac32 = fac
out["montecarlo_algorithmic_headroom_llama70_h100"] = mc
plt.figure(figsize=(5, 3.4))
plt.hist(np.log10(fac32), bins=60, color="#1f4e79", alpha=0.85)
plt.xlabel("log10 cost-floor reduction factor (L=32k)"); plt.ylabel("samples")
plt.title("Algorithmic headroom on fixed hardware (Monte Carlo)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_mc_headroom.png"), dpi=150); plt.close()

# ---- Hardware-per-dollar: spec+price comparison and trend projection ----
hw_cmp = []
for (nm, F8, B, Mm, plo, pmid, phi) in [("H100 SXM", 1979e12, 3.35e12, 80e9, 2.0, 2.5, 4.0),
                                        ("B200", 4.5e15, 8e12, 180e9, 4.0, 6.0, 10.0)]:
    hw_cmp.append(dict(device=nm, price_range=[plo, pmid, phi],
                       TBps_per_usd_hr=[B / 1e12 / p for p in (phi, pmid, plo)],
                       PFLOPs_fp8_per_usd_hr=[F8 / 1e15 / p for p in (phi, pmid, plo)]))
out["hw_per_dollar"] = hw_cmp
# trend: Gholami et al. 2024: peak FLOPS 3.0x/2yr, DRAM BW 1.6x/2yr (per device). Assume
# constant real price per device (an assumption, varied below).
gF, gB = 3.0 ** 0.5, 1.6 ** 0.5
proj = dict(compute_floor_halving_years=float(np.log(2) / np.log(gF)),
            kv_floor_halving_years=float(np.log(2) / np.log(gB)),
            kv_floor_factor_per_year=float(gB), compute_floor_factor_per_year=float(gF))
for dp in [0.9, 1.0, 1.1]:  # device price drift per year
    proj[f"kv_factor_per_year_price_drift_{dp}"] = float(gB / dp)
out["hw_trend"] = proj

# ---- Per-task cost: quadratic growth with reasoning length ----
task = []
for T in [1000, 4000, 16000, 64000]:
    Lgrid_t = np.arange(1000, 1000 + T, max(1, T // 400))
    cs = np.array([floor_cost(LLAMA70, H100_BF16, L, slo=SLO)[0] for L in Lgrid_t])
    task.append(dict(T_tokens=T, usd_per_task=float(cs.mean() * T), usd_per_token_avg_per_M=float(cs.mean() * 1e6)))
out["per_task"] = task

# ---- Sensitivity of headline quantities to eta, price, SLO ----
sens = []
for eF in [0.4, 0.6, 0.8]:
    for eB in [0.6, 0.8, 0.9]:
        h = replace(H100_BF16, eta_F=eF, eta_B=eB)
        sens.append(dict(eta_F=eF, eta_B=eB, L_star=crossover_L(LLAMA70, h),
                         floor_32k=floor_cost(LLAMA70, h, 32768, slo=SLO)[0] * 1e6,
                         floor_2k=floor_cost(LLAMA70, h, 2048, slo=SLO)[0] * 1e6))
out["sensitivity_eta"] = sens
out["sensitivity_slo"] = [dict(slo_ms=(s * 1e3 if s else None), floor_1k=floor_cost(LLAMA70, H100_BF16, 1024, slo=s)[0] * 1e6,
                               floor_32k=floor_cost(LLAMA70, H100_BF16, 32768, slo=s)[0] * 1e6,
                               b_1k=floor_cost(LLAMA70, H100_BF16, 1024, slo=s)[3])
                          for s in [0.02, 0.03, 0.05, 0.1, None] if True]
for r, s in zip(out["sensitivity_slo"], [0.02, 0.03, 0.05, 0.1, None]):
    if s is None: r["slo_ms"] = None
out["price_note"] = "Floors scale linearly with $/device-hour (baseline H100 $2.5, B200 $6.0)."

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
json.dump(clean(out), open(os.path.join(RES, "analysis_results.json"), "w"), indent=1)
print(json.dumps(clean(out), indent=1)[:12000])
