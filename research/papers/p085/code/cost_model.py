"""Analytic decode cost-floor model (Section 3 of the paper).

All quantities are per serving replica of G accelerators with ideal tensor parallelism.
c(b) = P * T(b) / b,  T(b) = max( b*(2*Na + a*L)/F',  (W + b*L*kappa)/B' ),
subject to W + b*L*kappa <= M.  The floor is attained at b_max = (M-W)/(L*kappa).
"""
from dataclasses import dataclass, replace
import numpy as np

@dataclass
class Model:
    name: str
    N_total: float      # parameters
    N_active: float     # active parameters per token
    n_layers: int
    n_heads: int
    d_head: int
    n_kv: int = 0       # GQA kv heads (0 => MLA)
    d_c: int = 0        # MLA latent dim
    d_r: int = 0        # MLA decoupled rope dim
    q_w: float = 2.0    # bytes / weight
    q_kv: float = 2.0   # bytes / KV element

    def kappa(self):
        """KV-cache bytes per context token (all layers)."""
        if self.n_kv:
            return 2 * self.n_layers * self.n_kv * self.d_head * self.q_kv
        return self.n_layers * (self.d_c + self.d_r) * self.q_kv

    def attn_flops(self):
        """Attention FLOPs per context token per generated token (QK^T and PV)."""
        if self.n_kv:
            return 4 * self.n_layers * self.n_heads * self.d_head
        # MLA with weight absorption: scores over (d_c+d_r), values over d_c, per head
        return 2 * self.n_layers * self.n_heads * (self.d_c + self.d_r + self.d_c)

    def W(self):
        return self.N_total * self.q_w

@dataclass
class HW:
    name: str
    F: float            # peak dense FLOP/s per device at the compute precision used
    B: float            # HBM bytes/s per device
    M: float            # HBM bytes per device
    price_hr: float     # $ per device-hour
    G: int = 8
    eta_F: float = 0.6
    eta_B: float = 0.8
    def Fp(self): return self.G * self.F * self.eta_F
    def Bp(self): return self.G * self.B * self.eta_B
    def Mt(self): return self.G * self.M
    def P(self): return self.G * self.price_hr / 3600.0

def step_time(m, h, b, L, tau=1):
    comp = b * tau * (2 * m.N_active + m.attn_flops() * L) / h.Fp()
    mem = (m.W() + b * L * m.kappa()) / h.Bp()
    return np.maximum(comp, mem), comp, mem

def b_max(m, h, L):
    return (h.Mt() - m.W()) / (L * m.kappa())

def floor_cost(m, h, L, accept=1.0, tau=1, draft_overhead=0.0, slo=None):
    """Cost per generated token at the largest feasible batch (continuous relaxation).
    Feasible: memory capacity W + b L kappa <= M and, if slo (seconds per decode step)
    is given, T(b) <= slo.  accept: expected tokens emitted per verification pass
    (speculative / multi-token); tau: positions verified per pass; draft_overhead:
    extra time fraction for drafting.  Returns (cost $/token, compute-term s/token,
    memory-term s/token, b_eff)."""
    M, W, Fp, Bp = h.Mt(), m.W(), h.Fp(), h.Bp()
    if W >= M:
        return np.inf, np.inf, np.inf, 0.0
    per_seq_flops = tau * (2 * m.N_active + m.attn_flops() * L)
    b_cap = (M - W) / (L * m.kappa())
    b = b_cap
    if slo is not None:
        if W >= slo * Bp:
            return np.inf, np.inf, np.inf, 0.0
        b = min(b, (slo * Bp - W) / (L * m.kappa()), slo * Fp / per_seq_flops)
    comp = per_seq_flops / Fp
    mem = (W / b + L * m.kappa()) / Bp
    t = max(comp, mem) * (1 + draft_overhead)
    return h.P() * t / accept, comp, mem, b

def crossover_L(m, h):
    """Context length above which the floor is memory(KV)-bound. inf if never."""
    M, W = h.Mt(), m.W()
    s_mem = m.kappa() * M / ((M - W) * h.Bp())
    s_comp = m.attn_flops() / h.Fp()
    if s_mem <= s_comp:
        return np.inf
    return (2 * m.N_active / h.Fp()) / (s_mem - s_comp)

def elasticities(m, h, L, eps=1e-4, slo=None):
    """Numerical log-derivatives of the floor cost wrt model/hardware quantities."""
    base = floor_cost(m, h, L, slo=slo)[0]
    out = {}
    def d(newm=None, newh=None, newL=None):
        c = floor_cost(newm or m, newh or h, newL or L, slo=slo)[0]
        return (np.log(c) - np.log(base)) / np.log(1 + eps)
    out["kappa(q_kv)"] = d(newm=replace(m, q_kv=m.q_kv * (1 + eps)))
    out["W(q_w)"] = d(newm=replace(m, q_w=m.q_w * (1 + eps)))
    out["N_active"] = d(newm=replace(m, N_active=m.N_active * (1 + eps)))
    out["F"] = d(newh=replace(h, F=h.F * (1 + eps)))
    out["B"] = d(newh=replace(h, B=h.B * (1 + eps)))
    out["M"] = d(newh=replace(h, M=h.M * (1 + eps)))
    out["L"] = d(newL=L * (1 + eps))
    return out

# ---------------- sourced configurations ----------------
# Llama-3-70B: 80 layers, 64 heads, 8 KV heads, head dim 128 (HF config.json).
LLAMA70 = Model("Llama-3-70B (GQA-8)", 70.6e9, 70.6e9, 80, 64, 128, n_kv=8, q_w=2, q_kv=2)
# DeepSeek-V3: 671B total / 37B active, 61 layers, 128 heads, kv_lora_rank 512, rope dim 64.
DSV3 = Model("DeepSeek-V3 (MLA)", 671e9, 37e9, 61, 128, 128, n_kv=0, d_c=512, d_r=64, q_w=1, q_kv=2)
# H100 SXM: 989 TFLOP/s dense BF16 (1979 dense FP8), 3.35 TB/s, 80 GB.
H100_BF16 = HW("8xH100 (BF16)", 989e12, 3.35e12, 80e9, 2.5, G=8)
H100_FP8 = HW("8xH100 (FP8)", 1979e12, 3.35e12, 80e9, 2.5, G=8)
# B200: 4.5 PFLOP/s dense FP8, 8 TB/s, 192 GB nominal (180 GB usable reported).
B200_FP8 = HW("8xB200 (FP8)", 4.5e15, 8e12, 180e9, 6.0, G=8)
