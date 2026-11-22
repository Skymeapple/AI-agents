"""Parameter priors for the headroom-budget model of frontier training compute.

Every entry is tagged:
  [S] = range taken from a cited, verified source (see paper references);
  [A] = modelling assumption (author's choice), varied in sensitivity analysis.
Growth rates are continuous (natural-log) rates per year unless stated.
"""
import numpy as np

LN = np.log

# name: (distribution, low, high, tag, note)
PRIORS = {
    # --- anchors (mid-2025) ---
    "log10_C0":   ("uniform", np.log10(3e26), np.log10(8e26), "S",
                   "Largest known run mid-2025: Grok 4 ~5e26 FLOP (Epoch AI), +/- uncertainty"),
    # --- headline trend to be decomposed ---
    "gC0":        ("uniform", LN(4.0), LN(5.0), "S",
                   "Frontier training compute growth 4-5x/yr (Sevilla & Roldan 2024; Epoch trends)"),
    "gK0":        ("uniform", LN(2.0), LN(2.9), "S",
                   "Frontier amortized training cost growth 2.4x/yr, 90% CI 2.0-2.9 (Cottier et al. 2024)"),
    # --- secular factors ---
    "gh":         ("uniform", LN(2) / 2.9, LN(2) / 2.0, "S",
                   "GPU/accelerator FLOP/s per $ doubling 2.0-2.9 yr (Hobbhahn & Besiroglu 2022; Epoch hardware trends)"),
    "ge":         ("uniform", LN(1.30), LN(1.45), "S",
                   "Energy efficiency: supercomputer FLOP/s/W 1.34x/yr (Pilz et al. 2025); leading chips ~1.4x/yr (Epoch)"),
    "gY":         ("uniform", 0.025, 0.035, "S",
                   "World real GDP growth ~3.1-3.2% (IMF WEO Oct 2025)"),
    # --- bounded share factors ---
    "gI0":        ("uniform", 0.35, 0.70, "A",
                   "Initial growth of AI compute investment; 2025->2026 big-tech capex ~410->~725 $B implies ~0.57 (press reports)"),
    "sigma_max":  ("loguniform", 0.01, 0.10, "A",
                   "Cap on AI compute investment as share of world GDP (GATE explores >10%)"),
    "phi_max":    ("loguniform", 0.01, 0.05, "A",
                   "Cap on largest run's amortized cost as share of annual AI compute investment"),
    "x_max":      ("loguniform", 1.5, 4.0, "A",
                   "Remaining headroom from lower precision / utilisation (FP8->FP4 = 2x)"),
    "gT":         ("uniform", LN(1.3), LN(1.5), "S",
                   "Training-run duration growth 1.4x/yr, 90% CI 1.3-1.5 (Epoch 'longest training run')"),
    "T_max_days": ("uniform", 180.0, 365.0, "A",
                   "Cap on run duration; Epoch argues >~9 months is economically suboptimal"),
    # --- physical caps ---
    "P2030_GW":   ("loguniform", 1.0, 16.0, "S",
                   "Power available to largest run in 2030: 1-5 GW single campus (Sevilla et al. 2024); 4-16 GW demand (EPRI/Epoch 2025)"),
    "gPmax":      ("uniform", 0.10, 0.40, "A",
                   "Growth of power deliverable to one (possibly distributed) run; IEA DC electricity +15%/yr; EPRI US AI power ~+58%/yr 2025-30"),
    "log10_Clat": ("uniform", 29.0, 32.0, "S",
                   "Data-movement/latency ceiling: 1e28-1e31 (Erdil & Schneider-Joseph 2024), 3e30-1e32 (Sevilla et al. 2024); we take 1e29-1e32"),
}

# Fixed anchors from sources (point values)
FIXED = {
    "K0_usd": 4.9e8,      # [S] Grok 4 amortized training cost ~$490M (Epoch AI)
    "E0_J": 310e9 * 3600, # [S] Grok 4 training energy ~310 GWh (Epoch AI)
    "T0_days": 90.0,      # [A] consistent with 310 GWh at 100-150 MW (Epoch/EPRI)
    "I0_usd": 4.1e11,     # [S/press] 2025 capex of Microsoft, Alphabet, Amazon, Meta ~ $410B (proxy for AI compute investment)
    "Y0_usd": 1.15e14,    # [A] world nominal GDP ~ $115T (order of magnitude; IMF WEO database)
    "t0": 2025.5,
}

NAMES = list(PRIORS.keys())


def sample(n, rng):
    out = {}
    for k, (dist, lo, hi, _, _) in PRIORS.items():
        u = rng.random(n)
        out[k] = from_unit(k, u)
    return out


def from_unit(k, u):
    dist, lo, hi, _, _ = PRIORS[k]
    if dist == "uniform":
        return lo + (hi - lo) * u
    if dist == "loguniform":
        return np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u)
    raise ValueError(dist)
