"""Sourced inputs for the lag-elasticity model of AI chip export controls.

Every number here is either (S) taken from a cited public source (see paper.md
References for full citations) or (A) an explicit modelling assumption, which is
always varied in sensitivity analysis.  Units: time in years, compute in
H100-equivalents (H100e) unless noted.
"""
import numpy as np

LN2, LN10 = np.log(2.0), np.log(10.0)

# --- Frontier progress rates ---------------------------------------------------
# (S) Epoch AI: frontier LLM training compute grows ~5x/yr since 2020 (90% CI 4-6x).
COMPUTE_GROWTH_MEDIAN = 5.0
COMPUTE_GROWTH_CI90 = (4.0, 6.0)
# (S) Ho et al. (NeurIPS 2024): compute to reach fixed performance halves every
#     ~8 months (90% CI ~2-22 months).
ALGO_HALVING_MONTHS_MEDIAN = 8.0
ALGO_HALVING_MONTHS_CI90 = (2.0, 22.0)
# (A) narrower alternative prior used as a robustness check
ALGO_HALVING_MONTHS_CI90_NARROW = (5.0, 14.0)

# --- Observed US-China capability lag ------------------------------------------
# (S) Epoch AI ECI data insight: Chinese models trail US frontier by ~7 months on
#     average since 2023, range 4-14 months.  IFP (2026): 6.5 months.
LAG_OBS_MEDIAN_MONTHS = 7.0
LAG_OBS_CI90_MONTHS = (4.0, 14.0)

# --- Compute ratios ---------------------------------------------------------------
# (S) IFP (2026): frontier *training* compute ratio US:China = 16x at end-2026
#     (citing Pilz 2026, forthcoming); total AI compute ratio ~9x end-2026 -> ~13x
#     end-2027; US stock 41.5M H100e end-2026, 117M end-2027.
R_TRAIN_MEDIAN = 16.0
# (S) Epoch AI chip-imports data as reported in secondary coverage: US 72.2% and
#     China 9.6% of AI-chip compute arriving since 2022 (mid-2026) -> ratio ~7.5.
#     (S) Pilz et al. (2025): US ~75%, China ~15% of AI supercomputer performance.
US_SHARE_ARRIVALS, CN_SHARE_ARRIVALS = 0.722, 0.096
# (A) prior on the training ratio: median 16, 90% CI [7.5, 30]
R_TRAIN_CI90 = (7.5, 30.0)
R_STOCK_END2026, R_STOCK_END2027 = 9.0, 13.0
US_STOCK_END2026, US_STOCK_END2027 = 41.5e6, 117e6

# --- IFP multiplicative-recapture parameter --------------------------------------
# (S) IFP (2026): adversarial distillation recaptures share S = 0.5 of the lead.
S_IFP = 0.5
S_RANGE = (0.25, 0.75)  # (A)

# --- Counterfactual China share without controls ---------------------------------
# (S) NVIDIA: China was up to ~20-25% of data-centre revenue before Oct-2022
#     controls, falling to mid-single digits afterwards (company statements as
#     reported by Fortune / Yahoo Finance).
CN_SHARE_COUNTERFACTUAL = (0.15, 0.25)  # (A) range built around the 20-25% figure

# --- Intervention sizes (flows, H100e per year) -----------------------------------
# (S) IFP (2026): BIS cap of 75,000 H200 per Chinese customer; (S) reports of ~10
#     approved Chinese buyers (TechRepublic/Tom's Hardware, May 2026); >2M H200
#     reported orders (IFP 2026).  (S) H200 has the same dense compute as H100
#     (NVIDIA datasheet; SemiAnalysis) -> 1.0 H100e per H200.
H200_FLOW_CAP = 75_000 * 10
H200_FLOW_ORDERS = 2.0e6
# (S) Epoch AI: median 660k H100e smuggled to China through end-2025 (90%:
#     290k-1.6M).  (A) spread over ~2.5 years -> annual rate.
SMUGGLED_CUM_2025 = (290e3, 660e3, 1.6e6)
SMUGGLING_YEARS = 2.5

# --- Stock dynamics -----------------------------------------------------------------
# (A) accelerator useful life ~4 years -> depreciation delta = 0.25/yr; varied 0-0.4.
DELTA = 0.25
DELTA_RANGE = (0.0, 0.4)
