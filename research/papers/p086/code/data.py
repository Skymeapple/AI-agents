"""Sourced inputs.  Every number here is either (S) taken from a cited public
source or (A) an explicit modelling assumption.  See paper Section 4.1.
"""
# (S) Cash purchases of property & equipment, $bn, fiscal years (MSFT FY ends 30 June).
# Source: Capexwave, "How big is the AI capex wave?", compiled from each firm's 10-K
# (Microsoft 10-K filed 29 Jul 2026; Alphabet 5 Feb 2026; Amazon 6 Feb 2026; Meta 29 Jan 2026).
CAPEX = {
    'MSFT': {2021: 20.6, 2022: 23.9, 2023: 28.1, 2024: 44.5, 2025: 64.6},
    'GOOGL': {2021: 24.6, 2022: 31.5, 2023: 32.3, 2024: 52.5, 2025: 91.4},
    'AMZN': {2021: 61.1, 2022: 63.6, 2023: 52.7, 2024: 83.0, 2025: 131.8},
    'META': {2021: 18.7, 2022: 31.2, 2023: 27.0, 2024: 37.3, 2025: 69.7},
}
# (S) 2026 company guidance as of late-Oct 2026 earnings (Capexwave quoting each firm):
# Amazon ~220 (cash capex); Alphabet 195-205; Meta 135-145 (incl. finance-lease principal);
# Microsoft ~170 for calendar 2026 (incl. finance leases).  Midpoints used; bases differ.
GUIDE_2026 = {'MSFT': 170.0, 'GOOGL': 200.0, 'AMZN': 220.0, 'META': 140.0}

# (S) Q3-2026 quarterly revenue, $bn (CRN / StockTitan, from company filings):
CLOUD_Q3_2026 = {'AWS': 46.0, 'Azure': 33.2, 'GoogleCloud': 30.7}

# (S) Meta disclosed: extending most servers & network assets to 5.5 years (from 5)
# effective 1 Jan 2025 reduced 2025 depreciation by $2.92bn (based on assets in
# service at 31 Dec 2024).  Meta 2025 10-K, via Hudson Labs.
META_LIFE_CHANGE_EFFECT = 2.92

# (S) H100 rental price observations used for an implied-obsolescence check
#   ~$8/GPU-hr at 2023-24 peak (latent.space via Apex36; Silicon Data reports $7-10)
#   SemiAnalysis 1-yr contract index: $1.70 (Oct 2025) -> $2.35 (Mar 2026)
#   Silicon Data H100 index: $3.06 (1 Sep 2024) -> $2.36 (19 Jun 2025)
H100_OBS = dict(peak_2023=8.0, sa_oct2025=1.70, sa_mar2026=2.35, sd_sep2024=3.06, sd_jun2025=2.36)

# (S) Epoch AI: FLOP/s per $ of ML GPUs doubles every 2.07 yr (95% CI 1.54-3.13);
#     hardware at fixed performance ~30% cheaper each year.  H100 incl. server ~$44k.
EPOCH_DOUBLING_ML = (2.07, 1.54, 3.13)
EPOCH_ANNUAL_DECLINE = 0.30
H100_ALLIN_COST = 44000.0

# (A) Base-case modelling assumptions and Monte-Carlo ranges
BASE = dict(g=0.35, c=0.05, r=0.09, Tmax=8.0, L=5.75, s=0.60, L_struct=15.0, c_struct=0.02)
RANGES = dict(g=(0.20, 0.60), c=(0.02, 0.10), r=(0.07, 0.11), Tmax=(6.0, 9.0),
              L=(5.0, 6.0), s=(0.50, 0.70))
# (A) Back-cast of pre-2021 vintages (only affects early-year accounting stocks)
BACKCAST_GROWTH = 0.15
# (A) Post-2026 capex scenarios: annual growth rates for 2027, 2028, 2029, 2030
SCENARIOS = {
    'boom':        [0.30, 0.15, 0.10, 0.05],
    'plateau':     [0.00, 0.00, 0.00, 0.00],
    'retrenchment': [-0.25, -0.10, -0.05, 0.00],
}
