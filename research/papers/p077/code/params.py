"""Sourced parameters for the SQC accountancy model.
Every value carries its source; 'ASSUMPTION' marks modelling choices that are not sourced.
"""
import numpy as np

# --- Reference capability (fixed-capability threshold) ---------------------
F0 = 2.1e25          # FLOP, GPT-4 training compute estimate (Epoch AI, "models over 1e25 FLOP" data insight)
T0 = 2023.0          # ASSUMPTION: date at which F0 is the frontier / reference point

# --- Algorithmic progress ---------------------------------------------------
H_ALG_MED = 8.0/12   # years; compute to reach fixed LM performance halves ~every 8 months (Ho et al. 2024, NeurIPS)
H_ALG_LO, H_ALG_HI = 5.0/12, 14.0/12   # interval reported in arXiv v1 abstract of Ho et al. (2403.05812)

# --- Frontier training compute growth --------------------------------------
G_F_MED = 4.0        # x/yr, frontier models since 2018 (Sevilla & Roldan 2024, Epoch AI)
G_F_LO, G_F_HI = 3.0, 5.3   # 5.3x/yr = 2010-2024 frontier estimate (same source); 3.0 = ASSUMPTION slowdown scenario

# --- Hardware ----------------------------------------------------------------
R_H100 = 989e12      # FLOP/s, H100 SXM BF16 dense (NVIDIA datasheet: 1,979 with sparsity, half without)
U_MED = 0.40         # model FLOPs utilisation; PaLM reported 46.2% (Chowdhery et al. 2022)
U_LO, U_HI = 0.30, 0.50   # ASSUMPTION range

# --- Global AI-chip stock (H100-equivalents) --------------------------------
S_REF = 20.3e6       # H100e delivered by end of 2025 (Epoch AI data-centre research)
T_S = 2026.0
G_S_MED = 2.3        # x/yr, NVIDIA stock of computing power since 2019 (Epoch AI data insight, Feb 2025)
G_S_LO, G_S_HI = 2.0, 3.3   # upper: all-designer AI computing capacity ~3.3x/yr since 2022 (Epoch AI, "Global AI computing capacity is doubling every 7 months", Jan 2026); lower: ASSUMPTION

# --- Empirical inventory uncertainty anchor -----------------------------------
SMUG_MED, SMUG_LO, SMUG_HI = 0.66e6, 0.29e6, 1.6e6   # H100e smuggled to China through 2025, 90% CI (Epoch AI)
# implied 1-sigma of an approximately normal inventory error (derived, not sourced):
SIGMA_EMP_ABS = (SMUG_HI - SMUG_LO) / (2*1.645)
SIGMA_EMP_REL = SIGMA_EMP_ABS / S_REF

# --- IAEA reference values ----------------------------------------------------
IAEA_INFIELD = 3136          # in-field verifications 2023 (IAEA Safeguards Implementation Report 2023)
IAEA_DAYS = 14302            # days in field 2023 (same)
IAEA_FACILITIES = 1300       # ">1,300 facilities and LOFs" 2023 (IAEA Safeguards Statement 2023)
IAEA_RB = 427.7e6            # EUR/yr regular budget request 2024-25 (VCDNP budget primer)
IAEA_SG_SHARE = 0.39         # safeguards share of regular budget (VCDNP)
TIMELINESS_DIRECT = 1/12     # yr, timeliness goal for unirradiated direct-use material (IAEA Safeguards Glossary)

# --- Detection goal (ASSUMPTION following standard accountancy practice) ------
ALPHA = 0.05                 # false-alarm probability per balance test
PD_GOAL = 0.90               # detection probability goal

TAU_BASE = 90/365            # yr, ASSUMPTION: window in which a covert run must complete
