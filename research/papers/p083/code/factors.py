"""Emission and water-intensity factors (all sourced; see paper Sec. 3.3).

CARBON_LCA (gCO2e/kWh generated): exactly the factor set used by the gridemissions
  package (de Chalendar et al. 2019, PNAS; repository src/gridemissions/emissions.py),
  so that our production-based intensity is consistent with the consumption-based
  co2i series we also use.
EWIF (L/kWh generated, operational *consumption*): medians for recirculating
  (cooling-tower) technologies from Macknick et al. (2011, NREL/TP-6A20-50900,
  Table 2; republished in ERL 2012). gal/MWh * 3.785e-3 = L/kWh.
  coal tower generic 687; NGCC tower 198; nuclear tower 672; gas/oil steam tower 826.
  LOW = once-through medians (coal 250, NGCC 100, nuclear 269, steam 240);
  HIGH = tower maxima (coal 1100, NGCC 300, nuclear 845, steam 1170).
  Hydropower: excluded in the central case (as in Macknick et al. 2012 and Spang et
  al. 2014); sensitivity uses 17 L/kWh (NREL-derived value tabulated in
  arXiv:2512.08725) and 68 L/kWh (Torcellini et al. 2003, end-use basis).
  OTH/UNK: assumption, 2.0 L/kWh central, swept over [0, 3.8].
"""
G2L = 3.785411784e-3

CARBON_LCA = {"WAT": 4, "NUC": 16, "SUN": 46, "GAS": 469, "WND": 12, "COL": 1000,
              "OIL": 840, "OTH": 439, "UNK": 439}

EWIF_CENTRAL = {"COL": 687 * G2L, "GAS": 198 * G2L, "NUC": 672 * G2L, "OIL": 826 * G2L,
                "OTH": 2.0, "UNK": 2.0, "SUN": 0.0, "WND": 0.0, "WAT": 0.0}
EWIF_LOW = {"COL": 250 * G2L, "GAS": 100 * G2L, "NUC": 269 * G2L, "OIL": 240 * G2L,
            "OTH": 0.0, "UNK": 0.0, "SUN": 0.0, "WND": 0.0, "WAT": 0.0}
EWIF_HIGH = {"COL": 1100 * G2L, "GAS": 300 * G2L, "NUC": 845 * G2L, "OIL": 1170 * G2L,
             "OTH": 3.8, "UNK": 3.8, "SUN": 0.0, "WND": 0.0, "WAT": 0.0}
HYDRO_ALT = [0.0, 17.0, 68.0]
