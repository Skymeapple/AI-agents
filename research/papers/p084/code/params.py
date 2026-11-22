"""Parameters for the AI-accelerator supply-disruption model.

Every value carries a tag:
  [S:<ref>]  taken from (or directly computed from) a cited public source (see paper, References)
  [A]        modelling assumption, calibrated loosely to cited anchor events; swept in sensitivity
  [P]        policy/scenario parameter that is deliberately NOT forecast; reported as a sweep
"""
import copy

# ---------------------------------------------------------------------------
# Inputs to one AI accelerator (Leontief / fixed proportions).
# AI share of each input's global capacity in 2025:
#   [S:Epoch2026] four largest AI chip designers consumed ~90% of CoWoS and HBM,
#   ~12% of advanced (3/5 nm) logic die production.
# ---------------------------------------------------------------------------
AI_SHARE = {"logic": 0.12, "cowos": 0.90, "hbm": 0.90}

# ---------------------------------------------------------------------------
# Sites: (name, input, region, capacity share within the input, tsmc_process_portable)
# Capacity shares:
#  logic: [S:SIA-BCG2021] 92% of <10nm capacity in Taiwan (8% Korea) in 2019;
#         [S:Epoch-components] TSMC holds "vast majority" of 3/5nm.  TSMC Arizona
#         (N4) share and non-TSMC (Samsung/Intel) share are [A].
#  cowos: [S:TrendForce-Jun2026] TSMC 120-140k wpm (Taiwan) + OSAT partners 50-60k wpm;
#         TSMC-linked US packaging not before ~2027-2029 [S:TomsHW2024, TrendForce-Jun2025].
#         Non-Taiwan share (5%) is [A] (some OSAT capacity outside Taiwan).
#  hbm:   [S:TrendForce-Aug2025] 2025 projected shares SK hynix 52.3%, Samsung 28.7%,
#         Micron 19%.  SK hynix & Samsung HBM in Korea; Micron's largest HBM site
#         (DRAM, test, HBM3E stacking) is Taichung, Taiwan [S:TomsHW-Micron].
#         Treating all Micron HBM as Taiwan-gated is [A].
# ---------------------------------------------------------------------------
def default_sites():
    s = []
    # TSMC Taiwan advanced logic, split into 6 equal fab clusters [A]
    for k in range(6):
        s.append(dict(name=f"logic_TW{k}", inp="logic", region="TW", share=0.90 / 6, portable=True, supplier="TSMC"))
    s.append(dict(name="logic_US", inp="logic", region="US", share=0.03, portable=True, supplier="TSMC"))
    for k in range(2):
        s.append(dict(name=f"logic_KRUS{k}", inp="logic", region="KR", share=0.07 / 2, portable=False, supplier="other"))
    for k in range(6):
        s.append(dict(name=f"cowos_TW{k}", inp="cowos", region="TW", share=0.95 / 6, portable=True, supplier="TSMC/OSAT"))
    s.append(dict(name="cowos_out", inp="cowos", region="OTHER", share=0.05, portable=True, supplier="OSAT"))
    for k in range(2):
        s.append(dict(name=f"hbm_SKH{k}", inp="hbm", region="KR", share=0.523 / 2, portable=True, supplier="SKH"))
    s.append(dict(name="hbm_SS", inp="hbm", region="KR", share=0.287, portable=True, supplier="Samsung"))
    s.append(dict(name="hbm_MU", inp="hbm", region="TW", share=0.19, portable=True, supplier="Micron"))
    return s


# ---------------------------------------------------------------------------
# Hazards.  Durations are lognormal(median, sigma) in WEEKS.  Loss = fraction of
# the affected site's capacity removed for the duration (rectangular profile).
# ---------------------------------------------------------------------------
HAZARDS = {
    # Site-level incident (fire, contamination, utility failure).  [A]
    # Anchors: Renesas Naka fire 2021 -> ~97 days to 100% capacity [S:Renesas2021];
    # 2011 Thai floods: HDD output ~80% of pre-flood by Feb 2012 [S:Register2012];
    # Toyota 42 d, Honda 174 d (Thai floods) [S:SimchiLevi2014].
    "local": dict(rate_per_site=0.05, loss=(0.5, 1.0), dur_median=10.0, dur_sigma=0.7),
    # Major Taiwan earthquake affecting all Taiwan sites.  Rate: two fab-relevant
    # M>=7.4 events (1999 Chi-Chi, 2024 Hualien) in ~27 years -> ~0.075/yr [A from S].
    # 2024: >70% tool recovery in 10 h, >80% next day [S:Fortune2024];
    # 1999: 6-11 day production outage, mostly power [S:UMC1999,TSMC1999].
    "tw_quake": dict(rate=0.075, loss=(0.3, 0.9), dur_median=1.0, dur_sigma=0.6),
    # Korea regional shock (quake / grid / conflict) affecting all Korean sites. [A]
    "kr_regional": dict(rate=0.01, loss=(0.3, 1.0), dur_median=4.0, dur_sigma=1.0),
    # HBM supplier yield / qualification shock (per supplier).  Anchor: Samsung's
    # 12-Hi HBM3E qualification difficulties, share 41%->28.7% [S:TrendForce-Aug2025]. [A]
    "hbm_yield": dict(rate_per_supplier=0.15, loss=(0.2, 0.6), dur_median=13.0, dur_sigma=0.5),
    # Taiwan geopolitical cut-off (blockade/quarantine): all Taiwan output unavailable. [P]
    # Annual probability is a scenario parameter, not a forecast.
    "tw_geo": dict(rate=0.01, loss=(1.0, 1.0), dur_median=26.0, dur_sigma=0.8),
}

MODEL = dict(
    weeks=52,
    allocation="priority",   # "priority": AI outbids other users; "proportional": pro-rata rationing
    headroom_prop=0.05,      # [A] surge headroom under proportional rationing
    buffer_weeks=4.0,        # [A] finished/in-transit accelerator inventory, weeks of demand
    port_lag_weeks=52.0,     # [A] redesign+requalification lag to move a TSMC design to another foundry
)


def config(**over):
    """Return a deep-copied config dict with optional overrides."""
    cfg = dict(ai_share=copy.deepcopy(AI_SHARE), sites=default_sites(),
               hazards=copy.deepcopy(HAZARDS), model=copy.deepcopy(MODEL))
    for k, v in over.items():
        cfg[k] = v
    return cfg
