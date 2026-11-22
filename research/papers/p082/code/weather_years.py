"""E3: weather-year robustness. Design an islanded no-firm (PV+wind+battery+gas) system on one weather year
(2010-2013) at CFE target phi and VoLC v=1000 $/MWh, then fix capacities and re-dispatch on every other year,
still enforcing phi; report realized curtailment and cost per served MWh."""
import os, sys, json, itertools, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model
from experiments import cf, NOFIRM, ALL, RES
YEARS = [2010, 2011, 2012, 2013]

def design(args):
    site, mix, phi, yr = args
    al = NOFIRM if mix == "nofirm" else ALL
    cs, cw = cf(site, yr); o = model.solve(cs, cw, model.params(), phi=phi, v=1000, allow=al, dt=2)
    return dict(site=site, mix=mix, phi=phi, design_year=yr, caps=o["caps"], lcoe=o["lcoe_served"], curtail=o["curtail_frac"])

def evaluate(args):
    d, yr = args
    cs, cw = cf(d["site"], yr)
    caps = [d["caps"][k] for k in ["pv", "wind", "bat_p", "bat_e", "ct", "cc", "firm"]]
    o = model.solve(cs, cw, model.params(), phi=d["phi"], v=1000, dt=2, fixed_caps=caps)
    return dict(site=d["site"], mix=d["mix"], phi=d["phi"], design_year=d["design_year"], eval_year=yr,
                lcoe=o["lcoe_served"], curtail=o["curtail_frac"], total_per_mwh_load=o["total_cost"] / 8760)

if __name__ == "__main__":
    cfgs = list(itertools.product(["abilene_tx", "loudoun_va"], ["nofirm"], [0.95, 1.0], YEARS))
    with Pool(4) as p:
        ds = p.map(design, cfgs)
        evs = p.map(evaluate, [(d, y) for d in ds for y in YEARS if y != d["design_year"]])
    with open(os.path.join(RES, "E3.json"), "w") as f: json.dump(dict(designs=ds, evals=evs), f, indent=1)
    print("E3 done", len(ds), len(evs))
