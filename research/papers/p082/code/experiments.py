"""Run experiment sweeps E1-E6 in parallel; results saved as JSON lines under results/.
Usage: python3 code/experiments.py E1 [E2 ...]"""
import sys, os, json, itertools, time, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model
DATA = os.path.join(HERE, "..", "data"); RES = os.path.join(HERE, "..", "results")
SITES = ["abilene_tx", "loudoun_va", "phoenix_az"]
ALL = ("pv", "wind", "bat", "ct", "cc", "firm"); NOFIRM = ("pv", "wind", "bat", "ct", "cc")
DT = 2

def cf(site, yr=2012):
    d = np.load(os.path.join(DATA, f"cf_{site}_{yr}.npz")); return d["solar"], d["wind"]

def run(job):
    job = dict(job); t0 = time.time()
    cs, cw = cf(job["site"], job.get("year", 2012))
    P = model.params(**job.get("pkw", {}))
    o = model.solve(cs, cw, P, phi=job["phi"], v=job["v"], allow=tuple(job["allow"]), dt=job.get("dt", DT),
                    fixed_caps=job.get("fixed_caps"), max_curtail_frac=job.get("max_curtail_frac"))
    o.pop("series", None)
    job.update({k: (float(val) if isinstance(val, (np.floating, float)) else val) for k, val in o.items()})
    job["secs"] = time.time() - t0
    return job

def jobs_E1():
    for s, (an, al), phi, v in itertools.product(SITES, [("all", ALL), ("nofirm", NOFIRM)],
                                                 [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0], [250, 500, 1000, 2000, 10000]):
        yield dict(exp="E1", site=s, mix=an, allow=al, phi=phi, v=v)

def jobs_E2():
    for s, phi, v, fc in itertools.product(["abilene_tx", "loudoun_va"], [0.9, 1.0], [500, 2000],
                                           [5000, 7500, 9650, 12500, 15000, 20000]):
        yield dict(exp="E2", site=s, mix="all", allow=ALL, phi=phi, v=v, firm_capex=fc, pkw={"firm_capex": fc})

def jobs_E3_design():
    for yr, phi in itertools.product([2010, 2011, 2012, 2013], [0.95, 1.0]):
        yield dict(exp="E3d", site="abilene_tx", mix="nofirm", allow=NOFIRM, phi=phi, v=1000, year=yr)

def jobs_E4():
    for s, phi, v, (an, al) in [("abilene_tx", 0.9, 1000, ("all", ALL)), ("abilene_tx", 1.0, 500, ("nofirm", NOFIRM)),
                                ("loudoun_va", 0.95, 1000, ("nofirm", NOFIRM)), ("phoenix_az", 0.8, 2000, ("nofirm", NOFIRM)),
                                ("loudoun_va", 1.0, 250, ("all", ALL)), ("phoenix_az", 0.99, 1000, ("all", ALL))]:
        yield dict(exp="E4", site=s, mix=an, allow=al, phi=phi, v=v, dt=1)

def jobs_E5():
    base = dict(exp="E5", site="abilene_tx", mix="all", allow=ALL, phi=0.9, v=1000)
    yield dict(base, sens="base", pkw={})
    for name, kw in [("wacc_5", {"wacc": 0.05}), ("wacc_10", {"wacc": 0.10}), ("gas_2", {"gas_price": 2.0}),
                     ("gas_6", {"gas_price": 6.0}), ("firm_capex_6000", {"firm_capex": 6000.0}), ("firm_capex_15000", {"firm_capex": 15000.0})]:
        yield dict(base, sens=name, pkw=kw)
    for name, ov in [("pv_-30%", ("pv", 0.7)), ("pv_+30%", ("pv", 1.3)), ("bat_-30%", ("bat", 0.7)), ("bat_+30%", ("bat", 1.3)),
                     ("wind_-30%", ("wind", 0.7)), ("wind_+30%", ("wind", 1.3))]:
        yield dict(base, sens=name, scale=ov)

def jobs_E6():
    for s, (an, al), f in itertools.product(["abilene_tx", "loudoun_va"], [("all", ALL), ("nofirm", NOFIRM)],
                                            [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]):
        yield dict(exp="E6", site=s, mix=an, allow=al, phi=1.0, v=0.0, max_curtail_frac=f)

def run_scaled(job):
    if "scale" in job:
        k, m = job["scale"]; P = model.params(**job.get("pkw", {}))
        ov = {"bat_p": P["bat_p"] * m, "bat_e": P["bat_e"] * m} if k == "bat" else {k: P[k] * m}
        job = dict(job, pkw={"overrides": ov})
    return run(job)

if __name__ == "__main__":
    exps = sys.argv[1:]
    for e in exps:
        jobs = list(globals()["jobs_" + e]())
        out = os.path.join(RES, f"{e}.jsonl")
        t0 = time.time()
        with Pool(4) as p, open(out, "w") as fh:
            for r in p.imap_unordered(run_scaled, jobs):
                r.pop("pkw", None); fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
        print(e, len(jobs), "jobs in %.0fs" % (time.time() - t0), flush=True)
