"""Main analysis. Offline: reads only the committed tables in data/.

Outputs (results/):
  summary.json            every number quoted in the paper (measurement part)
  reach_weekly.csv        weekly revision-reach series for npm servers
  cohort_weekly.csv       weekly entrants / born-modern / migrations (input to 09_model.py)
  fig1_reach.png, fig2_migration.png, fig3_pypi.png
"""
import bisect, collections, csv, datetime as dt, gzip, json, math, os, re

import nodesemver
import numpy as np
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

B = os.path.join(os.path.dirname(__file__), "..")
D, R = os.path.join(B, "data"), os.path.join(B, "results")
os.makedirs(R, exist_ok=True)
REVS = ["2025-03-26", "2025-06-18", "2025-11-25", "2026-07-28"]
MODERN = "2026-07-28"
S = {}  # summary


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def day(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def share(k, n):
    return {"k": int(k), "n": int(n), "p": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


# ---------------------------------------------------------------- SDK maps
ts_sdk = json.load(open(os.path.join(D, "sdk_versions_ts.json")))
ts_sdk = [r for r in ts_sdk if r["supported"]]
SDK1 = sorted(((ts(r["date"]), r["sdk"], set(r["supported"])) for r in ts_sdk), key=lambda x: x[0])
SDK1_SUP = {v: sup for _, v, sup in SDK1}
v2map = json.load(open(os.path.join(D, "sdk_v2_map.json")))
SRV2 = sorted(((ts(r["date"]), r["v"], set(r["modern"] or [])) for r in v2map
               if r["pkg"] == "@modelcontextprotocol/server"), key=lambda x: x[0])
LEGACY_ALL = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}  # v2 SDKs keep the handshake revisions
SRV2_SUP = {v: (sup | LEGACY_ALL) for _, v, sup in SRV2}
GA_TS = ts([r["date"] for r in v2map if r["pkg"] == "@modelcontextprotocol/server" and r["v"] == "2.0.0"][0])
py_sdk = json.load(open(os.path.join(D, "sdk_versions_py.json")))

first_support = {}
for r in REVS:
    a = [(t, v) for t, v, sup in SDK1 if r in sup] + [(t, v) for t, v, sup in SRV2 if r in sup]
    stable = [(t, v) for t, v in a if "-" not in v]
    first_support[r] = {"ts_first_any": day(min(a)[0]), "ts_first_any_v": min(a)[1],
                        "ts_first_stable": day(min(stable)[0]), "ts_first_stable_v": min(stable)[1]}
S["sdk_first_support_ts"] = first_support
S["sdk1_latest"] = {"version": SDK1[-1][1], "date": day(SDK1[-1][0]), "supports": sorted(SDK1[-1][2])}

# ---------------------------------------------------------------- semver resolution (memoised)
_memo = {}


def resolve(kind, rng, n):
    """Max / min SDK version satisfying `rng` among the first n releases (by date) of line `kind`."""
    key = (kind, rng, n)
    if key in _memo:
        return _memo[key]
    line = SDK1 if kind == "sdk" else SRV2
    vs = [v for _, v, _ in line[:n]]
    stable = [v for v in vs if "-" not in v]
    r = rng.strip()
    try:
        if r in ("latest", "*", "", "x"):
            mx = nodesemver.max_satisfying(stable, "*", loose=True) if stable else None
            mn = nodesemver.min_satisfying(stable, "*", loose=True) if stable else None
        elif r in ("next", "beta", "alpha", "canary"):
            mx = vs[-1] if vs else None
            mn = mx
        else:
            mx = nodesemver.max_satisfying(vs, r, loose=True)
            mn = nodesemver.min_satisfying(vs, r, loose=True)
        out = (str(mx) if mx else None, str(mn) if mn else None)
    except Exception:
        out = ("ERR", "ERR")
    _memo[key] = out
    return out


def n_avail(line, t):
    return bisect.bisect_right([x[0] for x in line], t)


SDK1_T = [x[0] for x in SDK1]
SRV2_T = [x[0] for x in SRV2]


def classify(deps):
    """Dependency class of one published package version."""
    rt = {k.split(":", 1)[1]: v for k, v in deps.items() if k[:3] in ("dep", "pee")}
    dv = {k.split(":", 1)[1]: v for k, v in deps.items() if k[:3] == "dev"}
    if "@modelcontextprotocol/server" in rt or "@modelcontextprotocol/core" in rt:
        return "v2", rt.get("@modelcontextprotocol/server", "*"), rt.get("@modelcontextprotocol/sdk")
    if "@modelcontextprotocol/sdk" in rt:
        return "sdk1", rt["@modelcontextprotocol/sdk"], None
    if "@modelcontextprotocol/server" in dv:
        return "dev_v2", dv["@modelcontextprotocol/server"], None
    if "@modelcontextprotocol/sdk" in dv:
        return "dev_sdk1", dv["@modelcontextprotocol/sdk"], None
    if rt or dv:
        return "framework", None, None
    return "opaque", None, None


def supports(cls, rng, t_now, t_pub):
    """(float_set, floor_set): revisions the package can speak under a fresh install at t_now,
    and under the oldest SDK its declared range admits.  Bundled (dev-only) SDKs are frozen at
    the SDK resolved when the package version was published."""
    if cls in ("sdk1", "dev_sdk1"):
        t = t_pub if cls.startswith("dev") else t_now
        mx, mn = resolve("sdk", rng, bisect.bisect_right(SDK1_T, t))
        if mx == "ERR" or mx is None:
            return None, None
        if cls.startswith("dev"):
            return SDK1_SUP[mx], SDK1_SUP[mx]
        return SDK1_SUP[mx], SDK1_SUP[mn]
    if cls in ("v2", "dev_v2"):
        t = t_pub if cls.startswith("dev") else t_now
        mx, mn = resolve("srv", rng, bisect.bisect_right(SRV2_T, t))
        if mx == "ERR" or mx is None:
            return None, None
        if cls.startswith("dev"):
            return SRV2_SUP[mx], SRV2_SUP[mx]
        return SRV2_SUP[mx], SRV2_SUP[mn]
    return None, None


# ---------------------------------------------------------------- load registry + npm
reg = [json.loads(l) for l in gzip.open(os.path.join(D, "registry_slim.jsonl.gz"), "rt")]
S["registry"] = {"servers": len(reg), "first_published": min(x["pub"] for x in reg)[:10],
                 "last_published": max(x["pub"] for x in reg)[:10],
                 "status": dict(collections.Counter(x["status"] for x in reg))}
comp = collections.Counter()
for x in reg:
    kinds = {p[0] for p in x["pkgs"]}
    comp[("remote" if x["remotes"] else "") + ("+pkg" if kinds else "")] += 1
S["registry"]["composition"] = {"remote_only": comp["remote"], "package_only": comp["+pkg"],
                                "both": comp["remote+pkg"], "neither": comp[""]}
S["registry"]["with_npm"] = sum(any(p[0] == "npm" for p in x["pkgs"]) for x in reg)
S["registry"]["with_pypi"] = sum(any(p[0] == "pypi" for p in x["pkgs"]) for x in reg)
bymonth = collections.Counter(x["pub"][:7] for x in reg)
S["registry"]["published_by_month"] = dict(sorted(bymonth.items()))
ns = collections.Counter(x["name"].split("/")[0] for x in reg)
S["registry"]["namespaces"] = len(ns)
S["registry"]["top10_namespace_share"] = round(sum(v for _, v in ns.most_common(10)) / len(reg), 4)

# deprecated HTTP+SSE transport among remotes, and server.json schema versions, by publish month
sse = collections.defaultdict(lambda: [0, 0])
sch = collections.defaultdict(collections.Counter)
for x in reg:
    m = x["pub"][:7]
    sch[m][x["schema"]] += 1
    if x["remotes"]:
        sse[m][1] += 1
        sse[m][0] += ("sse" in x["remotes"] and "streamable-http" not in x["remotes"])
tot = [sum(v[0] for v in sse.values()), sum(v[1] for v in sse.values())]
S["remote_transport"] = {"sse_only_overall": share(*tot),
                         "sse_only_by_month": {m: share(*sse[m]) for m in sorted(sse)}}
schema_dates = sorted({k for c in sch.values() for k in c if k and re.match(r"\d{4}-\d\d-\d\d", k)})
S["server_json_schema"] = {"versions": schema_dates,
                           "latest_share_by_month": {m: share(sch[m][schema_dates[-1]], sum(sch[m].values()))
                                                     for m in sorted(sch)}}

npm_ns = {}
for x in reg:
    for p in x["pkgs"]:
        if p[0] == "npm":
            npm_ns.setdefault(p[1], x["name"].split("/")[0])

pk = []
errs = 0
for line in gzip.open(os.path.join(D, "npm_deps.jsonl.gz"), "rt"):
    r = json.loads(line)
    if "error" in r or not r.get("versions"):
        errs += 1
        continue
    vers = [(ts(m["t"]), v, m["deps"]) for v, m in r["versions"].items() if m.get("t")]
    stable = [x for x in vers if "-" not in x[1]] or vers
    stable.sort()
    if not stable:
        errs += 1
        continue
    pk.append({"name": r["name"], "v": [(t, v) + classify(d) for t, v, d in stable]})
T = max(p["v"][-1][0] for p in pk)
S["npm"] = {"packages_listed": len(pk) + errs, "packages_with_history": len(pk), "unresolvable_404": errs,
            "versions_total": sum(len(p["v"]) for p in pk), "snapshot": day(T), "v2_ga": day(GA_TS)}


def state(p, t):
    i = bisect.bisect_right([x[0] for x in p["v"]], t) - 1
    return p["v"][i] if i >= 0 else None


# ---------------------------------------------------------------- snapshot composition
snap = collections.Counter()
rngtype = collections.Counter()
for p in pk:
    t_pub, v, cls, rng, also = state(p, T)
    snap[cls] += 1
    if cls == "sdk1":
        r = rng.strip()
        rngtype["caret" if r.startswith("^") else "exact" if re.match(r"^\d+\.\d+\.\d+$", r) else
                "tilde" if r.startswith("~") else "tag_or_star" if r in ("latest", "*") else "other"] += 1
n = len(pk)
S["npm"]["class_at_snapshot"] = {k: share(v, n) for k, v in snap.most_common()}
S["npm"]["sdk1_range_types"] = {k: share(v, snap["sdk1"]) for k, v in rngtype.most_common()}

# ---------------------------------------------------------------- weekly reach curves
WEEK = 7 * 86400
grid = []
t = T
while t > ts("2025-03-01T00:00:00Z"):
    grid.append(t)
    t -= WEEK
grid.reverse()
rows = []
DECL = ("sdk1", "v2", "dev_sdk1", "dev_v2")
for t in grid:
    alive = decl = 0
    fl = collections.Counter()
    fo = collections.Counter()
    for p in pk:
        st = state(p, t)
        if st is None:
            continue
        alive += 1
        if st[2] not in DECL:
            continue
        a, b = supports(st[2], st[3], t, st[0])
        if a is None:
            continue
        decl += 1
        for r in REVS:
            fl[r] += r in a
            fo[r] += r in b
    row = {"date": day(t), "alive": alive, "declared": decl}
    for r in REVS:
        row["float_" + r] = round(fl[r] / decl, 4) if decl else ""
        row["floor_" + r] = round(fo[r] / decl, 4) if decl else ""
    rows.append(row)
with open(os.path.join(R, "reach_weekly.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
S["reach_at_snapshot"] = {r: {"float": rows[-1]["float_" + r], "floor": rows[-1]["floor_" + r]} for r in REVS}
S["reach_at_snapshot"]["declared_n"] = rows[-1]["declared"]


def exact_reach(r, t, mode):
    k = d = 0
    for p in pk:
        st = state(p, t)
        if st is None or st[2] not in DECL:
            continue
        a, b = supports(st[2], st[3], t, st[0])
        if a is None:
            continue
        d += 1
        k += r in (a if mode == "float" else b)
    return k, d


# reach at fixed ages after the first *stable* SDK release supporting each revision
ages = [1, 7, 30, 60, 90, 118]
lag = {}
for r in REVS:
    t0 = ts(first_support[r]["ts_first_stable"] + "T23:59:59Z")
    lag[r] = {"spec_date": r, "sdk_stable": first_support[r]["ts_first_stable"],
              "spec_to_sdk_days": (dt.date.fromisoformat(first_support[r]["ts_first_stable"]) - dt.date.fromisoformat(r)).days}
    for a in ages:
        tt = t0 + a * 86400
        if tt > T + 86400:
            continue
        tt = min(tt, T)
        for mode in ("float", "floor"):
            k, d = exact_reach(r, tt, mode)
            lag[r]["%s_day%d" % (mode, a)] = share(k, d)
S["reach_by_age"] = lag
S["age_of_modern_at_snapshot_days"] = round((T - GA_TS) / 86400, 1)

# ---------------------------------------------------------------- migration across the major boundary
GA = GA_TS
cohort = []
for p in pk:
    st = state(p, GA)
    if st is None or st[2] != "sdk1":
        continue
    times = [x[0] for x in p["v"]]
    active = any(GA - 90 * 86400 <= x <= GA for x in times)
    post = [x for x in p["v"] if x[0] > GA]
    ev = next((x[0] for x in post if x[2] == "v2"), None)
    cohort.append({"name": p["name"], "active": active, "n_post": len(post), "event": ev,
                   "ns": npm_ns.get(p["name"])})
S["migration"] = {"cohort_n": len(cohort), "active_n": sum(c["active"] for c in cohort)}


def km(items, horizon):
    """Kaplan-Meier estimate of P(migrated by day d) with Greenwood 95% CI."""
    dur = np.array([((c["event"] or T) - GA) / 86400 for c in items])
    evt = np.array([c["event"] is not None for c in items])
    out = {}
    s, var = 1.0, 0.0
    for d in sorted(set(np.ceil(dur[evt]).astype(int))):
        at_risk = (dur >= d - 1 + 1e-9).sum() if d > 0 else len(dur)
        at_risk = (np.ceil(dur) >= d).sum()
        e = (evt & (np.ceil(dur).astype(int) == d)).sum()
        if at_risk == 0:
            continue
        s *= 1 - e / at_risk
        if at_risk > e:
            var += e / (at_risk * (at_risk - e))
        out[int(d)] = (s, var)
    res = {}
    for h in horizon:
        ks = [d for d in out if d <= h]
        if not ks:
            res[h] = {"F": 0.0, "ci95": [0.0, 0.0]}
            continue
        s, var = out[max(ks)]
        se = s * math.sqrt(var)
        res[h] = {"F": round(1 - s, 4), "ci95": [round(max(0, 1 - s - 1.96 * se), 4), round(min(1, 1 - s + 1.96 * se), 4)]}
    return res, out


H = [30, 60, 90, int((T - GA) / 86400)]
S["migration"]["km_all"], km_all = km(cohort, H)
S["migration"]["km_active"], km_act = km([c for c in cohort if c["active"]], H)
S["migration"]["km_dormant"], km_dor = km([c for c in cohort if not c["active"]], H)
mig = sum(c["event"] is not None for c in cohort)
rel = [c for c in cohort if c["n_post"] > 0]
S["migration"]["migrated"] = share(mig, len(cohort))
S["migration"]["released_after_ga"] = share(len(rel), len(cohort))
S["migration"]["migrated_given_release"] = share(sum(c["event"] is not None for c in rel), len(rel))
S["migration"]["active_released_after_ga"] = share(sum(c["n_post"] > 0 for c in cohort if c["active"]),
                                                   sum(c["active"] for c in cohort))
S["migration"]["dormant_released_after_ga"] = share(sum(c["n_post"] > 0 for c in cohort if not c["active"]),
                                                    sum(not c["active"] for c in cohort))
# constant-hazard (exponential) fit on the whole cohort: events / exposure
expo = sum(((c["event"] or T) - GA) / 86400 for c in cohort)
S["migration"]["hazard_per_day"] = mig / expo
S["migration"]["hazard_per_year"] = round(mig / expo * 365.25, 4)
expo_a = sum(((c["event"] or T) - GA) / 86400 for c in cohort if c["active"])
S["migration"]["hazard_per_year_active"] = round(sum(c["event"] is not None for c in cohort if c["active"]) / expo_a * 365.25, 4)
# split-half hazard (is it accelerating?)
half = GA + (T - GA) / 2
e1 = sum(1 for c in cohort if c["event"] and c["event"] <= half)
x1 = sum((min(c["event"] or T, half) - GA) / 86400 for c in cohort)
e2 = mig - e1
x2 = sum(max(0, ((c["event"] or T) - half)) / 86400 for c in cohort)
S["migration"]["hazard_per_year_first_half"] = round(e1 / x1 * 365.25, 4)
S["migration"]["hazard_per_year_second_half"] = round(e2 / x2 * 365.25, 4)

# comparison: the previous, non-breaking revision (2025-11-25).  Same construction: cohort = sdk1
# packages alive when the first stable SDK supporting it shipped; "event" = first release whose
# declared floor already implies the revision (an explicit range bump), observed for the same span.
t0 = ts(first_support["2025-11-25"]["ts_first_stable"] + "T00:00:00Z")
span = T - GA
c2 = []
for p in pk:
    st = state(p, t0)
    if st is None or st[2] != "sdk1":
        continue
    ev = None
    for x in p["v"]:
        if t0 < x[0] <= t0 + span and x[2] in ("sdk1", "v2"):
            a, b = supports(x[2], x[3], x[0], x[0])
            if b and "2025-11-25" in b:
                ev = x[0]
                break
    c2.append(ev)
kf, dn = exact_reach("2025-11-25", t0 + span, "float")
S["migration"]["prev_revision_comparison"] = {
    "cohort_n": len(c2), "span_days": round(span / 86400, 1),
    "explicit_floor_bump": share(sum(e is not None for e in c2), len(c2)),
    "float_reach_same_age": share(kf, dn)}

# ---------------------------------------------------------------- entrants: born modern?
ent = collections.defaultdict(lambda: [0, 0, 0])  # month -> [declared entrants, born v2, v2 at snapshot]
weekly = collections.defaultdict(lambda: {"entrants": 0, "born_modern": 0, "migrations": 0, "legacy_stock": 0})
for p in pk:
    t0p, v0, cls0 = p["v"][0][0], p["v"][0][1], p["v"][0][2]
    if cls0 not in ("sdk1", "v2"):
        continue
    wk = int((t0p - GA) // WEEK)
    weekly[wk]["entrants"] += 1
    weekly[wk]["born_modern"] += cls0 == "v2"
    if t0p > GA:
        m = day(t0p)[:7]
        ent[m][0] += 1
        ent[m][1] += cls0 == "v2"
        ent[m][2] += state(p, T)[2] == "v2"
for c in cohort:
    if c["event"]:
        weekly[int((c["event"] - GA) // WEEK)]["migrations"] += 1
# migrations by post-GA entrants that were born legacy
late = 0
for p in pk:
    if p["v"][0][0] > GA and p["v"][0][2] == "sdk1":
        ev = next((x[0] for x in p["v"] if x[2] == "v2"), None)
        if ev:
            weekly[int((ev - GA) // WEEK)]["migrations"] += 1
            late += 1
# bulk entrants: first publications from a namespace that launches >= 5 npm servers in the same week
nsweek = collections.Counter()
for p in pk:
    if p["v"][0][2] in ("sdk1", "v2"):
        nsweek[(npm_ns.get(p["name"]), int((p["v"][0][0] - GA) // WEEK))] += 1
bulk_tot = [0, 0, 0, 0]
for p in pk:
    t0p, cls0 = p["v"][0][0], p["v"][0][2]
    if cls0 in ("sdk1", "v2"):
        wk_ = int((t0p - GA) // WEEK)
        if nsweek[(npm_ns.get(p["name"]), wk_)] >= 5:
            weekly[wk_]["bulk_entrants"] = weekly[wk_].get("bulk_entrants", 0) + 1
            weekly[wk_]["bulk_born_modern"] = weekly[wk_].get("bulk_born_modern", 0) + (cls0 == "v2")
            if t0p > GA:
                bulk_tot[0] += 1; bulk_tot[1] += cls0 == "v2"
        elif t0p > GA:
            bulk_tot[2] += 1; bulk_tot[3] += cls0 == "v2"
S["bulk_entrants_post_ga"] = {"bulk_born_modern": share(bulk_tot[1], bulk_tot[0]),
                              "non_bulk_born_modern": share(bulk_tot[3], bulk_tot[2])}
# namespace-level entrants: each publisher namespace counted once, at its first post-GA npm server
ns_first = {}
for p in pk:
    t0p, cls0 = p["v"][0][0], p["v"][0][2]
    if cls0 in ("sdk1", "v2"):
        k_ = npm_ns.get(p["name"])
        if k_ not in ns_first or t0p < ns_first[k_][0]:
            ns_first[k_] = (t0p, cls0)
ns_ent = collections.defaultdict(lambda: [0, 0])
for k_, (t0p, cls0) in ns_first.items():
    if t0p > GA:
        ns_ent[day(t0p)[:7]][0] += 1
        ns_ent[day(t0p)[:7]][1] += cls0 == "v2"
        weekly[int((t0p - GA) // WEEK)].setdefault("ns_entrants", 0)
        weekly[int((t0p - GA) // WEEK)]["ns_entrants"] += 1
        weekly[int((t0p - GA) // WEEK)]["ns_born_modern"] = weekly[int((t0p - GA) // WEEK)].get("ns_born_modern", 0) + (cls0 == "v2")
# migration hazard of post-GA entrants that were born on the v1 SDK
ev_l = ex_l = 0
for p in pk:
    if p["v"][0][0] > GA and p["v"][0][2] == "sdk1":
        ev = next((x[0] for x in p["v"] if x[2] == "v2"), None)
        ev_l += ev is not None
        ex_l += ((ev or T) - p["v"][0][0]) / 86400
stock_ga = collections.Counter(state(p, GA)[2] for p in pk if state(p, GA))
S["entrants"] = {"stock_at_ga": dict(stock_ga),
                 "by_month_namespace_first": {m: share(v[1], v[0]) for m, v in sorted(ns_ent.items())},
                 "namespace_first_total": share(sum(v[1] for v in ns_ent.values()), sum(v[0] for v in ns_ent.values())),
                 "born_legacy_hazard_per_year": round(ev_l / ex_l * 365.25, 4),
                 "born_legacy_events": ev_l, "born_legacy_exposure_days": round(ex_l, 1),
                 "by_month": {m: {"born_modern": share(v[1], v[0]), "modern_at_snapshot": share(v[2], v[0])}
                              for m, v in sorted(ent.items())},
                 "post_ga_total": share(sum(v[1] for v in ent.values()), sum(v[0] for v in ent.values())),
                 "born_legacy_later_migrated": late}
with open(os.path.join(R, "cohort_weekly.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["week_since_ga", "week_start", "entrants", "born_modern", "migrations", "ns_entrants", "ns_born_modern",
                "bulk_entrants", "bulk_born_modern"])
    for wk in sorted(weekly):
        if wk >= -60 and GA + (wk + 1) * WEEK <= T:
            w.writerow([wk, day(GA + wk * WEEK), weekly[wk]["entrants"], weekly[wk]["born_modern"], weekly[wk]["migrations"],
                        weekly[wk].get("ns_entrants", 0), weekly[wk].get("ns_born_modern", 0),
                        weekly[wk].get("bulk_entrants", 0), weekly[wk].get("bulk_born_modern", 0)])

# decomposition of the snapshot stock of v2 packages
v2_now = [p for p in pk if state(p, T)[2] == "v2"]
born = sum(p["v"][0][2] == "v2" for p in v2_now)
pre = sum(p["v"][0][0] <= GA and p["v"][0][2] != "v2" for p in v2_now)
S["decomposition"] = {"v2_at_snapshot": len(v2_now), "born_v2": share(born, len(v2_now)),
                      "migrated_incumbents": share(pre, len(v2_now)),
                      "migrated_post_ga_entrants": share(len(v2_now) - born - pre, len(v2_now))}
dn_sdk = sum(1 for p in pk if state(p, T)[2] in ("sdk1", "v2"))
S["decomposition"]["modern_share_runtime_declared"] = share(len(v2_now), dn_sdk)
pre_ga_alive = [p for p in pk if p["v"][0][0] <= GA and state(p, T)[2] in ("sdk1", "v2")]
post_ga_alive = [p for p in pk if p["v"][0][0] > GA and state(p, T)[2] in ("sdk1", "v2")]
S["decomposition"]["stock_share_post_ga_entrants"] = share(len(post_ga_alive), dn_sdk)
S["decomposition"]["modern_among_pre_ga"] = share(sum(state(p, T)[2] == "v2" for p in pre_ga_alive), len(pre_ga_alive))
S["decomposition"]["modern_among_post_ga"] = share(sum(state(p, T)[2] == "v2" for p in post_ga_alive), len(post_ga_alive))

# robustness: one vote per publisher namespace
wsum = collections.defaultdict(float)
cnt = collections.Counter(npm_ns.get(p["name"]) for p in pk if state(p, T)[2] in ("sdk1", "v2"))
num = den = 0.0
for p in pk:
    c = state(p, T)[2]
    if c in ("sdk1", "v2"):
        w_ = 1.0 / cnt[npm_ns.get(p["name"])]
        den += w_
        num += w_ * (c == "v2")
S["robustness"] = {"namespace_weighted_modern_share": round(num / den, 4), "namespaces": len(cnt)}
# robustness: exclude the 10 largest namespaces
big = {k for k, _ in cnt.most_common(10)}
k = sum(1 for p in pk if state(p, T)[2] == "v2" and npm_ns.get(p["name"]) not in big)
d = sum(1 for p in pk if state(p, T)[2] in ("sdk1", "v2") and npm_ns.get(p["name"]) not in big)
S["robustness"]["modern_share_excl_top10_namespaces"] = share(k, d)
# robustness: active packages only (released within 90 days of the snapshot)
k = sum(1 for p in pk if state(p, T)[2] == "v2" and T - state(p, T)[0] <= 90 * 86400)
d = sum(1 for p in pk if state(p, T)[2] in ("sdk1", "v2") and T - state(p, T)[0] <= 90 * 86400)
S["robustness"]["modern_share_active_90d"] = share(k, d)
dormant365 = sum(1 for p in pk if state(p, T)[2] == "sdk1" and T - state(p, T)[0] > 180 * 86400)
S["robustness"]["sdk1_no_release_180d"] = share(dormant365, snap["sdk1"])

# ---------------------------------------------------------------- PyPI cross-section
MCP_REL = sorted(((ts(r["date"]), r["sdk"]) for r in py_sdk), key=lambda x: x[0])
fm = json.load(open(os.path.join(D, "fastmcp_pypi_reqs.json")))
FAST = {r["v"]: r for r in fm["fastmcp->mcp"]}
SLIM = {r["v"]: r for r in fm["fastmcp-slim->mcp"]}
rem = json.load(open(os.path.join(D, "mcp_v2_removed_modules.json")))
imports = {}
for l in open(os.path.join(D, "pypi_imports.jsonl")):
    r = json.loads(l)
    imports[r["name"]] = r
PY_REV = {}
for r in py_sdk:
    v = Version(r["sdk"])
    PY_REV[r["sdk"]] = set(r["supported"] or [])
types_modern = {r["v"] for r in v2map if r["pkg"] == "mcp-types" and r["modern"]}
for r in fm["mcp->mcp-types"]:
    if r["req"]:
        tv = r["req"][0].split("==")[1].strip()
        if tv in types_modern:
            PY_REV[r["v"]] = LEGACY_ALL | {MODERN}
S["py_sdk"] = {"first_modern_stable": min((r["date"][:10], r["sdk"]) for r in py_sdk
                                          if MODERN in PY_REV.get(r["sdk"], ()) and not Version(r["sdk"]).is_prerelease),
               "v1_latest": max((Version(r["sdk"]), r["date"][:10]) for r in py_sdk if Version(r["sdk"]).major == 1)[1]}
py_T = None


def pick(cands, spec, hi=True):
    ok = [Version(v) for v in cands if not Version(v).is_prerelease and spec.contains(v)]
    if not ok:
        return None
    return str(max(ok) if hi else min(ok))


def mcp_spec(reqs, name="mcp"):
    spec, found = SpecifierSet(), False
    for q in reqs:
        if q.startswith("EXTRA:"):
            continue
        try:
            rq = Requirement(q)
        except Exception:
            continue
        if rq.name.lower().replace("_", "-") == name and (rq.marker is None or "extra" not in str(rq.marker)):
            spec &= rq.specifier
            found = True
    return spec if found else None


pyrecs = [json.loads(l) for l in open(os.path.join(D, "pypi_deps.jsonl"))]
pyrecs = [r for r in pyrecs if "error" not in r]
py_T = max(max(r["releases"].values()) for r in pyrecs if r["releases"])
S["pypi"] = {"packages": len(pyrecs), "snapshot": py_T[:10]}
pt = ts(py_T)
mcp_av = [v for t, v in MCP_REL if t <= pt]
fast_av = [v for v, r in FAST.items() if ts(r["date"]) <= pt]
cls = collections.Counter()
birth = collections.defaultdict(collections.Counter)
birth_m = collections.defaultdict(collections.Counter)
PY_GA = "2026-07-28T23"


def tally(r, c):
    cls[c] += 1
    if r["releases"]:
        first = min(r["releases"].values())
        birth["post_ga" if first > PY_GA else "pre_ga"][c] += 1
        if first > PY_GA:
            birth_m[first[:7]][c] += 1


brk = collections.Counter()
detail = []
REMOVED = set(rem["removed"][max(v for v in rem["removed"] if v in mcp_av)])
for r in pyrecs:
    sm, sf = mcp_spec(r["requires"]), mcp_spec(r["requires"], "fastmcp")
    last_rel = max(r["releases"].values()) if r["releases"] else None
    if sm is None and sf is None:
        tally(r, "opaque")
        continue
    via = sm is None
    if sf is not None:
        fv = pick(fast_av, sf)
        if fv is None:
            tally(r, "unresolvable")
            continue
        src = FAST[fv]["req"] or [q for q in SLIM.get(fv, {"req": []})["req"] if 'extra == "server"' in q]
        ind = SpecifierSet()
        for q in src:
            ind &= Requirement(q.split(";")[0]).specifier
        sm = ind if sm is None else (sm & ind)
    hi, lo = pick(mcp_av, sm), pick(mcp_av, sm, hi=False)
    if hi is None:
        tally(r, "unresolvable")
        continue
    f_mod, l_mod = MODERN in PY_REV[hi], MODERN in PY_REV[lo]
    if via:
        c = "via_fastmcp_modern" if f_mod else "via_fastmcp_legacy"
    elif l_mod:
        c = "requires_v2"
    elif f_mod:
        c = "floats_into_v2"
    else:
        c = "capped_v1"
    tally(r, c)
    if c == "floats_into_v2":
        im = imports.get(r["name"])
        if im is None or "error" in im:
            brk["not_scanned"] += 1
        else:
            mods = set(im["mods"])
            v1 = mods & REMOVED
            v2 = {m for m in mods if m.startswith("mcp.server.mcpserver")}
            if v1 and not v2:
                brk["imports_removed_v1_module"] += 1
                brk["... of which mcp.server.fastmcp"] += any(m.startswith("mcp.server.fastmcp") for m in v1)
                brk["... released after py v2 GA"] += last_rel > "2026-07-28T23"
                brk["... born after py v2 GA"] += min(r["releases"].values()) > PY_GA
            elif v1 and v2:
                brk["imports_both"] += 1
            elif v2:
                brk["imports_v2_only_module"] += 1
            elif mods:
                brk["retained_modules_only"] += 1
            else:
                brk["no_mcp_import_found"] += 1
    if c == "capped_v1":
        brk["capped_floor_rev_" + max(PY_REV[hi] or {"?"})] += 1
npy = len(pyrecs)
S["pypi"]["class"] = {k: share(v, npy) for k, v in cls.most_common()}
decl_py = npy - cls["opaque"] - cls["unresolvable"]
S["pypi"]["declared_n"] = decl_py
S["pypi"]["float_modern_reach_declared"] = share(cls["requires_v2"] + cls["floats_into_v2"] + cls["via_fastmcp_modern"], decl_py)
S["pypi"]["floor_modern_reach_declared"] = share(cls["requires_v2"], decl_py)
fl = cls["floats_into_v2"]
scanned = fl - brk["not_scanned"]
S["pypi"]["floats_into_v2_scan"] = {
    "n": fl, "scanned": scanned,
    "imports_removed_v1_module": share(brk["imports_removed_v1_module"], scanned),
    "of_which_fastmcp_path": brk["... of which mcp.server.fastmcp"],
    "of_which_released_after_ga": brk["... released after py v2 GA"],
    "of_which_first_released_after_ga": brk["... born after py v2 GA"],
    "imports_both": share(brk["imports_both"], scanned),
    "imports_v2_only_module": share(brk["imports_v2_only_module"], scanned),
    "retained_modules_only": share(brk["retained_modules_only"], scanned),
    "no_mcp_import_found": share(brk["no_mcp_import_found"], scanned)}
working = cls["requires_v2"] + cls["via_fastmcp_modern"] + brk["imports_both"] + brk["imports_v2_only_module"]
S["pypi"]["modern_reach_excluding_known_import_breaks"] = {
    "lower": share(working, decl_py),
    "upper": share(working + brk["retained_modules_only"] + brk["no_mcp_import_found"] + brk["not_scanned"], decl_py)}
S["pypi"]["known_broken_share_of_declared"] = share(brk["imports_removed_v1_module"], decl_py)
S["pypi"]["capped_v1_latest_rev"] = {k.replace("capped_floor_rev_", ""): v for k, v in brk.items() if k.startswith("capped_floor")}
S["pypi"]["by_birth"] = {k: {kk: share(vv, sum(v.values())) for kk, vv in v.most_common()} for k, v in sorted(birth.items())}
S["pypi"]["by_birth_declared_modern_floor"] = {
    k: share(v["requires_v2"] + v["via_fastmcp_modern"],
             sum(v.values()) - v["opaque"] - v["unresolvable"]) for k, v in sorted(birth.items())}
S["pypi"]["by_birth_month_post_ga"] = {
    k: share(v["requires_v2"] + v["via_fastmcp_modern"], sum(v.values()) - v["opaque"] - v["unresolvable"])
    for k, v in sorted(birth_m.items())}
# cohort view: packages whose latest release is after py v2 GA
after = [r for r in pyrecs if r["releases"] and max(r["releases"].values()) > "2026-07-28T23"]
S["pypi"]["latest_release_after_ga"] = share(len(after), npy)
S["pypi"]["scan_coverage"] = dict(collections.Counter(("error:" + r["error"]) if "error" in r else "ok" for r in imports.values()))
S["pypi"]["removed_module_count"] = len(REMOVED)

json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

col = {"2025-03-26": "#4C72B0", "2025-06-18": "#55A868", "2025-11-25": "#C44E52", "2026-07-28": "#8172B2"}
fig, ax = plt.subplots(figsize=(9, 4.6))
xs = [dt.date.fromisoformat(r["date"]) for r in rows]
for r in REVS:
    ax.plot(xs, [row["float_" + r] for row in rows], color=col[r], lw=2, label="%s  fresh-install reach" % r)
    ax.plot(xs, [row["floor_" + r] for row in rows], color=col[r], lw=1.2, ls="--", label="%s  declared-floor reach" % r)
    ax.axvline(dt.date.fromisoformat(r), color=col[r], lw=0.6, ls=":")
ax.set_ylabel("share of npm servers with a declared SDK dependency")
ax.set_ylim(0, 1.02)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.legend(fontsize=7, ncol=2, loc="center left")
ax.set_title("Reach of each MCP revision among registry-listed npm servers (dotted: spec date)")
fig.tight_layout()
fig.savefig(os.path.join(R, "fig1_reach.png"), dpi=150)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for lab, o, c in (("all incumbents", km_all, "k"), ("active (release in 90 d before GA)", km_act, "#C44E52"),
                  ("dormant", km_dor, "#4C72B0")):
    ds = sorted(o)
    axs[0].step([0] + ds, [0] + [1 - o[d][0] for d in ds], where="post", label=lab, color=c)
axs[0].set_xlabel("days since v2 SDK general availability")
axs[0].set_ylabel("cumulative share migrated to v2 SDK (Kaplan-Meier)")
axs[0].legend(fontsize=8)
wk = sorted(k for k in weekly if 0 <= k and GA + (k + 1) * WEEK <= T)
axs[1].bar(wk, [weekly[k]["born_modern"] / max(1, weekly[k]["entrants"]) for k in wk], color="#8172B2")
axs[1].set_xlabel("week of first publication (weeks since GA)")
axs[1].set_ylabel("share of new npm servers born on the v2 SDK")
fig.tight_layout()
fig.savefig(os.path.join(R, "fig2_migration.png"), dpi=150)

fig, ax = plt.subplots(figsize=(8, 3.6))
order = ["requires_v2", "via_fastmcp_modern", "floats_into_v2", "capped_v1", "via_fastmcp_legacy", "opaque", "unresolvable"]
ax.barh(order[::-1], [cls[k] for k in order[::-1]], color="#55A868")
ax.set_xlabel("registry-listed PyPI packages (snapshot %s)" % py_T[:10])
fig.tight_layout()
fig.savefig(os.path.join(R, "fig3_pypi.png"), dpi=150)
print("ok")
