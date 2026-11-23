"""
Layered compliance-partition model.

A firm chooses (i) which segments to serve, (ii) a partition of served segments into
weight-level model variants, (iii) a refinement of each weight variant into runtime-level
variants (system prompt / filter / output-pipeline configurations). Obligations are
assessed at a layer (weights, runtime, process). Within a variant, an ordered obligation is
met at the maximum level demanded by any segment in it; format obligations stack; a pair
of segments with mutually unsatisfiable requirements cannot share a variant at the layer
where the conflicting obligation is assessed.

The optimal structure is found exactly by subset dynamic programming over the 2^n masks
(n = 6 segments), which is exact for this n (the general problem is NP-hard; see paper,
Proposition 4).
"""
import math
import numpy as np
from obligations import SEGMENTS, DIMENSIONS, CONFLICTS, LAYER_OVERRIDES, MARKET_REF

N = len(SEGMENTS)
FULL = (1 << N) - 1
IDX = {s: i for i, s in enumerate(SEGMENTS)}
INF = float("inf")


def members(mask):
    return [i for i in range(N) if mask >> i & 1]


def loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_params(rng):
    """Synthetic priors (arbitrary units: 1 = value of serving a USD 1tn-GDP market).
    These are NOT estimates; results are reported as frequencies over this prior box."""
    p = {}
    p["m"] = np.array([MARKET_REF[s] * math.exp(rng.normal(0, 0.5)) for s in SEGMENTS])
    p["F_w"] = loguniform(rng, 0.5, 10.0)     # cost of maintaining an extra weight-level variant
    p["F_r"] = loguniform(rng, 0.02, 0.5)     # cost of an extra runtime-level variant
    a, h, b = {}, {}, {}
    for d, spec in DIMENSIONS.items():
        if spec["layer"] == "process":
            a[d] = loguniform(rng, 0.01, 0.2)
            b[d] = loguniform(rng, 0.005, 0.1)
            h[d] = 0.0
        elif d == "D8_political_content_norm":
            a[d] = loguniform(rng, 0.05, 0.5)   # alignment fine-tuning / filtering pipeline
            h[d] = loguniform(rng, 0.001, 0.4)  # value lost by non-CN users of a CN-aligned variant
            b[d] = 0.0
        elif spec["layer"] == "weights":
            a[d] = loguniform(rng, 0.1, 1.0)
            h[d] = 0.0
            b[d] = 0.0
        else:  # runtime formats
            a[d] = loguniform(rng, 0.01, 0.2)
            h[d] = rng.uniform(0.0, 0.02) if d == "D7_visible_disclosure" else 0.0
            b[d] = 0.0
    p["a"], p["h"], p["b"] = a, h, b
    p["eu_incident_level"] = int(rng.integers(1, 4))  # ambiguity of 'without undue delay'
    return p


def median_params():
    p = {"m": np.array([MARKET_REF[s] for s in SEGMENTS]),
         "F_w": math.sqrt(0.5 * 10), "F_r": math.sqrt(0.02 * 0.5)}
    a, h, b = {}, {}, {}
    for d, spec in DIMENSIONS.items():
        if spec["layer"] == "process":
            a[d], b[d], h[d] = math.sqrt(0.01 * 0.2), math.sqrt(0.005 * 0.1), 0.0
        elif d == "D8_political_content_norm":
            a[d], h[d], b[d] = math.sqrt(0.05 * 0.5), math.sqrt(0.001 * 0.4), 0.0
        elif spec["layer"] == "weights":
            a[d], h[d], b[d] = math.sqrt(0.1 * 1.0), 0.0, 0.0
        else:
            a[d], b[d] = math.sqrt(0.01 * 0.2), 0.0
            h[d] = 0.01 if d == "D7_visible_disclosure" else 0.0
    p["a"], p["h"], p["b"] = a, h, b
    p["eu_incident_level"] = 2
    return p


class Scenario:
    """A regime configuration: conflict hypothesis, layer overrides, coordination instruments."""

    def __init__(self, conflict_key="H1_CN_vs_USF", layer_override=None, instruments=()):
        self.conflict_key = conflict_key
        self.layers = {d: s["layer"] for d, s in DIMENSIONS.items()}
        if layer_override:
            self.layers.update(LAYER_OVERRIDES[layer_override])
        self.instruments = set(instruments)
        self.conflicts = []
        for c in CONFLICTS[conflict_key]:
            d, x, y = c[:3]
            lay = c[3] if len(c) > 3 else self.layers[d]   # layer at which the conflict binds
            self.conflicts.append((lay, IDX[x], IDX[y]))

    def req(self, d, p):
        """Requirement vector for dimension d after applying instruments."""
        r = dict(DIMENSIONS[d]["req"])
        if d == "D1_incident_reporting":
            r["EU"] = p["eu_incident_level"]
        harmonize = {"I2_harmonize_frameworks": "D2_safety_framework",
                     "I3_harmonize_training_summary": "D4_training_data_summary",
                     "I4_harmonize_marking": "D6_machine_readable_marking",
                     "I5_harmonize_visible": "D7_visible_disclosure",
                     "I6_harmonize_provenance": "D5_training_data_provenance"}
        for inst, dim in harmonize.items():
            if inst in self.instruments and d == dim:
                r = {s: (frozenset({"HARMONIZED"}) if len(v) else v) for s, v in r.items()}
        return [r[s] for s in SEGMENTS]


def block_cost(mask, layer, scen, p, reqs):
    """Compliance cost of one variant serving `mask` for all dimensions at `layer`."""
    ms = members(mask)
    for lay, i, j in scen.conflicts:
        if lay == layer and (mask >> i & 1) and (mask >> j & 1):
            return INF
    c = 0.0
    m = p["m"]
    for d, spec in DIMENSIONS.items():
        if scen.layers[d] != layer:
            continue
        r = reqs[d]
        if spec["kind"] == "ordered":
            L = max(r[i] for i in ms)
            c += p["a"][d] * L + p["h"][d] * sum(m[i] * (L - r[i]) for i in ms)
        else:
            U = set().union(*[r[i] for i in ms])
            c += p["a"][d] * len(U) + p["h"][d] * sum(m[i] * (len(U) - len(r[i])) for i in ms)
    return c


def process_cost(mask, scen, p, reqs):
    ms = members(mask)
    if not ms:
        return 0.0
    c = 0.0
    for d, spec in DIMENSIONS.items():
        if scen.layers[d] != "process":
            continue
        r = reqs[d]
        if spec["kind"] == "ordered":
            L = max(r[i] for i in ms)
            n_filings = sum(1 for i in ms if r[i] > 0)
            if "I1_MR_incident" in scen.instruments and d == "D1_incident_reporting":
                n_filings = min(n_filings, 1)
            c += p["a"][d] * L + p["b"][d] * n_filings
        else:
            U = set().union(*[r[i] for i in ms])
            c += p["a"][d] * len(U)
    return c


def partition_dp(cost_of_block):
    """best[B] = min over partitions of B of sum cost_of_block(part). Returns best, choice."""
    best = [INF] * (FULL + 1)
    choice = [0] * (FULL + 1)
    best[0] = 0.0
    for B in range(1, FULL + 1):
        low = B & -B
        rest = B ^ low
        sub = rest
        bv, bc = INF, 0
        while True:
            part = sub | low
            v = cost_of_block[part] + best[B ^ part]
            if v < bv:
                bv, bc = v, part
            if sub == 0:
                break
            sub = (sub - 1) & rest
        best[B], choice[B] = bv, bc
    return best, choice


def unroll(choice, B):
    parts = []
    while B:
        parts.append(choice[B])
        B ^= choice[B]
    return parts


def solve(scen, p):
    reqs = {d: scen.req(d, p) for d in DIMENSIONS}
    rcost = [INF] + [p["F_r"] + block_cost(B, "runtime", scen, p, reqs) for B in range(1, FULL + 1)]
    rbest, rchoice = partition_dp(rcost)
    wcost = [INF] + [p["F_w"] + block_cost(B, "weights", scen, p, reqs) + rbest[B]
                     for B in range(1, FULL + 1)]
    # the first weight variant and first runtime variant are free (baseline product);
    # forking cost is charged for each *additional* variant: subtract F_w + F_r once below.
    wbest, wchoice = partition_dp(wcost)
    best = None
    m = p["m"]
    for S in range(0, FULL + 1):
        if S == 0:
            net, cost = 0.0, 0.0
        else:
            cost = wbest[S] - p["F_w"] - p["F_r"] + process_cost(S, scen, p, reqs)
            net = sum(m[i] for i in members(S)) - cost
        if best is None or net > best[0] + 1e-12:
            best = (net, S, cost)
    net, S, cost = best
    wparts = unroll(wchoice, S) if S else []
    rparts = [unroll(rchoice, W) for W in wparts]
    return dict(net=net, served=S, cost=cost, wparts=wparts, rparts=rparts, reqs=reqs)


def conflict_chromatic_lower_bound(scen, S, layer_set):
    """Chromatic number of conflict graph restricted to served set S, for conflicts whose
    dimension is assessed at a layer in layer_set (exact by brute force; n <= 6)."""
    import itertools
    nodes = members(S)
    edges = [(i, j) for lay, i, j in scen.conflicts if lay in layer_set
             and i in nodes and j in nodes]
    if not edges:
        return 1 if nodes else 0
    for k in range(1, len(nodes) + 1):
        for col in itertools.product(range(k), repeat=len(nodes)):
            c = dict(zip(nodes, col))
            if all(c[i] != c[j] for i, j in edges):
                return k
    return len(nodes)


def export_events(sol, scen):
    """Return list of (exporter i, importer j, dim) such that j's product is held to a
    strictly higher level/extra token because of i's requirement (rule export)."""
    ev = []
    reqs = sol["reqs"]
    for wi, W in enumerate(sol["wparts"]):
        for R in sol["rparts"][wi]:
            for d, spec in DIMENSIONS.items():
                lay = scen.layers[d]
                if lay == "process" and spec["kind"] == "format":
                    continue  # jurisdiction-specific paperwork is not imposed on others' users
                block = {"weights": W, "runtime": R, "process": sol["served"]}[lay]
                ms = members(block)
                r = reqs[d]
                for j in members(R):
                    for i in ms:
                        if i == j:
                            continue
                        if spec["kind"] == "ordered":
                            L = max(r[k] for k in ms)
                            if r[i] == L and r[j] < L:
                                ev.append((i, j, d))
                        else:
                            if len(r[i] - r[j]) > 0:
                                ev.append((i, j, d))
    return ev
