"""Online VM placement on the public Azure packing trace.

Public proxy for HFM's "Online Cloud Resource Allocator for Azure Trace Replay"
track. Data: Azure Packing Trace 2020 (Microsoft, CC-BY 4.0),
https://github.com/Azure/AzurePublicDataset (release "dataset-packing-2020").

Each VM request must be placed on a server the moment it arrives, without knowing
how long it will run. Servers have unit capacity in cores, memory, SSD and network
(the trace gives each VM type as a fraction of a machine). Cost is server-days:
the total time servers spend non-empty. There are no evictions or failures here,
unlike HFM's objective.

It tests the idea in research/human-frontier-mission.md section 8: predict each
VM's lifetime from its metadata and pack VMs that should leave together.

Policies:
  first fit        lowest-numbered server with room
  best fit         server left with the least spare capacity
  lifetime-aware   among servers with room, prefer the one whose predicted
                   emptying time is extended least by this VM, then best fit.
                   Lifetimes are predicted causally: the median lifetime of
                   already-finished VMs of the same tenant and type (falling back
                   to tenant, then to all VMs)
  re-predicting    the lifetime-aware rule with estimates that are revised as
                   VMs age: remaining life given age, re-estimated for a server
                   when a VM leaves it or when it outlives its estimate
  tenant affinity  no lifetime estimate at all: prefer a server that already holds
                   this tenant's VMs, then first fit
  batch affinity   prefer a server holding this tenant's VMs launched in the same
                   minute, then the tenant's other VMs, then first fit. Two thirds
                   of VMs arrive in such batches and most batch-mates stop together
  oracle           the same rule with true lifetimes (upper bound for the rule)
  class-aware      a coarser use of the same prediction: sort VMs into three
                   lifetime classes (under an hour, under a day, longer) and keep
                   classes on separate servers, then best fit
  class oracle     the class rule with true lifetimes
  noisy x          the lifetime-aware rule fed true lifetimes multiplied by
                   lognormal noise of log-scale spread x, to show how accurate a
                   predictor has to be

A random sample of tenants is replayed on one machine type so the run fits a small
container. HFM's replay, constraints and score are private; numbers are NOT
comparable to HFM's.

Usage: python research/hfm-experiments/cloud_placement.py <packing_trace_zone_a_v1.sqlite> [tenant-modulus]
"""
from __future__ import annotations

import bisect
import heapq
import random
import sqlite3
import statistics
import sys
from collections import defaultdict

import numpy as np

HORIZON = 14.0


def load(path, modulus, residue):
    c = sqlite3.connect(path)
    machine = c.execute("select machineId from vmType group by 1 order by count(*) desc limit 1").fetchone()[0]
    types = {t: np.array([core or 0, mem or 0, ssd or 0, nic or 0]) for t, core, mem, ssd, nic in
             c.execute("select vmTypeId, core, memory, ssd, nic from vmType where machineId=?", (machine,))}
    rows = c.execute("select vmId, tenantId, vmTypeId, priority, starttime, endtime from vm "
                     "where starttime >= 0 and tenantId % ? = ? order by starttime, vmId", (modulus, residue)).fetchall()
    vms = [(s, min(e if e is not None else HORIZON, HORIZON), ten, t, types[t]) for _, ten, t, _, s, e in rows
           if t in types and types[t].max() <= 1.0]
    return [v for v in vms if v[1] > v[0]]


class Predictor:
    """Causal lifetime estimate from VMs that have already finished."""

    def __init__(self):
        self.by_key, self.by_tenant, self.all = defaultdict(list), defaultdict(list), []

    def learn(self, tenant, vtype, life):
        for lst in (self.by_key[(tenant, vtype)], self.by_tenant[tenant], self.all):
            lst.append(life)
            if len(lst) > 200:
                del lst[0]

    def predict(self, tenant, vtype):
        for lst in (self.by_key[(tenant, vtype)], self.by_tenant[tenant], self.all):
            if len(lst) >= 3:
                return statistics.median(lst)
        return 0.05


class SurvivalPredictor:
    """Remaining lifetime given age, from finished VMs of the same tenant and type.

    The estimate is the median remaining life among past VMs that lived longer than
    the current age. A VM that has outlived every finished VM in its group is assumed
    to have about as long left as it has already lived (lifetimes are heavy-tailed).
    This follows the re-prediction idea of Google's LAVA scheduler (arXiv 2412.09840).
    """

    def __init__(self):
        self.groups = defaultdict(list)          # sorted lifetimes per group
        self.rng = random.Random(0)

    def learn(self, tenant, vtype, life):
        for key in ((tenant, vtype), tenant, None):
            lst = self.groups[key]
            bisect.insort(lst, life)
            if len(lst) > 300:
                del lst[self.rng.randrange(len(lst))]

    def remaining(self, tenant, vtype, age):
        for key in ((tenant, vtype), tenant, None):
            lst = self.groups.get(key)
            if lst:
                i = bisect.bisect_right(lst, age)
                n = len(lst) - i
                if n >= 3:
                    return lst[i + n // 2] - age
        return max(age, 0.02)


def replay(vms, policy):
    cap = np.zeros((64, 4))             # used capacity per server
    count = np.zeros(64, int)           # VMs on each server
    exit_at = np.zeros(64)              # predicted (or true) time the server empties
    opened = np.zeros(64)
    srv_class = np.zeros(64, int)       # longest lifetime class placed on the server since it opened
    n_srv, busy, seq = 0, 0.0, 0
    ends = []                           # (end time, tie-break, server, demand, tenant, type, start)
    pred = Predictor()
    surv = SurvivalPredictor()
    residents = defaultdict(dict)       # server -> {vm sequence number: (start, tenant, type)}
    repredict = policy == "re-predicting"
    affinity = policy in ("batch affinity", "tenant affinity")
    tenant_on, batch_on = defaultdict(dict), defaultdict(dict)   # server -> counts of a tenant's (and a batch's) VMs
    noise, log_err = np.random.default_rng(0), []
    for start, end, tenant, vtype, dem in vms:
        while ends and ends[0][0] <= start:
            t, q, s, d, ten, ty, st = heapq.heappop(ends)
            cap[s] -= d
            count[s] -= 1
            pred.learn(ten, ty, t - st)
            if affinity:
                tenant_on[s][ten] -= 1
                batch_on[s][(ten, int(st * 1440))] -= 1
            if repredict:
                surv.learn(ten, ty, t - st)
                residents[s].pop(q, None)
                # the server's emptying time is re-estimated from the VMs still on it
                exit_at[s] = t + max((surv.remaining(a, b, t - c) for c, a, b in residents[s].values()), default=0.0)
            if count[s] == 0:
                busy += t - opened[s]
                cap[s] = 0
        if policy in ("oracle", "class oracle"):
            life = end - start
        elif policy.startswith("noisy"):      # true lifetime times lognormal noise of the given log-scale spread
            life = (end - start) * float(np.exp(float(policy.split()[1]) * noise.standard_normal()))
        elif repredict:
            life = surv.remaining(tenant, vtype, 0.0)
            log_err.append(np.log(max(life, 1e-4) / max(end - start, 1e-4)))
        else:
            life = pred.predict(tenant, vtype)
            log_err.append(np.log(max(life, 1e-4) / max(end - start, 1e-4)))
        leave = start + life
        klass = 0 if life < 1 / 24 else (1 if life < 1.0 else 2)   # under an hour, under a day, longer
        fits = np.flatnonzero((cap[:n_srv] + dem <= 1.0 + 1e-9).all(axis=1) & (count[:n_srv] > 0))
        if repredict:
            for o in fits[exit_at[fits] < start]:        # servers that outlived their estimate: estimate again
                exit_at[o] = start + max(surv.remaining(a, b, start - c) for c, a, b in residents[o].values())
        if len(fits) == 0:
            empties = np.flatnonzero(count[:n_srv] == 0)
            if len(empties):
                s = empties[0]
            else:
                if n_srv == len(cap):
                    cap = np.vstack([cap, np.zeros_like(cap)])
                    count, exit_at, opened, srv_class = (np.concatenate([a, np.zeros_like(a)]) for a in (count, exit_at, opened, srv_class))
                s = n_srv
                n_srv += 1
            opened[s] = start
            exit_at[s] = leave
            srv_class[s] = klass
        else:
            if policy == "first fit":
                s = fits[0]
            elif policy in ("batch affinity", "tenant affinity"):
                # VMs a tenant launches together usually stop together, so keep them together:
                # prefer a server holding this tenant's VMs from the same minute, then any of
                # this tenant's VMs, then first fit
                minute = int(start * 1440)
                same_batch = [o for o in fits if batch_on[o].get((tenant, minute))] if policy == "batch affinity" else []
                same_tenant = [o for o in fits if tenant_on[o].get(tenant)]
                s = same_batch[0] if same_batch else (same_tenant[0] if same_tenant else fits[0])
            else:
                slack = (1.0 - cap[fits] - dem).sum(axis=1)
                if policy == "best fit":
                    s = fits[np.argmin(slack)]
                elif policy in ("class-aware", "class oracle"):
                    # keep short-lived and long-lived VMs on different servers, then best fit
                    s = fits[np.lexsort((slack, np.abs(srv_class[fits] - klass)))[0]]
                else:
                    # how much this VM would push back the server's emptying time; overdue servers count as "now"
                    extend = np.maximum(leave - np.maximum(exit_at[fits], start), 0.0)
                    s = fits[np.lexsort((slack, np.round(extend, 3)))[0]]
            exit_at[s] = max(exit_at[s], leave)
            srv_class[s] = max(srv_class[s], klass)
        cap[s] += dem
        count[s] += 1
        seq += 1
        if affinity:
            tenant_on[s][tenant] = tenant_on[s].get(tenant, 0) + 1
            key = (tenant, int(start * 1440))
            batch_on[s][key] = batch_on[s].get(key, 0) + 1
        if repredict:
            residents[s][seq] = (start, tenant, vtype)
        heapq.heappush(ends, (end, seq, s, dem, tenant, vtype, start))
    while ends:
        t, _, s, d, ten, ty, st = heapq.heappop(ends)
        count[s] -= 1
        if count[s] == 0:
            busy += t - opened[s]
    if policy in ("lifetime-aware", "re-predicting"):
        replay.spread[policy] = float(np.sqrt(np.mean(np.square(log_err))))
    return busy


def main():
    path = sys.argv[1]
    modulus = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    replay.spread = {}
    pols = ("first fit", "best fit", "tenant affinity", "batch affinity", "lifetime-aware", "re-predicting", "oracle", "noisy 0.5", "noisy 1.0", "noisy 2.0", "class-aware", "class oracle")
    if len(sys.argv) > 3:
        pols = tuple(sys.argv[3].split(","))
    print(f"{'sample':>6} {'VMs':>8} | " + " ".join(f"{p:>15}" for p in pols) + " | lower bound")
    rel = {p: [] for p in pols}
    for residue in range(4):  # four disjoint tenant samples
        vms = load(path, modulus, residue)
        lower = sum((e - s) * d.max() for s, e, _, _, d in vms)  # no packing can use fewer server-days than this
        r = {p: replay(vms, p) for p in pols}
        for p in pols:
            rel[p].append(r[p] / r["best fit"] - 1)
        print(f"{residue:>6} {len(vms):>8} | " + " ".join(f"{r[p]:>15.0f}" for p in pols) + f" | {lower:.0f}", flush=True)
    print("\nserver-days versus best fit (mean over four samples; samples better than best fit):")
    for p in pols:
        print(f"  {p:<16} {np.mean(rel[p]):>+7.1%}   {sum(g < 0 for g in rel[p])}/4")
    print("\nroot-mean-square log error of the lifetime estimate at arrival (last sample): "
          + ", ".join(f"{k} {v:.2f}" for k, v in replay.spread.items()))

if __name__ == "__main__":
    main()
