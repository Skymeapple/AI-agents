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

import heapq
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


def replay(vms, policy):
    cap = np.zeros((64, 4))             # used capacity per server
    count = np.zeros(64, int)           # VMs on each server
    exit_at = np.zeros(64)              # predicted (or true) time the server empties
    opened = np.zeros(64)
    srv_class = np.zeros(64, int)       # longest lifetime class placed on the server since it opened
    n_srv, busy, seq = 0, 0.0, 0
    ends = []                           # (end time, tie-break, server, demand, tenant, type, start)
    pred = Predictor()
    noise, log_err = np.random.default_rng(0), []
    for start, end, tenant, vtype, dem in vms:
        while ends and ends[0][0] <= start:
            t, _, s, d, ten, ty, st = heapq.heappop(ends)
            cap[s] -= d
            count[s] -= 1
            pred.learn(ten, ty, t - st)
            if count[s] == 0:
                busy += t - opened[s]
                cap[s] = 0
        if policy in ("oracle", "class oracle"):
            life = end - start
        elif policy.startswith("noisy"):      # true lifetime times lognormal noise of the given log-scale spread
            life = (end - start) * float(np.exp(float(policy.split()[1]) * noise.standard_normal()))
        else:
            life = pred.predict(tenant, vtype)
            log_err.append(np.log(max(life, 1e-4) / max(end - start, 1e-4)))
        leave = start + life
        klass = 0 if life < 1 / 24 else (1 if life < 1.0 else 2)   # under an hour, under a day, longer
        fits = np.flatnonzero((cap[:n_srv] + dem <= 1.0 + 1e-9).all(axis=1) & (count[:n_srv] > 0))
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
        heapq.heappush(ends, (end, seq, s, dem, tenant, vtype, start))
    while ends:
        t, _, s, d, ten, ty, st = heapq.heappop(ends)
        count[s] -= 1
        if count[s] == 0:
            busy += t - opened[s]
    if policy == "lifetime-aware":
        replay.pred_spread = float(np.std(log_err))
    return busy


def main():
    path = sys.argv[1]
    modulus = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    pols = ("first fit", "best fit", "lifetime-aware", "oracle", "noisy 0.5", "noisy 1.0", "noisy 2.0", "class-aware", "class oracle")
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
    print(f"\nlog-scale error of the median predictor (last sample): {replay.pred_spread:.2f}; "
          "'noisy x' is the lifetime-aware rule fed true lifetimes with log-scale error x")

if __name__ == "__main__":
    main()
