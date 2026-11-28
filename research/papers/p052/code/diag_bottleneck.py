"""Diagnostic for the b=1 bottleneck mimic: compare the steered (report, logit) path with the
natural (report, logit) relation, to explain rho > 1. Output: ../results/diag_bottleneck.json"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from testbed import ReferenceAgent, BottleMLP, sample_episodes, train_mimic, run_agent
from run_all import set_probe, controlled_episodes, report_gradient_direction
torch.set_num_threads(2)
A = ReferenceAgent(0); out = {}
for b in [1, 64]:
    torch.manual_seed(301)
    m = train_mimic(lambda b=b: BottleMLP(b=b), A, 301, epochs=80)
    rng = np.random.default_rng(7)
    # natural: probe at t=11 after shocks at various lags -> range of states
    x = sample_episodes(3000, 12, rng); set_probe(x, 11, 0.45)
    l, r, h = run_agent(m, x); la, ra, _ = A.forward(x)
    nat = np.polyfit(r[:, 11], l[:, 11], 1)[0]
    natA = np.polyfit(ra[:, 11], la[:, 11], 1)[0]
    # mimic's own report vs A's state
    rep_fit = np.polyfit(ra[:, 11], r[:, 11], 1)[0]
    g = report_gradient_direction(m, x); u = g / np.linalg.norm(g)
    xb = controlled_episodes(500, 12, [], rng); set_probe(xb, 11, 0.45)
    l0, r0, h0 = run_agent(m, xb)
    path = []
    for a in np.linspace(0, 3, 31) * np.sqrt(h[:, 11].var(0).sum()):
        st = np.zeros((500, 12, h.shape[-1]), np.float32); st[:, 11] = a * u
        l1, r1, _ = run_agent(m, xb, steer=st)
        path.append([float((r1[:, 11] - r0[:, 11]).mean()), float((l1[:, 11] - l0[:, 11]).mean())])
    # natural conditional: logit as function of report bins among no-recent-shock vs shocked
    bins = np.linspace(0, 1.2, 7); idx = np.digitize(r[:, 11], bins)
    natcurve = [[float(r[idx == k, 11].mean()), float(l[idx == k, 11].mean())] for k in range(1, len(bins)) if (idx == k).sum() > 20]
    out[str(b)] = dict(natural_slope_mimic=float(nat), natural_slope_A=float(natA),
                       mimic_report_vs_A_state_slope=float(rep_fit), steer_path=path, natural_curve=natcurve)
    print(b, "nat slope mimic %.2f A %.2f; report/state slope %.3f" % (nat, natA, rep_fit))
    print("  steer path (dreport, dlogit):", [(round(p[0], 2), round(p[1], 2)) for p in path[::5]])
    print("  natural curve (report, logit):", [(round(p[0], 2), round(p[1], 2)) for p in natcurve])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "diag_bottleneck.json"), "w"), indent=1)
