"""E12 (revision): quantify the 'metastable' exploit-free state of KL-regularised GRPO (E6).
Math-like sequence verifier (alpha=.75, beta=.15, eta=.95, h0=0.009).  For each lam we track
exploit mass over 32,000 steps (4x the E6 horizon) and compare it with the exploit mass and
accuracy of the KL optimum (tilt at t_eff).  Response to review W7 (Fig. 6)."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import exp4_sequence as E
from exp6_kl_paradox import train_kl, t_eff
from theory import tilted_accuracy

R = os.path.join(os.path.dirname(__file__), "..", "results")
STEPS = 32000


def job(a):
    lam, pname, seed = a
    cls = {"tabular": E.Tabular, "factorised": E.Factorised}[pname]
    f = E.verifier(0.75, 0.15, 0.95)
    h = train_kl(cls(), f, lam, steps=STEPS, seed=seed)
    return lam, pname, seed, h[:, :3].tolist()


if __name__ == "__main__":
    f = E.verifier(0.75, 0.15, 0.95); p0 = E.Factorised().probs()
    jobs = [(lam, pn, s) for lam in [0.5, 0.3, 0.2] for pn in ["factorised", "tabular"] for s in range(3)]
    with Pool(int(os.environ.get("NPROC", 3))) as pool:
        res = pool.map(job, jobs)
    out = {"steps": STEPS, "rows": []}
    for lam in [0.5, 0.3, 0.2]:
        te = t_eff(p0, f, lam)
        lw = np.log(p0) + te * f; pt = np.exp(lw - lw.max()); pt /= pt.sum()
        row = {"lam": lam, "t_eff": float(te), "tilt_A": float(pt @ E.CORRECT), "tilt_exploit": float(pt @ E.EXPLOIT)}
        for pn in ["factorised", "tabular"]:
            hs = [np.array(h) for (l, p, s, h) in res if l == lam and p == pn]
            def at(h, step): return h[np.argmin(np.abs(h[:, 0] - step))]
            row[pn] = {
                "A_at_8000": [float(at(h, 8000)[1]) for h in hs], "A_at_32000": [float(h[-1, 1]) for h in hs],
                "exploit_at_8000": [float(at(h, 8000)[2]) for h in hs], "exploit_at_32000": [float(h[-1, 2]) for h in hs],
                "log10_exploit_slope_per_1k_steps_last_half": [
                    float(np.polyfit(h[h[:, 0] >= STEPS / 2, 0] / 1000, np.log10(h[h[:, 0] >= STEPS / 2, 2] + 1e-300), 1)[0]) for h in hs]}
        out["rows"].append(row); print(row, flush=True)
    json.dump(out, open(os.path.join(R, "exp12.json"), "w"), indent=1)
