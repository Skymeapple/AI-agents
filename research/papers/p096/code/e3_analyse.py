"""Score every E3 indicator as a warning system: lead time at a control-calibrated false-alarm rate.

For indicator I: D(t) = I(t) - I(t_ref) (t_ref = first checkpoint >= 100 steps).
Alarm statistic A(t) = min(D(t), D(t_prev))  (two consecutive exceedances required).
Threshold h_I = max over control runs (labels independent of inputs) of max_t A(t).
Alarm time = first checkpoint with A(t) > h_I.  Emergence t_e = first checkpoint with test acc >= 0.9.
Lead = log10(t_e) - log10(t_alarm) (0 if alarm does not precede emergence).
Forecast: t_e_hat = t_alarm * r, r = median ratio on the other seeds (leave-one-out); report |log10 error|.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
INDS = ["acc", "logp", "fcS", "hcS", "gS", "supp", "dsupp", "lwS", "topk", "dtopk"]
T_REF = 100


def stat(r, ind):
    st = np.array(r["step"]); v = np.array(r[ind], float)
    i0 = np.searchsorted(st, T_REF)
    D = v - v[i0]
    A = np.full_like(D, -np.inf)
    A[i0 + 1:] = np.minimum(D[i0 + 1:], D[i0:-1])
    return st, A


def analyse(n):
    res = json.load(open(os.path.join(OUT, f"e3_parity_n{n}.json")))
    runs = [r for r in res if not r["control"]]
    ctrl = [r for r in res if r["control"]]
    out = dict(n=n, n_runs=len(runs), n_ctrl=len(ctrl), indicators={})
    te = []
    for r in runs:
        st = np.array(r["step"]); acc = np.array(r["acc"])
        j = np.where(acc >= 0.9)[0]
        te.append(st[j[0]] if len(j) else np.nan)
    te = np.array(te, float)
    out["t_emerge"] = te.tolist()
    out["emerged"] = int(np.isfinite(te).sum())
    for ind in INDS:
        ctrl_max = [np.max(stat(c, ind)[1]) for c in ctrl]
        h = float(np.max(ctrl_max))
        # leave-one-control-out false alarm: does held-out control exceed threshold from the others?
        fa = [float(ctrl_max[i] > max(ctrl_max[:i] + ctrl_max[i + 1:])) for i in range(len(ctrl))]
        ta = []
        for r in runs:
            st, A = stat(r, ind)
            j = np.where(A > h)[0]
            ta.append(st[j[0]] if len(j) else np.nan)
        ta = np.array(ta, float)
        lead = np.where(np.isfinite(ta) & np.isfinite(te) & (ta < te), np.log10(te) - np.log10(ta), 0.0)
        ok = np.isfinite(ta) & np.isfinite(te) & (ta < te)
        rho = float(spearmanr(ta[ok], te[ok])[0]) if ok.sum() >= 4 else None
        errs = []
        if ok.sum() >= 4:
            lr = np.log10(te[ok]) - np.log10(ta[ok])
            for i in range(ok.sum()):
                rmed = np.median(np.delete(lr, i))
                errs.append(abs(np.log10(ta[ok][i]) + rmed - np.log10(te[ok][i])))
        # baseline forecaster with no information: predict median log t_e of other seeds
        base = []
        lte = np.log10(te[np.isfinite(te)])
        for i in range(len(lte)):
            base.append(abs(np.median(np.delete(lte, i)) - lte[i]))
        out["indicators"][ind] = dict(threshold=h, loo_false_alarm=float(np.mean(fa)),
                                      t_alarm=ta.tolist(), lead_mean=float(lead.mean()),
                                      lead_median=float(np.median(lead)), lead_min=float(lead.min()),
                                      frac_warned=float(ok.mean()), spearman=rho,
                                      loo_forecast_abs_err_dec=float(np.mean(errs)) if errs else None,
                                      no_info_baseline_err_dec=float(np.mean(base)))
    return out


def main():
    allout = []
    for n in [int(a) for a in sys.argv[1:]] or [30, 40]:
        p = os.path.join(OUT, f"e3_parity_n{n}.json")
        if not os.path.exists(p):
            continue
        o = analyse(n)
        allout.append(o)
        print(f"n={n} runs={o['n_runs']} emerged={o['emerged']} ctrl={o['n_ctrl']} median t_e={np.nanmedian(o['t_emerge']):.0f}")
        for ind, d in o["indicators"].items():
            print(f"  {ind:6s} lead_mean={d['lead_mean']:.3f} median={d['lead_median']:.3f} min={d['lead_min']:.3f} "
                  f"warned={d['frac_warned']:.2f} rho={d['spearman']} fcErr={d['loo_forecast_abs_err_dec']} "
                  f"base={d['no_info_baseline_err_dec']:.3f} looFA={d['loo_false_alarm']:.2f}")
    json.dump(allout, open(os.path.join(OUT, "e3_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
