"""Value of lost compute (VoLC) implied by idle accelerator capital, per MWh of facility energy.
Inputs are public estimates (see paper Sec. 3.4): GB200 NVL72 rack price ~$3.0M (HSBC estimate, May 2024; press
reports range $2.8-3.4M), rack power ~120 kW (NVIDIA DGX GB200 user guide) to 132 kW (HPE spec sheet).
Amortization period, cost of capital and PUE are assumptions swept over plausible ranges.
VoLC here counts only accelerator-rack capital (no building, network fabric, staff or profit margin), so it is a lower bound."""
import itertools, json, os, numpy as np
def crf(r, n): return r * (1 + r) ** n / ((1 + r) ** n - 1)
rows = []
for price, kw, life, r, pue in itertools.product([2.8e6, 3.0e6, 3.4e6], [120, 132], [4, 5, 6], [0.08, 0.10, 0.12], [1.1, 1.2, 1.3]):
    v = price * crf(r, life) / (kw / 1000 * pue * 8760)
    rows.append(dict(price=price, kw=kw, life=life, r=r, pue=pue, v=v))
vs = np.array([x["v"] for x in rows])
central = 3.0e6 * crf(0.10, 5) / (0.126 * 1.2 * 8760)
out = dict(n=len(rows), min=float(vs.min()), p10=float(np.percentile(vs, 10)), median=float(np.median(vs)),
           p90=float(np.percentile(vs, 90)), max=float(vs.max()), central=float(central),
           central_assumptions="$3.0M rack, 126 kW, 5-yr life, 10% cost of capital, PUE 1.2")
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "volc_summary.json"), "w"), indent=1)
