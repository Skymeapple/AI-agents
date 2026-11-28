# Data provenance (p075)

Used in the paper:

- `epoch_eci_2026-11-18/eci_scores.csv` — Epoch AI, Epoch Capabilities Index (ECI) per-model scores with
  confidence bounds, release dates, organization, country and model accessibility (https://epoch.ai/data/eci_scores.csv,
  CC BY 4.0). epoch.ai was not reachable from this environment (proxy 403). The file was taken unchanged from the
  public GitHub repository `LongHuynh-vn/current-capabilities-index` (commit 9d19960, 2026-11-18), directory
  `eci-custom-2026-11-18/data/epoch_2026-11-18/`, whose SOURCES.md states it was downloaded from
  https://epoch.ai/data/eci_scores.csv on 2026-11-18. `benchmark_data.zip` (Epoch Benchmarking Hub raw data) is
  from the same directory and is kept for reference only (not used by the code).
- `epoch_eci_2026-11-11/eci_scores.csv` — same source, earlier vintage (2026-11-11); used for a vintage robustness check.

SHA-256 hashes are in `SHA256SUMS`. Because the files come via a third-party mirror, readers should re-download
from epoch.ai and re-run `code/analysis.py`; all results are regenerated from these CSVs.

`_unused_prior_attempt/` holds files left by an earlier, abandoned attempt at this project whose provenance could
not be established; nothing in the paper uses them.
