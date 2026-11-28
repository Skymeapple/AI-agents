# p079 — The language tax in years: decomposing the AI affordability gap

Reproduce everything (CPU, about 1 minute after downloads):

    bash code/run_all.sh

Steps: `code/fetch_assets.sh` downloads 11 production tokenizers (from PyPI wheels of litellm / mistral-common / anthropic==0.38.0
and from GitHub raw for Gemma and Llama), the NLTK `udhr2` parallel corpus, and World Bank WDI indicators into `data/`;
`code/tokenize_udhr.py` writes `results/token_counts.csv` and `results/premium.csv`;
`code/analysis.py` writes `results/summary.json`, `results/country_decomposition.csv`, figures `results/fig1-5*.png`;
`code/robustness.py` writes `results/robustness.json`.

Hand-coded inputs: `code/country_language.csv` (predominant language per country, author-coded) and
`code/external_flores_o200k.csv` (premiums transcribed from arXiv:2606.24460 for validation).
Note: World Bank data are fetched live (vintage used in the paper: last updated 2026-11-11, GNI year 2025).
