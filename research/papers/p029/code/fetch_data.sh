#!/bin/sh
# Re-download the Who&When dataset (Zhang et al., ICML 2025; ~17 MB) into ../ext/ww (git-ignored).
cd "$(dirname "$0")/.." && mkdir -p ext && [ -d ext/ww ] || git clone --depth 1 https://github.com/ag2ai/Agents_Failure_Attribution.git ext/ww
