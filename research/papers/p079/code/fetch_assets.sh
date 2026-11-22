#!/usr/bin/env bash
# Downloads all public inputs into ../data (tokenizer files, UDHR parallel corpus, World Bank indicators).
set -euo pipefail
D="$(cd "$(dirname "$0")/.." && pwd)/data"; mkdir -p "$D"/{whl,tok,tiktoken_cache}; cd "$D"
pip download --no-deps -q "mistral-common==1.13.1" "litellm==1.110.2" "anthropic==0.38.0" -d whl
unzip -o -q whl/litellm-*.whl 'litellm/litellm_core_utils/tokenizers/*' -d whl/lt
unzip -o -q whl/mistral_common-*.whl 'mistral_common/data/*' -d whl/mc
unzip -o -q whl/anthropic-0.38.0*.whl anthropic/tokenizer.json -d whl/an
# tiktoken cache files (named by sha1 of the official URL) shipped inside litellm
cp whl/lt/litellm/litellm_core_utils/tokenizers/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 tiktoken_cache/
cp whl/lt/litellm/litellm_core_utils/tokenizers/fb374d419588a4632f3f557e76b4b70aebbca790 tiktoken_cache/
cp whl/lt/litellm/litellm_core_utils/tokenizers/ec7223a39ce59f226a68acc30dc1af2788490e15 tiktoken_cache/
cp whl/lt/litellm/litellm_core_utils/tokenizers/deepseek_tokenizer.json tok/deepseek_v3.json
cp whl/an/anthropic/tokenizer.json tok/claude_legacy.json
cp whl/mc/mistral_common/data/tokenizer.model.v1 tok/mistral_v1.model
cp whl/mc/mistral_common/data/tekken_240911.json tok/tekken_240911.json
G=https://raw.githubusercontent.com
curl -sfL -o tok/gemma1.model $G/google/gemma_pytorch/main/tokenizer/tokenizer.model
curl -sfL -o tok/gemma3.model $G/google/gemma_pytorch/main/tokenizer/gemma3_cleaned_262144_v2.spiece.model
curl -sfL -o tok/llama3.model $G/meta-llama/llama-models/main/models/llama3/tokenizer.model
curl -sfL -o tok/llama4.model $G/meta-llama/llama-models/main/models/llama4/tokenizer.model
curl -sfL -o udhr2.zip $G/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip && unzip -o -q udhr2.zip
# World Bank WDI: GNI per capita Atlas (current US$), GNI per capita PPP, population; most recent non-empty value
for ind in NY.GNP.PCAP.CD NY.GNP.PCAP.PP.CD; do
  curl -sf "https://api.worldbank.org/v2/country/all/indicator/$ind?format=json&mrnev=1&per_page=400" -o "wb_$ind.json"
done
curl -sf "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&date=2023:2025&per_page=1000" -o wb_SP.POP.TOTL.json
curl -sf "https://api.worldbank.org/v2/country?format=json&per_page=400" -o wb_countries.json
sha256sum tok/* tiktoken_cache/* > checksums.txt
echo done
