# Check whether GitHub Models (free, rate-limited model inference for GitHub accounts)
# is reachable from a workflow with the built-in token. If it is, the agent-harness
# projects can be tested at small scale without any paid key.
curl -sS -m 60 -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Accept: application/vnd.github+json" \
  https://models.github.ai/catalog/models | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(len(d),'models in catalogue')
    for m in d[:60]: print(' ',m.get('id'),'|',m.get('rate_limit_tier'),'|',(m.get('limits') or {}).get('max_input_tokens'))
except Exception as e: print('catalogue error',e)"
for model in openai/gpt-4o-mini openai/gpt-4.1-mini meta/llama-3.3-70b-instruct; do
  echo "== $model"
  curl -sS -m 90 -X POST https://models.github.ai/inference/chat/completions \
    -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Accept: application/vnd.github+json" -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}],\"max_tokens\":5}" \
    -D - | grep -iE "^HTTP|x-ratelimit|\"content\"|\"message\"|error" | head -12
done
