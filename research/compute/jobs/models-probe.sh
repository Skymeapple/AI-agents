# Check whether GitHub Models (free, rate-limited model inference for GitHub accounts)
# is reachable from a workflow with the built-in token. If it is, the agent-harness
# projects can be tested at small scale without any paid key.
echo "token present: $([ -n "$GH_MODELS_TOKEN" ] && echo yes || echo no)"
echo "== catalogue"
curl -sS -m 60 -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Accept: application/vnd.github+json" \
  -o cat.json -w "http %{http_code}, %{size_download} bytes, %{content_type}\n" https://models.github.ai/catalog/models
python3 - <<'PY'
import json
try:
    d = json.load(open("cat.json"))
    print(len(d), "models")
    for m in d[:80]:
        print(" ", m.get("id"), "|", m.get("rate_limit_tier"), "|", (m.get("limits") or {}).get("max_input_tokens"))
except Exception as e:
    print("not JSON:", e, "| first bytes:", open("cat.json", "rb").read(300))
PY
for model in openai/gpt-4o-mini openai/gpt-4.1-mini meta/llama-3.3-70b-instruct; do
  echo "== $model"
  curl -sS -m 90 -X POST https://models.github.ai/inference/chat/completions \
    -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Accept: application/vnd.github+json" -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ready\"}],\"max_tokens\":5}" \
    -D headers.txt -o body.json -w "http %{http_code}, %{size_download} bytes, %{time_total}s\n"
  grep -iE "x-ratelimit" headers.txt | head -8
  head -c 600 body.json; echo
done
echo "== older endpoint"
curl -sS -m 90 -X POST https://models.inference.ai.azure.com/chat/completions \
  -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":5}' \
  -D headers.txt -o body.json -w "http %{http_code}, %{size_download} bytes, %{time_total}s\n"
grep -iE "x-ratelimit" headers.txt | head -8
head -c 600 body.json; echo
echo "== GitHub API route"
curl -sS -m 60 -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Accept: application/vnd.github+json" \
  -w "\nhttp %{http_code}\n" "https://api.github.com/models/catalog" | head -c 400; echo
