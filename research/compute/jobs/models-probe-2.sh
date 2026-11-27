# Second check of GitHub Models from a workflow: show response headers, try the
# documented API-version header, the official SDK-style endpoint and DNS.
echo "== dns"; getent hosts models.github.ai models.inference.ai.azure.com api.github.com || true
body='{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":5}'
echo "== plain, verbose headers"
curl -sS -m 60 -X POST https://models.github.ai/inference/chat/completions -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Content-Type: application/json" -d "$body" -D - | head -c 1500; echo
echo "== with API version and user agent"
curl -sS -m 60 -X POST https://models.github.ai/inference/chat/completions -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" -H "User-Agent: research-script" -H "Content-Type: application/json" -d "$body" -D - | head -c 1500; echo
echo "== catalogue with version header"
curl -sS -m 60 https://models.github.ai/catalog/models -H "Authorization: Bearer $GH_MODELS_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" -D - | head -c 1200; echo
echo "== python"
python3 - <<'PY'
import os, json, urllib.request
req = urllib.request.Request("https://models.github.ai/inference/chat/completions", method="POST",
    data=json.dumps({"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "Reply with the single word: ready"}], "max_tokens": 5}).encode(),
    headers={"Authorization": "Bearer " + os.environ["GH_MODELS_TOKEN"], "Content-Type": "application/json", "Accept": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=60); print(r.status, dict(r.headers)); print(r.read()[:800])
except Exception as e:
    print("error", e, getattr(e, "read", lambda: b"")()[:800])
PY
