# Find open-weight Lean 4 prover models in GGUF form, and the miniF2F-lean4 toolchain version.
python3 - <<'PY'
import json, urllib.request
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-script"}), timeout=60))
for q in ("Kimina-Prover gguf", "DeepSeek-Prover-V2 gguf", "Goedel-Prover gguf", "prover gguf", "Kimina-Prover", "Goedel-Prover-V2", "DeepSeek-Prover-V1.5"):
    try:
        res = get(f"https://huggingface.co/api/models?search={urllib.request.quote(q)}&limit=12&sort=downloads&full=true")
    except Exception as e:
        print(q, "ERR", e); continue
    print(f"\n== {q}: {len(res)}")
    for m in res:
        ggufs = [s["rfilename"] for s in m.get("siblings", []) if s["rfilename"].endswith(".gguf")]
        print(f"  {m['id']} dl={m.get('downloads')} gated={m.get('gated')} lic={[t for t in m.get('tags', []) if t.startswith('license')]} gguf={[g for g in ggufs if 'Q4_K_M' in g or 'q4_k_m' in g][:3]}")
PY
curl -sSL https://raw.githubusercontent.com/yangky11/miniF2F-lean4/main/lean-toolchain; echo
curl -sSL https://raw.githubusercontent.com/yangky11/miniF2F-lean4/main/lake-manifest.json | head -30
