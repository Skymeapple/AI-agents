"""Measure token counts of the UDHR parallel corpus under 12 production tokenizers.

Output: results/token_counts.csv  (lang, tokenizer, tokens, bytes, chars)
        results/premium.csv       (lang x tokenizer premium relative to English)
All inputs are fetched by code/fetch_assets.sh into data/.
"""
import base64, glob, json, os, re, sys, unicodedata
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(DATA, "tiktoken_cache")
import tiktoken
from tokenizers import Tokenizer
import sentencepiece as spm

LLAMA3_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
O200K_PAT = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def load_bpe_file(path):
    ranks = {}
    for line in open(path, "rb"):
        if line.strip():
            tok, r = line.split()
            ranks[base64.b64decode(tok)] = int(r)
    return ranks


def tekken(path):
    d = json.load(open(path))
    n = d["config"]["default_vocab_size"] - d["config"]["default_num_special_tokens"]
    ranks = {base64.b64decode(v["token_bytes"]): v["rank"] for v in d["vocab"][:n]}
    return tiktoken.Encoding("tekken", pat_str=d["config"]["pattern"], mergeable_ranks=ranks, special_tokens={})


def build():
    T = DATA + "/tok/"
    toks = {}
    # (name, release year, vocab family) ; encode fn returns list of ids
    for n in ["p50k_base", "cl100k_base", "o200k_base"]:
        try:
            e = tiktoken.get_encoding(n)
            toks[n] = e.encode_ordinary
        except Exception as ex:  # r50k not in cache -> skip
            print("skip", n, ex, file=sys.stderr)
    e3 = tiktoken.Encoding("llama3", pat_str=LLAMA3_PAT, mergeable_ranks=load_bpe_file(T + "llama3.model"), special_tokens={})
    toks["llama3"] = e3.encode_ordinary
    e4 = tiktoken.Encoding("llama4", pat_str=O200K_PAT, mergeable_ranks=load_bpe_file(T + "llama4.model"), special_tokens={})
    toks["llama4"] = e4.encode_ordinary
    toks["tekken"] = tekken(T + "tekken_240911.json").encode_ordinary
    for n in ["claude_legacy", "deepseek_v3"]:
        tk = Tokenizer.from_file(T + n + ".json")
        toks[n] = (lambda tk: lambda s: tk.encode(s, add_special_tokens=False).ids)(tk)
    for n in ["mistral_v1", "gemma1", "gemma3"]:
        sp = spm.SentencePieceProcessor(model_file=T + n + ".model")
        toks[n] = (lambda sp: lambda s: sp.encode(s))(sp)
    return toks


def clean(raw):
    s = unicodedata.normalize("NFC", raw)
    lines = [re.sub(r"\s+", " ", l).strip() for l in s.splitlines()]
    return "\n".join(l for l in lines if l)


def main():
    toks = build()
    files = sorted(glob.glob(DATA + "/udhr2/*.txt"))
    rows = []
    for f in files:
        lang = os.path.basename(f)[:-4]
        try:
            txt = clean(open(f, encoding="utf-8").read())
        except UnicodeDecodeError:
            continue
        b = len(txt.encode("utf-8"))
        if b < 5000:  # truncated / partial translations
            continue
        for name, enc in toks.items():
            rows.append((lang, name, len(enc(txt)), b, len(txt)))
    os.makedirs(RES, exist_ok=True)
    with open(RES + "/token_counts.csv", "w") as fo:
        fo.write("lang,tokenizer,tokens,bytes,chars\n")
        for r in rows:
            fo.write(",".join(map(str, r)) + "\n")
    eng = {t: n for l, t, n, *_ in rows if l == "eng"}
    with open(RES + "/premium.csv", "w") as fo:
        names = list(toks)
        fo.write("lang,bytes_ratio," + ",".join(names) + "\n")
        by = {}
        for l, t, n, b, c in rows:
            by.setdefault(l, {})[t] = n; by[l]["_b"] = b
        be = by["eng"]["_b"]
        for l, d in by.items():
            fo.write(l + f",{d['_b']/be:.4f}," + ",".join(f"{d[t]/eng[t]:.4f}" for t in names) + "\n")
    print("languages:", len({r[0] for r in rows}), "tokenizers:", list(toks))


if __name__ == "__main__":
    main()
