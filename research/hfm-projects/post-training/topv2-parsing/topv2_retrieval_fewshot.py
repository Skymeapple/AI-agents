"""TOPv2 without fine-tuning: retrieved examples in the prompt, with and without a grammar.

Follow-up to topv2_classical.py for HFM's TOPv2 track. That study found 10% of test
parses use a tree shape absent from the 12,288 training rows, and proposed
grammar-constrained decoding as the route to them. A GPU fine-tune is not available
here, so this tests the idea without training: a small open model (Qwen2.5-3B-Instruct,
4-bit, llama.cpp on a free 4-core runner) is shown the 8 training utterances most
similar to the test utterance, with their parses, and writes the parse.

Modes
  plain    free generation
  grammar  generation constrained to well-formed trees whose labels are intents and
           slots that occur in the training rows

Test set: 300 random test parses plus 100 more whose tree shape is unseen in the
training rows, so the unseen group is large enough to read.

Usage: python topv2_retrieval_fewshot.py <model.gguf> <train.parquet> <test.parquet> <plain|grammar> <out.jsonl>
       python topv2_retrieval_fewshot.py score <plain.jsonl> [<grammar.jsonl>]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def frame(parse):
    return " ".join(t for t in parse.split() if t == "]" or (t.startswith("[") and len(t) > 1))


def well_formed(p):
    depth = 0
    for t in p.split():
        depth += (t.startswith("[") and len(t) > 1) - (t == "]")
        if depth < 0:
            return False
    return depth == 0 and p.startswith("[")


def score(*paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        rnd = [r for r in rows if r["random"]]
        acc = lambda g, k="ok": f"{100 * np.mean([r[k] for r in g]):.1f}% of {len(g)}" if g else "n/a"
        print(f"--- {rows[0]['mode']}")
        print(f"  exact match on the random sample:        {acc(rnd)}")
        print(f"     tree shape seen in training:          {acc([r for r in rnd if r['seen']])}")
        print(f"  tree shape not seen (all such examples): {acc([r for r in rows if not r['seen']])}")
        print(f"     of those, shape right:                {acc([r for r in rows if not r['seen']], 'frame_ok')}")
        print(f"  output is a well-formed tree:            {acc(rows, 'well_formed')}")
        print(f"  nearest retrieved example has the right shape: {acc(rnd, 'top1_frame')}")
        print(f"  copying the nearest example's shape would be right at most: seen {acc([r for r in rnd if r['seen']], 'any_frame')}, unseen 0 by definition")
        print(f"  seconds per example: {np.mean([r['secs'] for r in rows]):.0f}")
    if len(paths) == 2:
        a, b = ({r["id"]: r for r in map(json.loads, open(p))} for p in paths)
        ids = sorted(set(a) & set(b))
        print(f"paired on {len(ids)}: plain only right {sum(a[i]['ok'] and not b[i]['ok'] for i in ids)}, grammar only right {sum(b[i]['ok'] and not a[i]['ok'] for i in ids)}")


def main():
    if sys.argv[1] == "score":
        return score(*sys.argv[2:])
    model_path, trp, tep, mode, out = sys.argv[1:6]
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from llama_cpp import Llama, LlamaGrammar
    norm = lambda s: " ".join(str(s).split())
    tr = [norm(x) for x in pd.read_parquet(trp)["semantic_parse"]]
    te = [norm(x) for x in pd.read_parquet(tep)["semantic_parse"]]
    rng = np.random.default_rng(0)
    pool = [tr[i] for i in rng.permutation(len(tr))[:12288]]            # the same 12,288 rows as topv2_classical.py
    words = lambda p: " ".join(t for t in p.split() if t != "]" and not (t.startswith("[") and len(t) > 1))
    pool_u = [words(p) for p in pool]
    frames = {frame(p) for p in pool}
    order = np.random.default_rng(1).permutation(len(te))
    random_ids = list(order[:300])
    unseen_extra = [i for i in order[300:] if frame(te[i]) not in frames][:100]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pool_u)
    P = vec.transform(pool_u)
    labels = sorted({t[1:] for p in pool for t in p.split() if t.startswith("[") and len(t) > 1})
    grammar = None
    if mode == "grammar":
        g = 'root ::= node\nnode ::= "[" label (" " item)+ " ]"\nitem ::= node | word\nword ::= [^ \\[\\]\\n]+\nlabel ::= ' + " | ".join(json.dumps(l) for l in labels) + "\n"
        grammar = LlamaGrammar.from_string(g, verbose=False)
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    sys_msg = ("Convert the utterance into its semantic parse. Copy every word of the utterance in order; wrap spans in [IN:INTENT ... ] and [SL:SLOT ... ] brackets "
               "exactly as in the examples. Reply with the parse only.")
    n_ok = 0
    with open(out, "w") as fo:
        for k, i in enumerate(random_ids + unseen_extra):
            gold, u = te[i], words(te[i])
            sims = (P @ vec.transform([u]).T).toarray().ravel()
            near = np.argsort(-sims)[:8]
            shots = "\n\n".join(f"Utterance: {pool_u[j]}\nParse: {pool[j]}" for j in near[::-1])      # nearest last
            t0 = time.time()
            r = llm.create_chat_completion(messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": f"{shots}\n\nUtterance: {u}\nParse:"}],
                                           max_tokens=160, temperature=0.0, grammar=grammar)
            pred = norm(r["choices"][0]["message"]["content"].strip().split("\n")[0])
            rec = dict(id=int(i), mode=mode, random=k < 300, seen=frame(gold) in frames, gold=gold, pred=pred, ok=pred == gold, frame_ok=frame(pred) == frame(gold),
                       well_formed=well_formed(pred), top1_frame=frame(pool[near[0]]) == frame(gold), any_frame=any(frame(pool[j]) == frame(gold) for j in near), secs=time.time() - t0)
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            n_ok += rec["ok"]
            if (k + 1) % 20 == 0:
                print(f"{mode}: {k + 1} examples, {100 * n_ok / (k + 1):.1f}% exact, {rec['secs']:.0f}s last", flush=True)
    score(out)


if __name__ == "__main__":
    main()
