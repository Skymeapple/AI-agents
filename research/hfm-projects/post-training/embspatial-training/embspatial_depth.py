"""EmbSpatial-Bench nearest/farthest questions: does a monocular depth estimate supply what boxes lack?

The no-image study found that object boxes answer left/right/above/under at 98-100%
but nearest/farthest at only about 69%. This script runs a small off-the-shelf
monocular depth model (Depth Anything V2 Small, not trained on this benchmark) on
each image on CPU and reads the estimated depth inside each object's box.

Usage: python embspatial_depth.py <test.parquet> <features.jsonl>
"""
from __future__ import annotations

import io
import json
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image


def main():
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    df = pd.read_parquet(sys.argv[1], columns=["question_id", "data_source", "relation", "image", "answer_options", "answer", "objects"])
    df = df[df.relation.isin(["close", "far"])]
    out = open(sys.argv[2], "w")
    rows, t0 = [], time.time()
    for n, r in enumerate(df.itertuples(), 1):
        img = Image.open(io.BytesIO(r.image["bytes"])).convert("RGB")
        W, H = img.size
        d = pipe(img)["predicted_depth"]
        d = np.asarray(d.squeeze().numpy() if hasattr(d, "numpy") else d, dtype=np.float32)   # higher = nearer (inverse depth)
        sy, sx = d.shape[0] / H, d.shape[1] / W
        names = list(r.objects["name"]); feats = []
        for o in r.answer_options:
            if o not in names:
                feats.append(None); continue
            a, b, c, e = [float(v) for v in r.objects["bbox"][names.index(o)]]
            x1, y1, x2, y2 = (a, b, c, e) if r.data_source == "ai2thor" else (a, b, a + c, b + e)
            X1, X2 = int(max(0, x1 * sx)), int(min(d.shape[1], max(x1 * sx + 1, x2 * sx)))
            Y1, Y2 = int(max(0, y1 * sy)), int(min(d.shape[0], max(y1 * sy + 1, y2 * sy)))
            p = d[Y1:Y2, X1:X2]
            if p.size == 0:
                feats.append(None); continue
            cy, cx = p.shape[0] // 2, p.shape[1] // 2
            core = p[p.shape[0] // 4: max(p.shape[0] // 4 + 1, 3 * p.shape[0] // 4), p.shape[1] // 4: max(p.shape[1] // 4 + 1, 3 * p.shape[1] // 4)]
            feats.append({"median": float(np.median(p)), "p90": float(np.percentile(p, 90)), "p10": float(np.percentile(p, 10)), "centre": float(p[cy, cx]), "core": float(np.median(core)),
                          "bottom": y2 / H, "area": (x2 - x1) * (y2 - y1) / (W * H)})
        rec = {"id": r.question_id, "source": r.data_source, "relation": r.relation, "answer": int(r.answer), "features": feats}
        rows.append(rec); out.write(json.dumps(rec) + "\n"); out.flush()
        if n % 100 == 0:
            print(f"{n} images, {(time.time() - t0) / n:.1f}s each", flush=True)

    def acc(key, rs):
        hit = 0
        for r in rs:
            c = [(f[key], i) for i, f in enumerate(r["features"]) if f]
            hit += (max(c) if r["relation"] == "close" else min(c))[1] == r["answer"]
        return 100 * hit / max(len(rs), 1)

    print(f"\nnearest/farthest questions: {len(rows)}")
    print(f"{'rule: nearest = highest ...':<44}{'all':>7}{'nearest':>9}{'farthest':>10}{'ai2thor':>9}{'mp3d':>7}{'scannet':>9}")
    for label, key in (("box bottom edge (no depth model)", "bottom"), ("estimated nearness, median in the box", "median"), ("estimated nearness, middle half of the box", "core"),
                       ("estimated nearness, box centre pixel", "centre"), ("estimated nearness, 90th percentile", "p90"), ("estimated nearness, 10th percentile", "p10")):
        sub = lambda f: [r for r in rows if f(r)]
        print(f"{label:<44}{acc(key, rows):>7.1f}{acc(key, sub(lambda r: r['relation'] == 'close')):>9.1f}{acc(key, sub(lambda r: r['relation'] == 'far')):>10.1f}"
              + "".join(f"{acc(key, sub(lambda r, s=s: r['source'] == s)):>{w}.1f}" for s, w in (("ai2thor", 9), ("mp3d", 7), ("scannet", 9))))
    best = max(("median", "core", "centre", "p90", "p10"), key=lambda k: acc(k, rows))
    print(f"\nwith left/right/above/under answered by box centres (2,411 of 2,434 correct), the whole benchmark would score {(2411 + acc(best, rows) / 100 * len(rows)) / 3640 * 100:.1f}% using '{best}' (rule picked on these same questions; see the table for the spread)")


if __name__ == "__main__":
    main()
