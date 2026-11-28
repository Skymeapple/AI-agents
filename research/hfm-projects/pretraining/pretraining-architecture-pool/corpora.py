"""Fetch a few tens of megabytes of text from each public pretraining corpus.

  web    HuggingFaceFW/fineweb-edu          (educational web pages)
  dclm   mlfoundations/dclm-baseline-1.0    (filtered web crawl)
  code   codeparrot/codeparrot-clean-valid  (Python files from GitHub)
  math   HuggingFaceTB/finemath             (web pages with mathematics)

For each corpus the script lists the dataset's files, takes the first one that matches,
streams it, and writes <name>.train.txt and <name>.heldout.txt. Held-out documents come
after the training documents in the file and never overlap them.

Usage: python corpora.py <out dir> <train MB> [name ...]
"""
import gzip
import io
import json
import re
import sys
import urllib.request

SOURCES = {
    "web": ("HuggingFaceFW/fineweb-edu", r"^sample/10BT/.*\.parquet$", "text"),
    "dclm": ("mlfoundations/dclm-baseline-1.0", r"\.jsonl\.zst$", "text"),
    "code": ("codeparrot/codeparrot-clean-valid", r"\.json\.gz$", "content"),
    "math": ("HuggingFaceTB/finemath", r"^finemath-3plus/.*\.parquet$", "text"),
}
HELDOUT_MB = 4
AGENT = {"User-Agent": "research-script"}


def first_file(repo, pattern):
    with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}", headers=AGENT), timeout=120) as r:
        files = sorted(s["rfilename"] for s in json.load(r)["siblings"])
    match = [f for f in files if re.search(pattern, f)]
    if not match:
        raise SystemExit(f"{repo}: nothing matches {pattern}; files include {files[:30]}")
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{match[0]}"


def documents(url, field):
    if url.endswith(".parquet"):
        import fsspec
        import pyarrow.parquet as pq
        with fsspec.open(url, headers=AGENT).open() as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                for v in pf.read_row_group(g, columns=[field]).column(field).to_pylist():
                    yield v
        return
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=300)
    if url.endswith(".zst"):
        import zstandard
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
    else:
        stream = io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8", errors="replace")
    for line in stream:
        if line.strip():
            yield json.loads(line)[field]


def main():
    out, mb = sys.argv[1], float(sys.argv[2])
    for name in sys.argv[3:] or SOURCES:
        repo, pattern, field = SOURCES[name]
        url = first_file(repo, pattern)
        sizes, docs = {"train": 0, "heldout": 0}, {"train": 0, "heldout": 0}
        files = {k: open(f"{out}/{name}.{k}.txt", "wb") for k in sizes}
        sample = None
        for doc in documents(url, field):
            if not doc:
                continue
            part = "train" if sizes["train"] < mb * 1e6 else "heldout"
            b = (doc.strip() + "\n\n").encode("utf-8")
            files[part].write(b)
            sizes[part] += len(b)
            docs[part] += 1
            sample = sample or doc[:160].replace("\n", " ")
            if sizes["heldout"] >= HELDOUT_MB * 1e6:
                break
        for f in files.values():
            f.close()
        print(f"CORPUS {name}: {url}\n  train {sizes['train'] / 1e6:.1f} MB in {docs['train']} documents, held-out {sizes['heldout'] / 1e6:.1f} MB in {docs['heldout']} documents"
              f"\n  starts: {sample!r}", flush=True)


if __name__ == "__main__":
    main()
