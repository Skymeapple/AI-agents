"""Tools for tasks 5.3, 5.4 and RP1. Each is built and tested on fixtures; real
use needs inputs that are blocked here (expert reviewers, outputs from human
postdocs, network access to a bibliographic API such as OpenAlex).

  build_review_packets / analyse_reviews   task 5.3, blinded expert review
  ImpactTracker                            task 5.4, quarterly citation and replication snapshots
  extract_open_questions / taste_pipeline  RP1, time-split question-impact prediction
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

# ---------------- 5.3 blinded review ----------------

IDENTIFYING = re.compile(r"\b(agent|AI|LLM|model|Claude|GPT|Gemini|Kosmos|our system|I am)\b", re.I)
RUBRIC = {"correctness": "1-5: claims supported by the evidence given",
          "novelty": "1-5: not already established in the literature at the stated date",
          "significance": "1-5: would change what the field does next",
          "clarity": "1-5: a peer could reproduce it from the text"}


def blind(text: str) -> str:
    return IDENTIFYING.sub("[REDACTED]", text)


def build_review_packets(items: list[dict], reviewers: list[str], per_item=3, seed=0):
    """items: {"id", "source": "agent"|"human", "text"}. Each item goes to `per_item`
    reviewers, sources balanced per reviewer, order shuffled, source hidden."""
    rng = random.Random(seed)
    key = {}
    packets = {r: [] for r in reviewers}
    load = Counter()
    for it in sorted(items, key=lambda _: rng.random()):
        code = hashlib.sha256(f"{seed}:{it['id']}".encode()).hexdigest()[:8]
        key[code] = {"id": it["id"], "source": it["source"]}
        for r in sorted(reviewers, key=lambda r: (load[r], rng.random()))[:per_item]:
            packets[r].append({"code": code, "text": blind(it["text"])})
            load[r] += 1
    for r in packets:
        rng.shuffle(packets[r])
    return packets, key, RUBRIC


def analyse_reviews(scores: list[dict], key: dict, n_perm=5000, seed=0):
    """scores: {"reviewer", "code", criterion: int}. Returns agent-minus-human mean per
    criterion with a permutation-test p-value (labels shuffled across items)."""
    rng = np.random.default_rng(seed)
    out = {}
    for crit in RUBRIC:
        per_item = {}
        for s in scores:
            per_item.setdefault(s["code"], []).append(s[crit])
        codes = list(per_item)
        vals = np.array([np.mean(per_item[c]) for c in codes])
        lab = np.array([key[c]["source"] == "agent" for c in codes])
        diff = vals[lab].mean() - vals[~lab].mean()
        perm = [vals[p].mean() - vals[~p].mean() for p in (rng.permutation(lab) for _ in range(n_perm))]
        out[crit] = {"agent_minus_human": float(diff),
                     "p_perm": float((np.sum(np.abs(perm) >= abs(diff)) + 1) / (n_perm + 1))}
    return out


# ---------------- 5.4 impact tracker ----------------

@dataclass
class ImpactTracker:
    """Quarterly snapshots of citations and replications for each agent finding.
    `fetch` maps a DOI to {"cited_by": int, "replications": int}; in production it
    calls a bibliographic API (blocked in this environment)."""
    fetch: Callable[[str], dict]
    findings: dict[str, dict] = field(default_factory=dict)
    snapshots: list[dict] = field(default_factory=list)

    def register(self, finding_id: str, doi: str, claimed_on: str):
        self.findings[finding_id] = {"doi": doi, "claimed_on": claimed_on}

    def snapshot(self, quarter: str):
        for fid, f in self.findings.items():
            m = self.fetch(f["doi"])
            self.snapshots.append({"quarter": quarter, "finding": fid, **m})

    def dashboard(self):
        latest = {}
        for s in self.snapshots:
            latest[s["finding"]] = s
        return {"findings": len(self.findings),
                "replicated": sum(1 for s in latest.values() if s["replications"] > 0),
                "total_citations": sum(s["cited_by"] for s in latest.values())}


# ---------------- RP1 research taste ----------------

OPEN_Q = re.compile(r"[^.]*\b(remains? (unknown|unclear|elusive|open)|is (still )?(unknown|poorly understood)|"
                    r"open (question|problem)|has yet to be|it is not known)\b[^.]*\.", re.I)


def extract_open_questions(abstract: str) -> list[str]:
    return [m.group(0).strip() for m in OPEN_Q.finditer(abstract)]


def _tfidf(docs):
    toks = [[w for w in re.findall(r"[a-z]+", d.lower()) if len(w) > 2] for d in docs]
    vocab = sorted({w for t in toks for w in t})
    idx = {w: i for i, w in enumerate(vocab)}
    df = Counter(w for t in toks for w in set(t))
    X = np.zeros((len(docs), len(vocab)))
    for r, t in enumerate(toks):
        for w, n in Counter(t).items():
            X[r, idx[w]] = (1 + math.log(n)) * (math.log((1 + len(docs)) / (1 + df[w])) + 1)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def taste_pipeline(records: list[dict], split_year: int, l2=1.0):
    """records: {"year", "question", "impact_t10", "early_citations"}. Train on
    year < split_year, test on later; compare a text model with an early-citation
    heuristic (protocol arm D). Returns Spearman rho for each on the test set."""
    X = _tfidf([r["question"] for r in records])
    y = np.log1p([r["impact_t10"] for r in records])
    tr = np.array([r["year"] < split_year for r in records])
    A = X[tr]
    w = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
    pred = X[~tr] @ w
    heur = np.array([r["early_citations"] for r in records])[~tr]
    return {"n_train": int(tr.sum()), "n_test": int((~tr).sum()),
            "rho_text_model": float(spearmanr(pred, y[~tr]).statistic),
            "rho_early_citation_heuristic": float(spearmanr(heur, y[~tr]).statistic)}
