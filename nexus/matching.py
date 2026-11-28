"""Open problems against discovered directions: who is working on what.

The repository holds several catalogues of unsolved problems (AI-100, ai2035,
the Human Frontier challenges, ten cosmology problems, ten mathematics problems,
worldsim's ten global problems) and, separately, a way to find the research
directions that are actually active in a corpus (:mod:`sciscout.discovery`).
Nothing connected the two. This module does, and reports three lists:

*matched*    a problem and a direction that is plainly about it
*neglected*  a problem the corpus could have covered, with no direction on it
*orphans*    an active direction tied to no catalogued problem

The method is lexical, like the discovery it sits on: a problem is a TF-IDF
vector in the corpus's own vocabulary, a direction is the centroid of its works,
and the score is their cosine. That is a weak matcher, so the rule that turns
scores into links refuses rather than guesses, in the spirit of
:mod:`sciscout.classify`: a link needs an absolute similarity floor *and* the
problem must beat the runner-up problem of its own catalogue by a margin. A
direction that sits between two problems is linked to neither.

Two failure modes are kept apart because they mean opposite things. A corpus of
machine-learning papers says nothing about dark matter, and calling that problem
neglected would be reading an absence of data as a finding. So every catalogue
declares a field, the corpus declares the fields it covers, and only an unmatched
problem inside those fields is *neglected*; the rest are *outside* the corpus and
nothing is concluded. The declaration is a judgement and is printed as one. (A
data-driven test was tried first -- how much of a problem's vocabulary the corpus
uses, and its best similarity to any single paper -- and neither separated
cosmology from machine learning on the real corpus, so it was not shipped.)

None of this is asserted. :func:`evaluate_leave_one_out` measures it on the one
real, labelled corpus the repository has (the ai2035 frontier harvest): remove a
question's papers and the question should turn up neglected; remove the question
and its papers' directions should turn up orphaned.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from sciscout.discovery import DiscoveredTrack, DiscoveryResult, discover, tokenize
from sciscout.models import Work
from sciscout.provenance import Estimate, Provenance, assumed, observed, parse_provenance, sourced
from sciscout.sources.local import LocalCorpus

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "nexus" / "matching"
FRONTIER_DIR = ROOT / "ai2035" / "frontier"
DEMO_CORPUS = ROOT / "data" / "demo_corpus.json"
DEFAULT_OUT = ROOT / "examples" / "nexus-matching.md"

# -- the match rule ------------------------------------------------------------
#
# Every constant here is a judgement call and says so. The floor was picked on
# the tuning half of the known-pairs experiment (six AI-100 questions) and the
# report prints how it does on the half it never saw.

SIMILARITY_FLOOR: Estimate[float] = Estimate(
    0.15,
    assumed(
        "cosine below which a problem and a direction share too little vocabulary "
        "to call a match; the value that best separated matched from neglected on the "
        "tuning half of the known-pairs experiment (6 questions), not derived"
    ),
    label="similarity floor",
)
MARGIN_OVER_RUNNER_UP: Estimate[float] = Estimate(
    1.25,
    assumed(
        "how far the best problem of a catalogue must beat the runner-up before a "
        "direction is linked to it. The tuning half could not tell 1.0 from 1.5 at the "
        "shipped floor, so this is a precaution, not a fitted value. Lower than "
        "sciscout.classify's 1.5 because neighbouring problems in one catalogue overlap "
        "far more than commercial sectors do"
    ),
    label="margin over runner-up",
)
REAL_ABSTRACT_THRESHOLD: Estimate[float] = Estimate(
    0.10,
    assumed(
        "discovery similarity floor for real abstracts; sciscout's default 0.16 was "
        "set on the synthetic demo corpus and places only 46% of the ai2035 "
        "frontier papers, 0.10 places 83%"
    ),
    label="discovery threshold",
)
# Which field each catalogue belongs to, and which fields the shipped real corpus
# covers. Judgement calls, used only to decide whether an unmatched problem may
# be called neglected.
AI100_FIELDS = {
    "能力与扩展": "ai", "推理与认知": "ai", "智能体": "ai", "对齐与安全": "ai",
    "可解释性": "ai", "评估与科学方法": "ai", "AI本身的性质": "ai-nature",
    "社会与经济": "ai-society", "治理与地缘政治": "ai-governance",
    "基础设施与资源": "ai-infrastructure",
}
HFM_AI_FIELDS = {"harness", "pre", "post"}  # agent harnesses, pre- and post-training
FIELD_PROVENANCE = assumed(
    "each catalogue's field, and the fields a corpus covers, are declared by hand: "
    "AI-100 by its own category column (technical categories -> 'ai'), Human Frontier "
    "by its field column (harness / pre / post -> 'ai'), the rest by catalogue"
)
FRONTIER_FIELDS: Estimate[frozenset] = Estimate(
    frozenset({"ai"}),
    assumed(
        "the ai2035 frontier harvest holds technical machine-learning papers only; it "
        "was built from search queries for 20 questions, so it is not a sample of all "
        "of machine learning either"
    ),
    label="fields the corpus covers",
)
TUNE_FLOORS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
TUNE_MARGINS = (1.0, 1.25, 1.5, 2.0)


# -- problems --------------------------------------------------------------------


@dataclass
class Problem:
    """One catalogued open problem, with the English text it is matched on."""

    id: str
    catalogue: str
    title: str
    text: str
    provenance: Provenance
    field: str = "unknown"
    # Tokens that must not be used when this problem is scored in a leak-free
    # evaluation (the queries its labelled papers were harvested with).
    withheld: frozenset[str] = frozenset()

    @property
    def key(self) -> str:
        return f"{self.catalogue}:{self.id}"


def _load_ai100(root: Path) -> list[Problem]:
    questions = root / "research" / "questions.json"
    gloss_path = DATA_DIR / "ai100_en.yaml"
    if not questions.exists() or not gloss_path.exists():
        return []
    gloss = yaml.safe_load(gloss_path.read_text())
    provenance = parse_provenance(gloss["source"], "ai100_en.yaml")
    problems = []
    for row in json.loads(questions.read_text()):
        entry = gloss["questions"].get(int(row["id"]))
        if not entry:
            continue
        text = entry["en"] + ". " + ". ".join(entry["terms"])
        problems.append(Problem(f"{int(row['id']):03d}", "ai100", entry["en"], text, provenance,
                                AI100_FIELDS.get(row.get("category", ""), "ai")))
    return problems


def _load_ai2035(root: Path) -> list[Problem]:
    path = root / "ai2035" / "questions.yaml"
    if not path.exists():
        return []
    queries_path = root / "ai2035" / "frontier" / "queries.yaml"
    queries = yaml.safe_load(queries_path.read_text()) if queries_path.exists() else {}
    titles_path = DATA_DIR / "ai2035_en.yaml"
    titles = yaml.safe_load(titles_path.read_text())["titles"] if titles_path.exists() else {}
    provenance = sourced(
        "arXiv search phrases listed for each question",
        reference="ai2035/questions.yaml",
        needs_review=True,  # the file says the question list itself is a reconstruction
    )
    problems = []
    for q in yaml.safe_load(path.read_text())["questions"]:
        phrases = [str(p) for p in q["arxiv"]]
        withheld = frozenset(t for phrase in queries.get(q["id"], []) for t in tokenize(str(phrase)))
        problems.append(Problem(
            q["id"], "ai2035", titles.get(q["id"], q["slug"].replace("-", " ")),
            ". ".join(phrases), provenance, "ai", withheld,
        ))
    return problems


def _load_hfm(root: Path) -> list[Problem]:
    paths = sorted((root / "research").glob("hfm-challenges-*.csv"))
    if not paths:
        return []
    path = paths[-1]
    provenance = sourced("challenge title and field", reference=f"research/{path.name}")
    problems = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            problems.append(Problem(
                row["source_id"].removeprefix("discovery_live_"), "hfm", row["title"],
                row["title"], provenance,
                "ai" if row["field"] in HFM_AI_FIELDS else row["field"],
            ))
    return problems


def _load_cosmos(root: Path) -> list[Problem]:
    problems = []
    for path in sorted((root / "cosmos").glob("p[0-9][0-9]_*.py")):
        source = path.read_text()
        match = re.match(r'\s*"""(.*?)"""', source, re.S)
        if not match:
            continue
        doc = match.group(1).strip()
        title = re.sub(r"^Problem \d+ - ", "", doc.splitlines()[0]).rstrip(".?")
        problems.append(Problem(
            path.stem[:3], "cosmos", title[0].upper() + title[1:], doc,
            sourced("module docstring", reference=f"cosmos/{path.name}"), "cosmology",
        ))
    return problems


def _load_maths(root: Path) -> list[Problem]:
    path = root / "open-problems-sim" / "README.md"
    if not path.exists():
        return []
    provenance = sourced("problem name and what is simulated", reference="open-problems-sim/README.md")
    problems = []
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].isdigit():
            problems.append(Problem(f"m{int(cells[0]):02d}", "maths", cells[1],
                                    f"{cells[1]}. {cells[2]}", provenance, "mathematics"))
    return problems


def _load_world(root: Path) -> list[Problem]:
    try:
        from worldsim.model import OUTCOMES
    except Exception:  # worldsim is optional here; its absence is not an error
        return []
    grouped: dict[str, list[str]] = {}
    for outcome in OUTCOMES:
        grouped.setdefault(outcome.problem, []).append(outcome.label)
    provenance = sourced("problem names and outcome labels", reference="worldsim/model.py OUTCOMES")
    return [
        Problem(f"w{i:02d}", "world", name, f"{name}. " + ". ".join(labels), provenance, "global")
        for i, (name, labels) in enumerate(grouped.items(), start=1)
    ]


CATALOGUES = {
    "ai100": ("AI-100 open questions", _load_ai100),
    "ai2035": ("ai2035 basic-science questions", _load_ai2035),
    "hfm": ("Human Frontier challenges", _load_hfm),
    "cosmos": ("Ten cosmology problems", _load_cosmos),
    "maths": ("Ten mathematics problems", _load_maths),
    "world": ("worldsim's ten global problems", _load_world),
}


def load_catalogues(root: Path = ROOT, only: list[str] | None = None) -> list[Problem]:
    """Every problem catalogue found under ``root``; a missing one is skipped."""
    problems: list[Problem] = []
    for name, (_, loader) in CATALOGUES.items():
        if only and name not in only:
            continue
        problems.extend(loader(root))
    return problems


# -- corpora ----------------------------------------------------------------------


def load_frontier(directory: Path = FRONTIER_DIR) -> list[Work]:
    """The ai2035 frontier harvest: real arXiv abstracts, labelled by question.

    The label (which question's query returned the paper) is kept in
    ``extra['question']`` for scoring only; matching never reads it.
    """
    works: dict[str, Work] = {}
    for path in sorted(Path(directory).glob("Q*.json")):
        qid = path.stem
        for paper in json.loads(path.read_text())["papers"]:
            existing = works.get(paper["id"])
            if existing:
                existing.extra["question"].append(qid)
                continue
            works[paper["id"]] = Work(
                id=paper["id"], title=paper["title"], abstract=paper.get("abstract", ""),
                published=dt.date.fromisoformat(paper["published"][:10]),
                disciplines=list(paper.get("categories", [])), source="arxiv",
                extra={"question": [qid]},
            )
    return list(works.values())


def load_corpus(path: Path) -> tuple[list[Work], str]:
    """Works from ``path`` and a one-line description of what they are."""
    path = Path(path)
    if path.is_dir():
        works = load_frontier(path)
        return works, (
            f"{len(works)} real arXiv abstracts from `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}` "
            "(machine-learning papers from 2025 on, harvested by search query per ai2035 question)"
        )
    result = LocalCorpus(path).harvest()
    works = result.works
    synthetic = bool(works) and all(w.id.startswith("synthetic:") for w in works)
    kind = "SYNTHETIC, machine-generated" if synthetic else "local"
    return works, f"{len(works)} {kind} works from `{path.name}`"


# -- the vector space --------------------------------------------------------------


def _terms(text: str) -> list[str]:
    """Unigrams and adjacent bigrams. A problem statement is a dozen words, and
    'world model' or 'dark matter' says far more than either word alone."""
    tokens = tokenize(text)
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def _problem_terms(problem: Problem, leak_free: bool) -> list[str]:
    """Like :func:`_terms`, but bigrams never span two listed phrases."""
    out: list[str] = []
    for segment in re.split(r"[.,;]", problem.text):
        tokens = tokenize(segment)
        if leak_free:
            tokens = [t for t in tokens if t not in problem.withheld]
        out += tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return out


@dataclass
class _Space:
    index: dict[str, int]
    idf: np.ndarray
    df: np.ndarray
    n_docs: int
    rows: list[dict[int, float]]  # one L2-normalised TF-IDF row per work

    @classmethod
    def build(cls, works: list[Work], min_df: int = 2) -> "_Space":
        documents = [_terms(w.text()) for w in works]
        frequency: Counter[str] = Counter()
        for doc in documents:
            frequency.update(set(doc))
        vocabulary = sorted(t for t, n in frequency.items() if n >= min_df)
        index = {t: i for i, t in enumerate(vocabulary)}
        n = len(works)
        df = np.array([frequency[t] for t in vocabulary], dtype=float)
        # Same smoothed idf and sublinear tf as sciscout.discovery._weighted_rows.
        idf = np.log((1 + n) / (1 + df)) + 1.0
        rows = []
        for doc in documents:
            counts = Counter(index[t] for t in doc if t in index)
            row = {c: (1.0 + math.log(k)) * idf[c] for c, k in counts.items()}
            norm = math.sqrt(sum(v * v for v in row.values()))
            rows.append({c: v / norm for c, v in row.items()} if norm else {})
        return cls(index, idf, df, n, rows)

    def centroid(self, members: list[int]) -> np.ndarray:
        vec = np.zeros(len(self.index), dtype=np.float32)
        for i in members:
            for c, v in self.rows[i].items():
                vec[c] += v
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def problem_vector(self, terms: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.index), dtype=np.float32)
        for t in set(terms):
            col = self.index.get(t)
            if col is not None:
                vec[col] = self.idf[col]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


# -- matching ----------------------------------------------------------------------


@dataclass
class Link:
    problem: Problem
    direction: DiscoveredTrack
    similarity: float
    runner_up: float  # best other problem of the same catalogue, on this direction

    @property
    def margin(self) -> float:
        return self.similarity / self.runner_up if self.runner_up > 0 else float("inf")


@dataclass
class Refusal:
    """A direction's best problem in one catalogue that the rule turned down."""

    problem: Problem
    direction: DiscoveredTrack
    similarity: float
    runner_up: float
    reason: str
    ambiguous: bool  # cleared the floor, lost on the margin


@dataclass
class Scored:
    """Problem-by-direction similarities, before any rule is applied."""

    problems: list[Problem]
    directions: list[DiscoveredTrack]
    similarity: np.ndarray  # problems x directions


@dataclass
class _Clustered:
    """A corpus, its directions, and their centroids in the corpus's vector space."""

    works: list[Work]
    found: DiscoveryResult
    space: _Space
    centroids: np.ndarray

    @classmethod
    def build(cls, works: list[Work], found: DiscoveryResult) -> "_Clustered":
        space = _Space.build(works)
        row_of = {id(w): i for i, w in enumerate(works)}
        centroids = (
            np.stack([space.centroid([row_of[id(w)] for w in d.works]) for d in found.tracks])
            if found.tracks else np.zeros((0, len(space.index)), dtype=np.float32)
        )
        return cls(works, found, space, centroids)

    def score(self, problems: list[Problem], leak_free: bool = False) -> Scored:
        vectors = (
            np.stack([self.space.problem_vector(_problem_terms(p, leak_free)) for p in problems])
            if problems else np.zeros((0, len(self.space.index)), dtype=np.float32)
        )
        return Scored(list(problems), list(self.found.tracks), vectors @ self.centroids.T)


def score(problems: list[Problem], found: DiscoveryResult, works: list[Work],
          leak_free: bool = False) -> Scored:
    """Cosine between every problem and every direction's centroid.

    ``works`` must be the corpus ``found`` was discovered in; it supplies the
    vocabulary and document frequencies. ``leak_free`` drops each problem's
    ``withheld`` tokens, for scoring against papers harvested with them.
    """
    return _Clustered.build(works, found).score(problems, leak_free)


@dataclass
class MatchResult:
    problems: list[Problem]
    directions: list[DiscoveredTrack]
    similarity: np.ndarray  # problems x directions
    links: list[Link]
    refusals: list[Refusal]
    floor: float
    margin: float
    # Fields the corpus is taken to cover; None means nobody has said, and then
    # no unmatched problem can be called neglected.
    corpus_fields: frozenset[str] | None

    def links_for(self, problem: Problem) -> list[Link]:
        return [l for l in self.links if l.problem is problem]

    @property
    def matched(self) -> list[Problem]:
        seen = {id(l.problem) for l in self.links}
        return [p for p in self.problems if id(p) in seen]

    def _unmatched(self) -> list[Problem]:
        seen = {id(l.problem) for l in self.links}
        return [p for p in self.problems if id(p) not in seen]

    def in_scope(self, problem: Problem) -> bool:
        return self.corpus_fields is not None and problem.field in self.corpus_fields

    @property
    def neglected(self) -> list[Problem]:
        """Unmatched problems in a field the corpus covers."""
        return [p for p in self._unmatched() if self.in_scope(p)]

    @property
    def outside(self) -> list[Problem]:
        """Unmatched problems the corpus cannot speak to either way."""
        return [p for p in self._unmatched() if not self.in_scope(p)]

    @property
    def orphans(self) -> list[DiscoveredTrack]:
        linked = {id(l.direction) for l in self.links}
        return [d for d in self.directions if id(d) not in linked]

    def best_direction(self, problem: Problem) -> tuple[DiscoveredTrack, float] | None:
        if not self.directions:
            return None
        row = self.similarity[self.problems.index(problem)]
        j = int(np.argmax(row))
        return self.directions[j], float(row[j])

    def counts(self) -> dict[str, Estimate[int]]:
        def obs(value: int, what: str) -> Estimate[int]:
            return Estimate(value, observed(
                f"{what}, counted from the match at floor {self.floor:g} and margin "
                f"{self.margin:g}"), label=what)
        return {
            "matched": obs(len(self.matched), "problems matched to a direction"),
            "neglected": obs(len(self.neglected), "in-scope problems with no direction"),
            "outside": obs(len(self.outside), "unmatched problems outside the corpus"),
            "orphans": obs(len(self.orphans), "directions tied to no problem"),
            "links": obs(len(self.links), "problem-direction links"),
        }


def apply_rule(
    scored: Scored,
    floor: float | None = None,
    margin: float | None = None,
    corpus_fields: frozenset[str] | None = None,
    without: Problem | None = None,
) -> MatchResult:
    """Turn similarities into links, refusing where the evidence is thin.

    A link (problem, direction) is made when the problem is the direction's best
    match within its catalogue, clears ``floor``, and beats that catalogue's
    runner-up by ``margin``. A direction may therefore be linked once per
    catalogue -- catalogues overlap, and "continual learning" is rightly both an
    AI-100 and an ai2035 problem -- but never to two problems of one catalogue.

    ``without`` drops one problem from the catalogue, as if it had never been
    written down.
    """
    floor = SIMILARITY_FLOOR.value if floor is None else floor
    margin = MARGIN_OVER_RUNNER_UP.value if margin is None else margin
    keep = [i for i, p in enumerate(scored.problems) if p is not without]
    problems = [scored.problems[i] for i in keep]
    similarity = scored.similarity[keep]

    by_catalogue: dict[str, list[int]] = {}
    for i, p in enumerate(problems):
        by_catalogue.setdefault(p.catalogue, []).append(i)
    links: list[Link] = []
    refusals: list[Refusal] = []
    for members in by_catalogue.values():
        block = similarity[members]  # catalogue problems x directions
        if block.shape[1] == 0:
            continue
        order = np.argsort(-block, axis=0, kind="stable")
        for j, direction in enumerate(scored.directions):
            i = members[order[0, j]]
            best = float(block[order[0, j], j])
            runner_up = float(block[order[1, j], j]) if len(members) > 1 else 0.0
            if best <= 0:
                continue
            ratio = best / runner_up if runner_up > 0 else float("inf")
            if best < floor:
                # Only near misses are worth a reader's time.
                if best >= 0.75 * floor:
                    refusals.append(Refusal(
                        problems[i], direction, best, runner_up,
                        f"similarity {best:.2f} is under the floor {floor:g}", False))
            elif ratio < margin:
                rival = problems[members[order[1, j]]]
                refusals.append(Refusal(
                    problems[i], direction, best, runner_up,
                    f"only {ratio:.2f}x the runner-up ({rival.key}), needing {margin:g}x", True))
            else:
                links.append(Link(problems[i], direction, best, runner_up))
    return MatchResult(problems, scored.directions, similarity, links, refusals, floor, margin,
                       corpus_fields)


def match(
    problems: list[Problem],
    found: DiscoveryResult,
    works: list[Work],
    floor: float | None = None,
    margin: float | None = None,
    corpus_fields: frozenset[str] | None = None,
    leak_free: bool = False,
) -> MatchResult:
    """Score ``problems`` against the directions in ``found`` and apply the rule."""
    return apply_rule(score(problems, found, works, leak_free), floor, margin, corpus_fields)


# -- measuring the matcher -----------------------------------------------------------


def _majority_label(direction: DiscoveredTrack, label_key: str) -> tuple[str | None, float]:
    counts: Counter[str] = Counter()
    for w in direction.works:
        labels = w.extra.get(label_key)
        for label in ([labels] if isinstance(labels, str) else labels or []):
            counts[label] += 1
    if not counts:
        return None, 0.0
    label, n = counts.most_common(1)[0]
    return label, n / direction.size


@dataclass
class LeaveOneOut:
    """Counts behind the three claims, for one set of labelled problems."""

    problems: int = 0
    # Present: the problem and its papers are both in the corpus.
    linked_correctly: int = 0  # problems linked to a direction of their own papers
    links: int = 0
    links_correct: int = 0
    # Papers removed and the corpus re-clustered: the problem should go unmatched.
    refused_when_absent: int = 0
    # Problem removed from the catalogue: its papers' directions should be orphans.
    own_directions: int = 0
    own_directions_orphaned: int = 0

    def rates(self) -> dict[str, float]:
        def ratio(a: int, b: int) -> float:
            return a / b if b else float("nan")
        return {
            "match recall": ratio(self.linked_correctly, self.problems),
            "link precision": ratio(self.links_correct, self.links),
            "neglect detection": ratio(self.refused_when_absent, self.problems),
            "orphan detection": ratio(self.own_directions_orphaned, self.own_directions),
        }

    @property
    def verdict_accuracy(self) -> float:
        """Mean of match recall and neglect detection: how often the matched /
        neglected verdict on a problem is right, weighting both cases equally."""
        if not self.problems:
            return 0.0
        return (self.linked_correctly + self.refused_when_absent) / (2 * self.problems)


class LeaveOneOutExperiment:
    """Test the matcher on a corpus whose works are labelled.

    ``labels`` maps a problem's key to the corpus label of the papers that are
    its own. For each such problem in turn: (1) with everything present, is it
    linked to a direction whose works mostly carry its label; (2) with those
    papers removed and the corpus re-clustered, does it come back unmatched;
    (3) with the problem removed from the catalogue, do the directions made of
    its papers come back orphaned.

    The clustering and the similarities do not depend on the floor or margin, so
    they are computed once here and :meth:`evaluate` re-applies the rule.
    ``problems`` is the whole catalogue the labelled problems compete in.
    ``leak_free`` matches without each problem's ``withheld`` words. A direction
    counts as a label's own when at least ``min_purity`` of its works carry it.
    """

    def __init__(self, problems: list[Problem], labels: dict[str, str], works: list[Work],
                 label_key: str = "question", threshold: float | None = None,
                 leak_free: bool = False, min_purity: float = 0.5,
                 cache: dict | None = None) -> None:
        threshold = REAL_ABSTRACT_THRESHOLD.value if threshold is None else threshold
        self.problems = list(problems)
        self.labels = labels
        self.labelled = [p for p in self.problems if p.key in labels]
        # Clusterings are shared between experiments over the same corpus.
        cache = cache if cache is not None else {}

        def clustered(excluded: str | None) -> _Clustered:
            key = (excluded, threshold)
            if key not in cache:
                kept = [w for w in works
                        if excluded is None or excluded not in w.extra.get(label_key, [])]
                cache[key] = _Clustered.build(kept, discover(kept, threshold=threshold))
            return cache[key]

        full = clustered(None)
        self.full = full.score(self.problems, leak_free)
        self.own = {}
        for d in full.found.tracks:
            label, purity = _majority_label(d, label_key)
            self.own[id(d)] = label if purity >= min_purity else None
        self.absent = {
            p.key: clustered(labels[p.key]).score(self.problems, leak_free) for p in self.labelled
        }

    def evaluate(self, subset: list[Problem] | None = None, floor: float | None = None,
                 margin: float | None = None) -> LeaveOneOut:
        """Score ``subset`` (default: every labelled problem) under one rule."""
        subset = self.labelled if subset is None else subset
        keys = {p.key for p in subset}
        out = LeaveOneOut(problems=len(subset))
        present = apply_rule(self.full, floor, margin)
        for link in present.links:
            if link.problem.key in keys:
                out.links += 1
                out.links_correct += self.own[id(link.direction)] == self.labels[link.problem.key]
        for p in subset:
            label = self.labels[p.key]
            out.linked_correctly += any(
                self.own[id(l.direction)] == label for l in present.links_for(p))
            absent = apply_rule(self.absent[p.key], floor, margin)
            out.refused_when_absent += not absent.links_for(p)
            without = apply_rule(self.full, floor, margin, without=p)
            orphaned = {id(d) for d in without.orphans}
            mine = [d for d in self.full.directions if self.own[id(d)] == label]
            out.own_directions += len(mine)
            out.own_directions_orphaned += sum(id(d) in orphaned for d in mine)
        return out


def load_known_pairs(path: Path | None = None) -> dict[str, str]:
    """AI-100 problem key -> the ai2035 label of the papers that should match it."""
    path = path or DATA_DIR / "known_pairs.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    parse_provenance(raw["source"], path.name)
    return {f"ai100:{int(k):03d}": str(v) for k, v in raw["pairs"].items()}


def tune_and_test(works: list[Work], label_key: str = "question",
                  threshold: float | None = None, root: Path = ROOT) -> dict:
    """Measure the matcher three ways on the labelled frontier corpus.

    *own words*: each ai2035 question against the papers its own query returned.
    The matching text and the harvest query nearly coincide, so this is the easy
    case and mostly shows the plumbing works.

    *leak-free*: the same with every harvest-query word withheld.

    *independent wording*: AI-100 questions paired by hand with an ai2035
    question, matched on a gloss written separately from the harvest queries,
    competing against the other 99 AI-100 questions. The floor was chosen on
    the first half of the pairs (by :attr:`LeaveOneOut.verdict_accuracy`); the
    second half is the held-out score.
    """
    cache: dict = {}
    ai2035 = load_catalogues(root, only=["ai2035"])
    own_labels = {p.key: p.id for p in ai2035}
    easy = LeaveOneOutExperiment(ai2035, own_labels, works, label_key, threshold, False, cache=cache)
    clean = LeaveOneOutExperiment(ai2035, own_labels, works, label_key, threshold, True, cache=cache)
    left = [len(_problem_terms(p, True)) for p in ai2035]

    ai100 = load_catalogues(root, only=["ai100"])
    pairs = load_known_pairs()
    paired = LeaveOneOutExperiment(ai100, pairs, works, label_key, threshold, False, cache=cache)
    ordered = sorted(paired.labelled, key=lambda p: p.id)
    half = len(ordered) // 2
    tune, test = ordered[:half], ordered[half:]
    # Ties on verdict accuracy are broken by link precision on the tuning half.
    grid = []
    for floor in TUNE_FLOORS:
        for margin in TUNE_MARGINS:
            result = paired.evaluate(tune, floor, margin)
            precision = result.links_correct / result.links if result.links else 0.0
            grid.append(((result.verdict_accuracy, precision), floor, margin))
    top = max(g[0] for g in grid)
    return {
        "own_words": easy.evaluate(),
        "leak_free": clean.evaluate(),
        "leak_free_terms_left": float(np.median(left)) if left else 0.0,
        "tune_ids": [p.id for p in tune],
        "test_ids": [p.id for p in test],
        "grid": grid,
        "best": [(f, m) for b, f, m in grid if b == top],
        "tune": paired.evaluate(tune),
        "test": paired.evaluate(test),
    }


# -- report ------------------------------------------------------------------------


def _span(direction: DiscoveredTrack) -> str:
    dates = sorted(w.published for w in direction.works if w.published)
    if not dates:
        return "undated"
    return f"{dates[0]:%Y-%m}..{dates[-1]:%Y-%m}"


def _cell(text: str, limit: int = 70) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render(result: MatchResult, corpus_note: str, found: DiscoveryResult,
           evaluation: dict | None = None, synthetic: bool = False, top: int = 25,
           fields_note: Estimate | None = None) -> str:
    counts = result.counts()
    scope = ("not declared" if result.corpus_fields is None
             else ", ".join(f"`{f}`" for f in sorted(result.corpus_fields)))
    lines = [
        "# Open problems against active research directions",
        "",
        f"Corpus: {corpus_note}.",
        f"Discovery: {found.summary()} (similarity floor {found.threshold:g}).",
        f"Catalogues: {len(result.problems)} problems in "
        f"{len({p.catalogue for p in result.problems})} catalogues.",
        f"Fields the corpus is taken to cover: {scope}.",
        "",
    ]
    if synthetic:
        lines += [
            "**This corpus is synthetic.** Its directions were planted by a generator, so a "
            "match here shows the matcher can connect a problem to a direction that shares "
            "its vocabulary, and an orphan shows the refusal rule working. Neither says "
            "anything about what real research groups are doing. No field is declared for "
            "it, so no problem is called neglected.",
            "",
        ]
    lines += [
        "| Result | Count |", "| --- | --- |",
        f"| Problems matched to at least one direction | {counts['matched'].value} |",
        f"| Problems neglected (in a field the corpus covers, no direction on them) | {counts['neglected'].value} |",
        f"| Problems outside the corpus (cannot be judged from it) | {counts['outside'].value} |",
        f"| Directions orphaned (tied to no catalogued problem) | {counts['orphans'].value} of {len(result.directions)} |",
        "",
        "The counts are `observed`: computed from this corpus by the rule below. The "
        "rule's constants and the field declarations are `assumed`.",
        "",
        "## The rule",
        "",
    ]
    for estimate, value in ((SIMILARITY_FLOOR, result.floor), (MARGIN_OVER_RUNNER_UP, result.margin)):
        note = "" if value == estimate.value else f" (overridden from {estimate.value:g} for this run)"
        lines.append(f"- {estimate.label} = {value:g}{note}  <- {estimate.provenance.describe()}")
    lines.append(f"- problem fields  <- {FIELD_PROVENANCE.describe()}")
    if fields_note is not None:
        lines.append(f"- {fields_note.label} = {scope}  <- {fields_note.provenance.describe()}")
    lines += [
        "",
        "A direction is linked to a problem when the problem is its best match within "
        "its catalogue, clears the floor, and beats that catalogue's runner-up by the "
        "margin. An unmatched problem is *neglected* only if its field is one the corpus "
        "covers; otherwise it is *outside* the corpus and nothing is concluded.",
        "",
        "## By catalogue",
        "",
        "| Catalogue | Problems | Matched | Neglected | Outside corpus | Matching text |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    matched_ids = {id(p) for p in result.matched}
    neglected_ids = {id(p) for p in result.neglected}
    for name, (title, _) in CATALOGUES.items():
        members = [p for p in result.problems if p.catalogue == name]
        if not members:
            continue
        n_matched = sum(id(p) in matched_ids for p in members)
        n_neglected = sum(id(p) in neglected_ids for p in members)
        lines.append(
            f"| {title} (`{name}`) | {len(members)} | {n_matched} | {n_neglected} "
            f"| {len(members) - n_matched - n_neglected} "
            f"| {members[0].provenance.grade.value}: {_cell(members[0].provenance.detail, 80)} |"
        )
    lines.append("")

    if evaluation:
        lines += _render_evaluation(evaluation)

    lines += ["## Matched", "",
              "| Problem | Direction (distinctive terms) | Works | Active | Similarity | Margin |",
              "| --- | --- | --- | --- | --- | --- |"]
    ordered = sorted(result.links, key=lambda l: (l.problem.catalogue, l.problem.id, -l.similarity))
    for link in ordered:
        margin = "sole" if math.isinf(link.margin) else f"{link.margin:.2f}x"
        lines.append(
            f"| `{link.problem.key}` {_cell(link.problem.title, 60)} "
            f"| {_cell(', '.join(link.direction.terms[:5]), 60)} | {link.direction.size} "
            f"| {_span(link.direction)} | {link.similarity:.2f} | {margin} |"
        )
    if not ordered:
        lines.append("| _none_ | | | | | |")

    lines += ["", "## Neglected problems", "",
              "In a field the corpus covers, with no direction plainly about them. The nearest "
              "direction is shown so each verdict can be checked.", "",
              "| Problem | Nearest direction | Similarity | Why not linked |",
              "| --- | --- | --- | --- |"]
    for problem in sorted(result.neglected, key=lambda p: (p.catalogue, p.id)):
        nearest = result.best_direction(problem)
        if nearest is None or nearest[1] <= 0:
            lines.append(f"| `{problem.key}` {_cell(problem.title, 70)} | none | 0.00 | no shared vocabulary |")
            continue
        why = "under the floor" if nearest[1] < result.floor else "another problem claims it, or too close to call"
        lines.append(f"| `{problem.key}` {_cell(problem.title, 70)} "
                     f"| {_cell(', '.join(nearest[0].terms[:4]), 50)} | {nearest[1]:.2f} | {why} |")
    if not result.neglected:
        lines.append("| _none_ | | | |")

    outside = result.outside
    lines += ["", "## Outside the corpus", "",
              f"{len(outside)} unmatched problems are in fields this corpus does not cover. "
              "That is an absence of data, not a finding that nobody works on them.", ""]
    by_field = Counter(p.field for p in outside)
    for name, n in by_field.most_common():
        sample = [p.title for p in outside if p.field == name][:3]
        lines.append(f"- `{name}`: {n} ({'; '.join(_cell(s, 45) for s in sample)}"
                     f"{'; …' if n > len(sample) else ''})")

    lines += ["", f"## Orphan directions (largest {top})", "",
              "Active directions no catalogued problem claims. Either the catalogues have a "
              "gap, or the direction's vocabulary differs from the problem statement's.", "",
              "| Direction (distinctive terms) | Works | Active | Cohesion | Nearest problem | Similarity |",
              "| --- | --- | --- | --- | --- | --- |"]
    orphans = sorted(result.orphans, key=lambda d: d.size, reverse=True)
    for direction in orphans[:top]:
        j = result.directions.index(direction)
        column = result.similarity[:, j]
        nearest = "none | 0.00"
        if len(column) and float(column.max()) > 0:
            i = int(np.argmax(column))
            nearest = f"`{result.problems[i].key}` {_cell(result.problems[i].title, 40)} | {float(column[i]):.2f}"
        lines.append(f"| {_cell(', '.join(direction.terms[:6]), 60)} | {direction.size} "
                     f"| {_span(direction)} | {direction.cohesion:.2f} | {nearest} |")
    if not orphans:
        lines.append("| _none_ | | | | | |")
    if len(orphans) > top:
        lines.append(f"\n{len(orphans) - top} smaller orphan directions not shown.")

    ambiguous = sorted((r for r in result.refusals if r.ambiguous), key=lambda r: -r.similarity)
    lines += ["", "## Refused", "",
              f"{len(ambiguous)} direction-catalogue pairs cleared the floor but not the margin: "
              "the direction sits between two problems and was linked to neither. "
              f"{len(result.refusals) - len(ambiguous)} more fell just under the floor.", ""]
    for refusal in ambiguous[:8]:
        lines.append(f"- {_cell(', '.join(refusal.direction.terms[:4]), 50)} -> "
                     f"`{refusal.problem.key}` at {refusal.similarity:.2f}: {refusal.reason}")

    lines += ["", "## Limitations", ""]
    lines += [f"- {text}" for text in _limitations(synthetic)]
    return "\n".join(lines) + "\n"


def _render_evaluation(ev: dict) -> list[str]:
    def row(name: str, score: LeaveOneOut, orphans: bool = False) -> str:
        r = score.rates()
        precision = f"{score.links_correct}/{score.links} ({r['link precision']:.0%})" if score.links else "no links"
        orphan = (f"{score.own_directions_orphaned}/{score.own_directions} ({r['orphan detection']:.0%})"
                  if orphans else "not defined")
        return (f"| {name} | {score.problems} "
                f"| {score.linked_correctly}/{score.problems} ({r['match recall']:.0%}) "
                f"| {precision} "
                f"| {score.refused_when_absent}/{score.problems} ({r['neglect detection']:.0%}) "
                f"| {orphan} |")
    floors = sorted({f for f, _ in ev["best"]})
    margins = sorted({m for _, m in ev["best"]})
    best = (f"floor {', '.join(f'{f:g}' for f in floors)} at margin "
            f"{', '.join(f'{m:g}' for m in margins)}")
    return [
        "## How well the matcher works, measured",
        "",
        "The ai2035 frontier papers are labelled by the question whose search returned "
        "them, which gives a leave-one-out test of all three lists. For each problem with "
        "known papers: is it linked to a direction made mostly of those papers (match "
        "recall, and link precision over every link it gets); with the papers removed and "
        "the corpus re-clustered, does it come back unmatched (neglect detection); with "
        "the problem removed from the catalogue, do its papers' directions come back "
        "orphaned (orphan detection).",
        "",
        "| Test | Problems | Match recall | Link precision | Neglect detection | Orphan detection |",
        "| --- | --- | --- | --- | --- | --- |",
        row("ai2035 questions on their own search phrases (easy)", ev["own_words"], True),
        row("the same with every harvest-query word withheld", ev["leak_free"], True),
        row(f"AI-100 known pairs, tuning half ({ev['tune_ids'][0]}-{ev['tune_ids'][-1]})", ev["tune"]),
        row(f"**AI-100 known pairs, held-out half ({ev['test_ids'][0]}-{ev['test_ids'][-1]})**", ev["test"]),
        "",
        "How to read it:",
        "",
        "- The first row is the easy case: a question's matching text is nearly the query "
        "its papers were found with. It shows the plumbing works and little more.",
        f"- Withholding the query words leaves a median of {ev['leak_free_terms_left']:g} "
        "terms per question, so almost nothing matches. The matcher runs on a topic's "
        "name-words and has nothing else to go on.",
        "- The known pairs are the fairest test here: 13 AI-100 questions paired by hand "
        "(`data/nexus/matching/known_pairs.yaml`, `assumed`) with the ai2035 question on "
        "the same topic, matched on a gloss written separately from the harvest queries, "
        "against the other 99 AI-100 questions. Orphan detection is not defined for them: "
        "other AI-100 questions legitimately claim the same papers.",
        f"- The floor was chosen on the tuning half only, by the share of matched / "
        f"neglected verdicts it got right, ties broken by link precision (best there: "
        f"{best}; shipped: floor {SIMILARITY_FLOOR.value:g}, margin "
        f"{MARGIN_OVER_RUNNER_UP.value:g}). The tuning half cannot tell the margins apart.",
        "- **Neglect detection is the weak number.** When a problem's own literature is "
        "taken out, a neighbouring direction often still clears the floor and the problem "
        "is reported matched. So the neglected list below is an undercount, and a "
        "\"matched\" verdict deserves a look at the direction's terms.",
        "- Every figure is `observed` on this corpus, and the samples are tiny: 20, 6 and 7 "
        "problems.",
        "",
    ]


def _limitations(synthetic: bool) -> list[str]:
    out = [
        "Matching is lexical. A problem stated in different words from the papers working "
        "on it is reported neglected, and a direction sharing a method's vocabulary with a "
        "problem can be linked to it wrongly. Read a neglected verdict as \"no direction in "
        "this corpus uses this problem's words\", not \"nobody is working on it\".",
        "The default corpus is not a sample of machine learning. It was harvested with "
        "search queries for the 20 ai2035 questions, so an AI-100 or Human Frontier "
        "problem with no direction may simply be one nobody queried for. A corpus "
        "harvested without reference to any catalogue is what a real neglect claim needs.",
        "The AI-100 catalogue is in Chinese. Its English rendering and matching terms "
        "(`data/nexus/matching/ai100_en.yaml`) were written by the model, not taken from "
        "the source and not reviewed; a poor gloss produces a false neglected verdict.",
        "The measured scores come from one corpus in one field, whose papers were found by "
        "lexical search in the first place. The matcher has never been tested on cosmology, "
        "mathematics or global-development literature.",
        "The floor and margin are judgement calls; the floor was tuned on six questions. Which fields a "
        "corpus covers is declared by hand; a vocabulary-overlap test was tried and could "
        "not tell cosmology from machine learning.",
        "Catalogues differ in how much text they give: Human Frontier challenges are "
        "matched on a title alone, worldsim problems on a name and outcome labels. Short "
        "text matches badly.",
        "Science's 125 questions, discussed elsewhere in this project, are not in the "
        "repository and are not matched.",
        "\"Active\" means present in the corpus. No growth rate is computed here; run "
        "`sciscout rank` on the same directions for momentum and priority.",
    ]
    if synthetic:
        out.insert(0, "The corpus is synthetic, so every verdict above is about the "
                      "generator's planted vocabulary, not about real research.")
    return out


# -- command line ------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"error: corpus not found: {corpus_path}")
        return 2
    works, note = load_corpus(corpus_path)
    if not works:
        print(f"error: no works loaded from {corpus_path}")
        return 2
    synthetic = all(w.id.startswith("synthetic:") for w in works)
    is_frontier = corpus_path.resolve() == FRONTIER_DIR.resolve()
    threshold = args.threshold
    if threshold is None:
        threshold = 0.16 if synthetic else REAL_ABSTRACT_THRESHOLD.value

    fields_note: Estimate | None = None
    corpus_fields: frozenset[str] | None = None
    if args.field:
        corpus_fields = frozenset(args.field)
        fields_note = Estimate(corpus_fields, assumed("declared on the command line"),
                               label="fields the corpus covers")
    elif is_frontier:
        fields_note = FRONTIER_FIELDS
        corpus_fields = FRONTIER_FIELDS.value

    problems = load_catalogues(only=args.catalogue or None)
    if not problems:
        print("error: no problem catalogues found")
        return 2
    try:
        found = discover(works, threshold=threshold, min_cluster_size=args.min_cluster_size)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    result = match(problems, found, works, args.floor, args.margin, corpus_fields)

    evaluation = None
    labelled = load_catalogues(only=["ai2035"])
    has_labels = any("question" in w.extra for w in works)
    if args.evaluate and has_labels and labelled:
        evaluation = tune_and_test(works, threshold=threshold)

    text = render(result, note, found, evaluation, synthetic=synthetic, top=args.top,
                  fields_note=fields_note)
    counts = result.counts()
    print(note)
    print(found.summary())
    print(f"{len(problems)} problems: {counts['matched'].value} matched, "
          f"{counts['neglected'].value} neglected, {counts['outside'].value} outside the corpus; "
          f"{counts['orphans'].value} of {len(result.directions)} directions orphaned")
    if evaluation:
        held = evaluation["test"]
        rates = held.rates()
        print(f"held-out known pairs ({held.problems} problems): match recall "
              f"{rates['match recall']:.0%}, link precision {rates['link precision']:.0%}, "
              f"neglect detection {rates['neglect detection']:.0%}")
    elif args.evaluate:
        print("matcher not measured: this corpus carries no per-question labels")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"report written to {args.out}")
    else:
        print()
        print(text)
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "matching",
        help="match open-problem catalogues against discovered research directions",
        description="Report matched, neglected and out-of-corpus problems, and orphan directions.",
    )
    parser.add_argument("--corpus", default=str(FRONTIER_DIR),
                        help="a corpus JSON file, or the ai2035 frontier directory (default: "
                             "the real arXiv abstracts in ai2035/frontier)")
    parser.add_argument("--catalogue", action="append", choices=sorted(CATALOGUES),
                        help="limit to one catalogue; repeatable (default: all)")
    parser.add_argument("--field", action="append",
                        help="a field the corpus covers (ai, cosmology, mathematics, global, "
                             "...); repeatable. Without it only the default corpus has a "
                             "declared scope, and nothing else is called neglected")
    parser.add_argument("--threshold", type=float, default=None,
                        help="discovery similarity floor (default 0.10 real, 0.16 synthetic)")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--floor", type=float, default=None, help="match similarity floor")
    parser.add_argument("--margin", type=float, default=None, help="margin over the runner-up")
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                        help="skip the leave-one-out measurement (about a minute)")
    parser.add_argument("--top", type=int, default=25, help="orphan directions to list")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="where to write the markdown report ('' prints it instead)")
    parser.set_defaults(func=run, evaluate=True)
