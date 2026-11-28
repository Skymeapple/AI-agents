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

Two failure modes are kept apart because they mean opposite things. A problem
whose words barely occur in the corpus is *outside* it -- the corpus says nothing
about dark matter if it holds only machine-learning papers -- and calling that
problem neglected would be reading an absence of data as a finding. Only a
problem the corpus does talk about, with no direction on it, is neglected.

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
# Every constant here is a judgement call and says so. The floor and the margin
# were picked on the tuning half of the leave-one-out experiment (ai2035 Q01-Q10)
# and the report prints how they do on the half they never saw (Q11-Q20).

SIMILARITY_FLOOR: Estimate[float] = Estimate(
    0.12,
    assumed(
        "cosine below which a problem and a direction share too little vocabulary "
        "to call a match; chosen on the tuning half (ai2035 Q01-Q10) of the "
        "leave-one-out experiment, not derived"
    ),
    label="similarity floor",
)
MARGIN_OVER_RUNNER_UP: Estimate[float] = Estimate(
    1.25,
    assumed(
        "how far the best problem of a catalogue must beat the runner-up before a "
        "direction is linked to it; chosen on the tuning half (ai2035 Q01-Q10). "
        "Lower than sciscout.classify's 1.5 because neighbouring problems in one "
        "catalogue overlap far more than commercial sectors do"
    ),
    label="margin over runner-up",
)
COVERAGE_FLOOR: Estimate[float] = Estimate(
    0.5,
    assumed(
        "share of a problem's distinctive vocabulary that must occur in the corpus "
        "before an unmatched problem is called neglected rather than outside the "
        "corpus; the report prints each catalogue's observed coverage beside it"
    ),
    label="coverage floor",
)
DISTINCTIVE_DF_RATIO: Estimate[float] = Estimate(
    0.02,
    assumed(
        "a term used by more than this share of the corpus is treated as generic "
        "('model', 'problem') and ignored when asking whether the corpus covers a "
        "problem's vocabulary"
    ),
    label="distinctive document-frequency ratio",
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
TUNE_FLOORS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
TUNE_MARGINS = (1.0, 1.1, 1.25, 1.5, 2.0)

_TERM_SPLIT = re.compile(r"[,;]")


# -- problems --------------------------------------------------------------------


@dataclass
class Problem:
    """One catalogued open problem, with the English text it is matched on."""

    id: str
    catalogue: str
    title: str
    text: str
    provenance: Provenance
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
        problems.append(Problem(f"{int(row['id']):03d}", "ai100", entry["en"], text, provenance))
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
            ". ".join(phrases), provenance, withheld,
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
            sourced("module docstring", reference=f"cosmos/{path.name}"),
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
                                    f"{cells[1]}. {cells[2]}", provenance))
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
        Problem(f"w{i:02d}", "world", name, f"{name}. " + ". ".join(labels), provenance)
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

    def coverage(self, terms: list[str], distinctive_ratio: float) -> float | None:
        """Share of a problem's distinctive single words the corpus uses at all.

        Words used by more than ``distinctive_ratio`` of the corpus are generic
        and say nothing about whether the corpus covers the problem. ``None``
        when the problem has no distinctive words to judge by.
        """
        words = {t for t in terms if " " not in t}
        distinctive = [
            t for t in words
            if t not in self.index or self.df[self.index[t]] <= distinctive_ratio * self.n_docs
        ]
        if not distinctive:
            return None
        return sum(t in self.index for t in distinctive) / len(distinctive)


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


@dataclass
class MatchResult:
    problems: list[Problem]
    directions: list[DiscoveredTrack]
    similarity: np.ndarray  # problems x directions
    coverage: dict[str, float | None]  # problem key -> corpus coverage
    links: list[Link]
    refusals: list[Refusal]
    floor: float
    margin: float
    coverage_floor: float

    def links_for(self, problem: Problem) -> list[Link]:
        return [l for l in self.links if l.problem is problem]

    @property
    def matched(self) -> list[Problem]:
        seen = {id(l.problem) for l in self.links}
        return [p for p in self.problems if id(p) in seen]

    def _unmatched(self) -> list[Problem]:
        seen = {id(l.problem) for l in self.links}
        return [p for p in self.problems if id(p) not in seen]

    def _covered(self, problem: Problem) -> bool:
        share = self.coverage[problem.key]
        return share is not None and share >= self.coverage_floor

    @property
    def neglected(self) -> list[Problem]:
        """Problems the corpus talks about with no direction on them."""
        return [p for p in self._unmatched() if self._covered(p)]

    @property
    def outside(self) -> list[Problem]:
        """Unmatched problems the corpus cannot speak to either way."""
        return [p for p in self._unmatched() if not self._covered(p)]

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
            "neglected": obs(len(self.neglected), "problems covered by the corpus with no direction"),
            "outside": obs(len(self.outside), "problems outside the corpus"),
            "orphans": obs(len(self.orphans), "directions tied to no problem"),
            "links": obs(len(self.links), "problem-direction links"),
        }


def match(
    problems: list[Problem],
    found: DiscoveryResult,
    works: list[Work],
    floor: float | None = None,
    margin: float | None = None,
    coverage_floor: float | None = None,
    leak_free: bool = False,
) -> MatchResult:
    """Link problems to the directions in ``found``.

    A link (problem, direction) is made when the problem is the direction's best
    match within its catalogue, clears ``floor``, and beats that catalogue's
    runner-up by ``margin``. A direction may therefore be linked once per
    catalogue -- catalogues overlap, and "continual learning" is rightly both an
    AI-100 and an ai2035 problem -- but never to two problems of one catalogue.

    ``works`` must be the corpus ``found`` was discovered in; it supplies the
    vocabulary and document frequencies. ``leak_free`` drops each problem's
    ``withheld`` tokens, for scoring against papers harvested with them.
    """
    floor = SIMILARITY_FLOOR.value if floor is None else floor
    margin = MARGIN_OVER_RUNNER_UP.value if margin is None else margin
    coverage_floor = COVERAGE_FLOOR.value if coverage_floor is None else coverage_floor

    space = _Space.build(works)
    row_of = {id(w): i for i, w in enumerate(works)}
    directions = list(found.tracks)
    centroids = (
        np.stack([space.centroid([row_of[id(w)] for w in d.works]) for d in directions])
        if directions else np.zeros((0, len(space.index)), dtype=np.float32)
    )
    term_lists = [_problem_terms(p, leak_free) for p in problems]
    vectors = (
        np.stack([space.problem_vector(t) for t in term_lists])
        if problems else np.zeros((0, len(space.index)), dtype=np.float32)
    )
    similarity = vectors @ centroids.T
    coverage = {
        p.key: space.coverage(t, DISTINCTIVE_DF_RATIO.value) for p, t in zip(problems, term_lists)
    }

    links: list[Link] = []
    refusals: list[Refusal] = []
    by_catalogue: dict[str, list[int]] = {}
    for i, p in enumerate(problems):
        by_catalogue.setdefault(p.catalogue, []).append(i)
    for j, direction in enumerate(directions):
        for members in by_catalogue.values():
            scores = sorted(((float(similarity[i, j]), i) for i in members), reverse=True)
            best, i = scores[0]
            runner_up = scores[1][0] if len(scores) > 1 else 0.0
            if best <= 0:
                continue
            ratio = best / runner_up if runner_up > 0 else float("inf")
            if best < floor:
                # Only near misses are worth a reader's time.
                if best >= 0.75 * floor:
                    refusals.append(Refusal(problems[i], direction, best, runner_up,
                                            f"similarity {best:.2f} is under the floor {floor:g}"))
            elif ratio < margin:
                refusals.append(Refusal(
                    problems[i], direction, best, runner_up,
                    f"only {ratio:.2f}x the runner-up "
                    f"({problems[scores[1][1]].key}), needing {margin:g}x"))
            else:
                links.append(Link(problems[i], direction, best, runner_up))
    return MatchResult(problems, directions, similarity, coverage, links, refusals,
                       floor, margin, coverage_floor)


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
    """Scores for the three claims, on one set of labelled problems."""

    problems: int
    # Present: problem and its papers both in. Did it get a link, and a right one?
    matched_present: int
    linked_correctly: int
    links: int
    links_correct: int
    # Papers removed: the problem should be reported neglected (or outside).
    refused_when_absent: int
    # Problem removed: its papers' directions should be orphans.
    own_directions: int
    own_directions_orphaned: int

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
    def balanced(self) -> float:
        values = [v for v in self.rates().values() if not math.isnan(v)]
        return sum(values) / len(values) if values else 0.0


def evaluate_leave_one_out(
    problems: list[Problem],
    works: list[Work],
    label_key: str = "question",
    threshold: float | None = None,
    floor: float | None = None,
    margin: float | None = None,
    leak_free: bool = True,
    min_purity: float = 0.5,
    _cache: dict | None = None,
) -> LeaveOneOut:
    """Test the matcher on a corpus whose works are labelled by problem id.

    For each problem in turn: (1) with everything present, is it linked to a
    direction whose works are mostly its own; (2) with its papers removed and the
    corpus re-clustered, does it come back unmatched; (3) with the problem
    removed from the catalogue, do the directions made of its papers come back
    orphaned. ``leak_free`` matches without the words the papers were harvested
    with, so the score is not just the search query finding itself.

    A direction counts as a problem's own when at least ``min_purity`` of its
    works carry that label.
    """
    threshold = REAL_ABSTRACT_THRESHOLD.value if threshold is None else threshold
    cache = _cache if _cache is not None else {}

    def clustered(excluded: str | None) -> tuple[list[Work], DiscoveryResult]:
        key = (excluded, threshold)
        if key not in cache:
            kept = [w for w in works if excluded is None or excluded not in w.extra.get(label_key, [])]
            cache[key] = (kept, discover(kept, threshold=threshold))
        return cache[key]

    all_works, full = clustered(None)
    own = {id(d): _majority_label(d, label_key) for d in full.tracks}
    present = match(problems, full, all_works, floor, margin, leak_free=leak_free)

    links_correct = sum(
        1 for l in present.links
        if own[id(l.direction)][0] == l.problem.id and own[id(l.direction)][1] >= min_purity
    )
    linked_correctly = sum(
        1 for p in problems
        if any(own[id(l.direction)][0] == p.id and own[id(l.direction)][1] >= min_purity
               for l in present.links_for(p))
    )

    refused = 0
    own_total = own_orphaned = 0
    for p in problems:
        kept, found = clustered(p.id)
        absent = match(problems, found, kept, floor, margin, leak_free=leak_free)
        refused += not absent.links_for(p)

        others = [q for q in problems if q is not p]
        without = match(others, full, all_works, floor, margin, leak_free=leak_free)
        orphan_ids = {id(d) for d in without.orphans}
        mine = [d for d in full.tracks if own[id(d)][0] == p.id and own[id(d)][1] >= min_purity]
        own_total += len(mine)
        own_orphaned += sum(id(d) in orphan_ids for d in mine)

    return LeaveOneOut(
        problems=len(problems), matched_present=len(present.matched),
        linked_correctly=linked_correctly, links=len(present.links), links_correct=links_correct,
        refused_when_absent=refused, own_directions=own_total, own_directions_orphaned=own_orphaned,
    )


def tune_and_test(
    problems: list[Problem], works: list[Work], label_key: str = "question",
    threshold: float | None = None,
) -> dict:
    """Pick the floor and margin on half the labelled problems, score the other half.

    The split is by problem id order (first half tunes), fixed, so the held-out
    half never influences the rule it is scored with.
    """
    ordered = sorted(problems, key=lambda p: p.id)
    half = len(ordered) // 2
    tune, test = ordered[:half], ordered[half:]
    cache: dict = {}
    grid = []
    for floor in TUNE_FLOORS:
        for margin in TUNE_MARGINS:
            score = evaluate_leave_one_out(tune, works, label_key, threshold, floor, margin,
                                           _cache=cache)
            grid.append((score.balanced, floor, margin, score))
    best = max(grid, key=lambda g: (round(g[0], 6), g[1], g[2]))
    shipped = (SIMILARITY_FLOOR.value, MARGIN_OVER_RUNNER_UP.value)
    return {
        "tune_ids": [p.id for p in tune],
        "test_ids": [p.id for p in test],
        "grid": [(b, f, m) for b, f, m, _ in grid],
        "best": (best[1], best[2]),
        "tune": evaluate_leave_one_out(tune, works, label_key, threshold, *shipped, _cache=cache),
        "test": evaluate_leave_one_out(test, works, label_key, threshold, *shipped, _cache=cache),
        "test_leaky": evaluate_leave_one_out(test, works, label_key, threshold, *shipped,
                                             leak_free=False, _cache=cache),
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
           evaluation: dict | None = None, synthetic: bool = False, top: int = 25) -> str:
    counts = result.counts()
    lines = [
        "# Open problems against active research directions",
        "",
        f"Corpus: {corpus_note}.",
        f"Discovery: {found.summary()} (similarity floor {found.threshold:g}).",
        f"Catalogues: {len(result.problems)} problems in "
        f"{len({p.catalogue for p in result.problems})} catalogues.",
        "",
    ]
    if synthetic:
        lines += [
            "**This corpus is synthetic.** Its directions were planted by a generator, so a "
            "match here shows the matcher can connect a problem to a direction with the "
            "right vocabulary, and a neglected or orphan verdict shows the refusal rule "
            "working. Neither says anything about what real research groups are doing.",
            "",
        ]
    lines += [
        "| Result | Count |", "| --- | --- |",
        f"| Problems matched to at least one direction | {counts['matched'].value} |",
        f"| Problems neglected (corpus covers the vocabulary, no direction on it) | {counts['neglected'].value} |",
        f"| Problems outside the corpus (cannot be judged from it) | {counts['outside'].value} |",
        f"| Directions orphaned (tied to no catalogued problem) | {counts['orphans'].value} of {len(result.directions)} |",
        "",
        "All four counts are `observed`: they are computed from this corpus by the rule "
        "below. The rule's three constants are `assumed`.",
        "",
        "## The rule",
        "",
    ]
    used = (
        (SIMILARITY_FLOOR, result.floor),
        (MARGIN_OVER_RUNNER_UP, result.margin),
        (COVERAGE_FLOOR, result.coverage_floor),
        (DISTINCTIVE_DF_RATIO, DISTINCTIVE_DF_RATIO.value),
    )
    for estimate, value in used:
        note = "" if value == estimate.value else f" (overridden from {estimate.value:g} for this run)"
        lines.append(f"- {estimate.label} = {value:g}{note}  <- {estimate.provenance.describe()}")
    lines += [
        "",
        "A direction is linked to a problem when the problem is its best match within "
        "its catalogue, clears the floor, and beats that catalogue's runner-up by the "
        "margin. An unmatched problem is *neglected* only if the corpus uses at least "
        "the coverage floor's share of its distinctive words; otherwise it is *outside* "
        "the corpus and nothing is concluded.",
        "",
        "## By catalogue",
        "",
        "| Catalogue | Problems | Matched | Neglected | Outside corpus | Median vocabulary coverage | Matching text |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    matched_ids = {id(p) for p in result.matched}
    neglected_ids = {id(p) for p in result.neglected}
    for name, (title, _) in CATALOGUES.items():
        members = [p for p in result.problems if p.catalogue == name]
        if not members:
            continue
        shares = [result.coverage[p.key] for p in members if result.coverage[p.key] is not None]
        median = f"{float(np.median(shares)):.0%}" if shares else "n/a"
        n_matched = sum(id(p) in matched_ids for p in members)
        n_neglected = sum(id(p) in neglected_ids for p in members)
        lines.append(
            f"| {title} (`{name}`) | {len(members)} | {n_matched} | {n_neglected} "
            f"| {len(members) - n_matched - n_neglected} | {median} "
            f"| {members[0].provenance.grade.value}: {_cell(members[0].provenance.detail, 60)} |"
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
              "The corpus uses these problems' vocabulary, but no direction is plainly about "
              "them. The nearest direction is shown so the verdict can be checked.", "",
              "| Problem | Coverage | Nearest direction | Similarity |", "| --- | --- | --- | --- |"]
    for problem in sorted(result.neglected, key=lambda p: (p.catalogue, p.id)):
        nearest = result.best_direction(problem)
        near = f"{_cell(', '.join(nearest[0].terms[:4]), 50)} | {nearest[1]:.2f}" if nearest else "none | 0.00"
        lines.append(f"| `{problem.key}` {_cell(problem.title, 70)} "
                     f"| {result.coverage[problem.key]:.0%} | {near} |")
    if not result.neglected:
        lines.append("| _none_ | | | |")

    outside = result.outside
    lines += ["", "## Outside the corpus", "",
              f"{len(outside)} problems are unmatched and the corpus uses under "
              f"{result.coverage_floor:.0%} of their distinctive vocabulary. This is an absence "
              "of data, not a finding that nobody works on them.", ""]
    by_cat = Counter(p.catalogue for p in outside)
    for name, n in by_cat.most_common():
        sample = [p.title for p in outside if p.catalogue == name][:4]
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

    ambiguous = [r for r in result.refusals if "runner-up" in r.reason]
    lines += ["", "## Refused", "",
              f"{len(ambiguous)} direction-catalogue pairs cleared the floor but not the margin: "
              "the direction sits between two problems and was linked to neither. "
              f"{len(result.refusals) - len(ambiguous)} more fell just under the floor.", ""]
    for refusal in sorted(ambiguous, key=lambda r: -r.similarity)[:8]:
        lines.append(f"- {_cell(', '.join(refusal.direction.terms[:4]), 50)} -> "
                     f"`{refusal.problem.key}` at {refusal.similarity:.2f}: {refusal.reason}")

    lines += ["", "## Limitations", ""]
    lines += [f"- {text}" for text in _limitations(synthetic)]
    return "\n".join(lines) + "\n"


def _render_evaluation(ev: dict) -> list[str]:
    def row(name: str, score: LeaveOneOut) -> str:
        r = score.rates()
        return (f"| {name} | {score.problems} "
                f"| {score.linked_correctly}/{score.problems} ({r['match recall']:.0%}) "
                f"| {score.links_correct}/{score.links} ({r['link precision']:.0%}) "
                f"| {score.refused_when_absent}/{score.problems} ({r['neglect detection']:.0%}) "
                f"| {score.own_directions_orphaned}/{score.own_directions} ({r['orphan detection']:.0%}) |")
    best_floor, best_margin = ev["best"]
    return [
        "## How well the matcher works, measured",
        "",
        "The ai2035 frontier papers are labelled by the question whose search returned "
        "them, which gives a leave-one-out test of all three lists. For each question: "
        "is it linked to a direction made mostly of its own papers; with its papers "
        "removed and the corpus re-clustered, does it come back unmatched; with the "
        "question removed, do its papers' directions come back orphaned.",
        "",
        "| Split | Questions | Match recall | Link precision | Neglect detection | Orphan detection |",
        "| --- | --- | --- | --- | --- | --- |",
        row(f"Tuning half ({ev['tune_ids'][0]}-{ev['tune_ids'][-1]}), leak-free", ev["tune"]),
        row(f"**Held-out half ({ev['test_ids'][0]}-{ev['test_ids'][-1]}), leak-free**", ev["test"]),
        row("Held-out half, with the harvest words left in", ev["test_leaky"]),
        "",
        f"The floor and margin were chosen on the tuning half only (grid optimum there: "
        f"floor {best_floor:g}, margin {best_margin:g}). \"Leak-free\" "
        "matches each question without any word from the search queries its papers were "
        "harvested with, so a link has to be made on the rest of its vocabulary; the last "
        "row shows how much easier the test is when the query is allowed to find itself. "
        "All figures are `observed` on this corpus and are small-sample: ten questions a "
        "half.",
        "",
    ]


def _limitations(synthetic: bool) -> list[str]:
    out = [
        "Matching is lexical. A problem stated in different words from the papers working "
        "on it is reported neglected, and a direction sharing a method's vocabulary with a "
        "problem can be linked to it wrongly. Read a neglected verdict as \"no direction "
        "uses this problem's words\", not \"nobody is working on it\".",
        "The AI-100 catalogue is in Chinese. Its English rendering and matching terms "
        "(`data/nexus/matching/ai100_en.yaml`) were written by the model, not taken from "
        "the source and not reviewed; a poor gloss produces a false neglected verdict.",
        "The measured scores come from one corpus in one field (machine learning), whose "
        "papers were found by lexical search in the first place. They say nothing about "
        "how the matcher does on cosmology or mathematics, where it has never been tested.",
        "The floor, margin and coverage floor are judgement calls tuned on ten questions.",
        "Catalogue coverage is uneven: Human Frontier challenges are matched on a title "
        "alone, worldsim problems on a name and outcome labels. Short text matches badly.",
        "Science's 125 questions, discussed elsewhere in this project, are not in the "
        "repository and are not matched.",
        "\"Active\" means present in the corpus. No growth rate is computed here; use "
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
    threshold = args.threshold
    if threshold is None:
        threshold = 0.16 if synthetic else REAL_ABSTRACT_THRESHOLD.value
    problems = load_catalogues(only=args.catalogue or None)
    if not problems:
        print("error: no problem catalogues found")
        return 2
    try:
        found = discover(works, threshold=threshold, min_cluster_size=args.min_cluster_size)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    result = match(problems, found, works, args.floor, args.margin, args.coverage_floor)

    evaluation = None
    labelled = [p for p in load_catalogues(only=["ai2035"])]
    has_labels = any("question" in w.extra for w in works)
    if args.evaluate and has_labels and labelled:
        evaluation = tune_and_test(labelled, works, threshold=threshold)
    elif args.evaluate and not has_labels:
        print("note: --evaluate needs works labelled by question; skipped for this corpus")

    text = render(result, note, found, evaluation, synthetic=synthetic, top=args.top)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    counts = result.counts()
    print(f"{note}")
    print(f"{found.summary()}")
    print(f"{len(problems)} problems: {counts['matched'].value} matched, "
          f"{counts['neglected'].value} neglected, {counts['outside'].value} outside the corpus; "
          f"{counts['orphans'].value} of {len(result.directions)} directions orphaned")
    if evaluation:
        for name, value in evaluation["test"].rates().items():
            print(f"held-out {name}: {value:.0%}")
    if args.out:
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
    parser.add_argument("--threshold", type=float, default=None,
                        help="discovery similarity floor (default 0.10 real, 0.16 synthetic)")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--floor", type=float, default=None, help="match similarity floor")
    parser.add_argument("--margin", type=float, default=None, help="margin over the runner-up")
    parser.add_argument("--coverage-floor", type=float, default=None)
    parser.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                        help="skip the leave-one-out measurement (about a minute)")
    parser.add_argument("--top", type=int, default=25, help="orphan directions to list")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="where to write the markdown report ('' prints it instead)")
    parser.set_defaults(func=run, evaluate=True)
