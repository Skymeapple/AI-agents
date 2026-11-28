"""The closed discovery loop: read, rank, verify, and let the result steer.

Three projects in this repository each do one step of a research process and
stop. autoscout reads the literature. sciscout clusters what was read into
directions and ranks them. aisci-verifiers checks whether a claim holds. None
of them hears from the others, so sciscout's ranking never learns that a
direction's claims keep failing, and autoscout keeps reading what its seed
agenda told it to.

This module connects them::

    corpus     a sciscout corpus, or the papers autoscout has flagged
    rank       sciscout discovers directions and scores their priority
    allocate   a checking budget is spread over directions by Thompson
               sampling on  priority x (share of the direction's claims that
               are true)
    verify     each chosen claim is run through the aisci verifier service
    learn      the outcome updates the direction's yield, discounted by how
               often that verifier is wrong
    steer      directions are re-ranked by verified priority, and the top of
               that ranking is written out as an agenda autoscout can read from

Two things are done carefully because the easy version is wrong.

*A pass is evidence, not truth.* ``aisci.calibration`` measured each
verifier's error rates, and they differ enormously: the tensor check passed 0
of 40 wrong algorithms, the weak test suite passed 6 of 8 wrong programs. So a
direction's yield is not its pass rate. Each outcome enters through the
likelihood ``P(pass) = yield * (1 - false_reject) + (1 - yield) * false_pass``,
which makes a pass from a weak verifier nearly uninformative and a verifier
with no calibration exactly uninformative. The experiment below shows what
happens without this: the loop is captured by whichever direction is checked
most leniently.

*Every check is recorded, including the failures.* Outcomes go to the
hash-chained store from ``aisci.datastore``, and claims that pass go into the
``aisci.worldmodel`` with the verifier run as their evidence. A loop that kept
only its successes could not be audited and would overstate every yield.

What is not here is the step that turns a real abstract into a checkable
claim. That needs a model, and the checkers execute what they are given, so it
also needs a sandbox. Offline, claims come from a synthetic bank
(``scripts/make_claim_bank.py``) whose candidates are real verifier payloads
but whose attachment to papers is planted. The loop is measured on it; nothing
here says how it does on real claims.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sciscout.classify import classify
from sciscout.discovery import discover
from sciscout.models import Work
from sciscout.pipeline import Pipeline
from sciscout.provenance import Estimate, ProvenanceLedger, assumed, observed
from sciscout.sources.local import LocalCorpus

ROOT = Path(__file__).resolve().parents[1]
AISCI_PATH = ROOT / "aisci-verifiers"
DEFAULT_CORPUS = ROOT / "data" / "demo_corpus.json"
DEFAULT_BANK = ROOT / "data" / "nexus" / "claim_bank.json"
DEFAULT_CALIBRATION = AISCI_PATH / "results" / "i1_calibration.json"
DEFAULT_STATE = Path("nexus_runs") / "loop"
DEFAULT_ASOF = dt.date(2026, 11, 28)

GRID = np.linspace(0.0, 1.0, 201)
POLICIES = ("loop", "naive", "priority", "uniform", "oracle")

# arXiv categories autoscout should harvest for a sector when it reads from an
# agenda this loop wrote. Only live harvesting uses them.
SECTOR_CATEGORIES: dict[str, Estimate[list[str]]] = {
    sector: Estimate(cats, assumed("arXiv categories judged closest to the sector"), label=sector)
    for sector, cats in {
        "energy_hardware": ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph"],
        "biopharma": ["q-bio.BM", "q-bio.QM", "q-bio.GN"],
        "semiconductors": ["quant-ph", "cond-mat.mes-hall", "physics.optics"],
        "software_ai": ["cs.LG", "cs.AI", "stat.ML"],
        "agriculture_food": ["q-bio.GN", "q-bio.PE"],
    }.items()
}


def _aisci():
    """Import the verifier suite, which lives beside the package rather than in it."""
    if str(AISCI_PATH) not in sys.path:
        sys.path.insert(0, str(AISCI_PATH))
    try:
        from aisci import datastore, verifiers, worldmodel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"nexus loop needs the aisci verifiers, which need {exc.name}; "
            "install with: pip install sympy scipy jsonschema"
        ) from exc
    return verifiers, datastore, worldmodel


# -- claims --------------------------------------------------------------------


@dataclass
class Claim:
    """Something a paper asserts that a verifier can check."""

    id: str
    work: str  # id of the work the claim comes from
    verifier: str
    statement: str
    candidate: Any
    # Which calibration regime the check runs under, e.g. "[weak tests]".
    variant: str = ""
    # Known only for synthetic claims. Scoring reads it; no decision does.
    truth: bool | None = None

    @property
    def regime(self) -> str:
        return self.verifier + self.variant


class ClaimSource:
    """Where checkable claims come from."""

    name = "source"

    def claims_for(self, work_ids: set[str]) -> list[Claim]:
        raise NotImplementedError


class BankSource(ClaimSource):
    """Replays a claim bank from disk, in file order."""

    name = "bank"

    def __init__(self, path: Path = DEFAULT_BANK) -> None:
        raw = json.loads(Path(path).read_text())
        self.note = raw.get("note", "")
        self.planted_yield: dict[str, float] = raw.get("planted_yield", {})
        self.traps: list[str] = raw.get("traps", [])
        self.claims = [
            Claim(id=c["id"], work=c["work"], verifier=c["verifier"],
                  statement=c["statement"], candidate=c["candidate"],
                  variant=c.get("variant", ""), truth=c.get("truth"))
            for c in raw["claims"]
        ]

    def claims_for(self, work_ids: set[str]) -> list[Claim]:
        return [c for c in self.claims if c.work in work_ids]


# -- verifier error rates ------------------------------------------------------


@dataclass(frozen=True)
class ErrorRates:
    """How often a verifier is wrong, in each direction."""

    false_pass: Estimate[float]
    false_reject: Estimate[float]

    def p_pass(self, yield_: np.ndarray) -> np.ndarray:
        """Probability a check passes when ``yield_`` of the claims are true."""
        fp, fr = self.false_pass.value, self.false_reject.value
        return yield_ * (1.0 - fr) + (1.0 - yield_) * fp


def _rate(errors: int, trials: int, what: str) -> Estimate[float]:
    """Posterior mean of an error rate under a uniform prior.

    The calibration sets are small -- one verifier has a single known-correct
    case -- so the raw rate would claim a certainty the data cannot support: 0
    errors in 1 trial is not a zero error rate. The uniform prior is a
    judgement; the counts are measured.
    """
    return Estimate(
        (errors + 1) / (trials + 2),
        observed(f"{errors} of {trials} {what} in aisci calibration, "
                 "posterior mean under a uniform prior"),
        label=what,
    )


UNCALIBRATED = ErrorRates(
    Estimate(0.5, assumed("verifier has no calibration; treated as a coin flip"),
             label="false pass"),
    Estimate(0.5, assumed("verifier has no calibration; treated as a coin flip"),
             label="false reject"),
)


def load_error_rates(path: Path = DEFAULT_CALIBRATION) -> dict[str, ErrorRates]:
    """Error rates per verifier regime, from the aisci calibration results."""
    raw = json.loads(Path(path).read_text())["calibration"]
    return {
        regime: ErrorRates(
            _rate(row["false_pass"], row["n_wrong"], "wrong candidates passed"),
            _rate(row["false_reject"], row["n_right"], "correct candidates rejected"),
        )
        for regime, row in raw.items()
    }


# -- belief about a direction's yield ------------------------------------------


@dataclass
class Check:
    claim: str
    work: str
    regime: str
    passed: bool
    reason: str
    cycle: int
    truth: bool | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class YieldBelief:
    """Posterior over the share of a direction's claims that are true.

    Held on a grid under a uniform prior, because the likelihood of a pass
    depends on the verifier that produced it and no conjugate form covers a
    mix of verifiers with different error rates.
    """

    log_posterior: np.ndarray = field(default_factory=lambda: np.zeros_like(GRID))
    checks: int = 0
    passes: int = 0

    def update(self, passed: bool, rates: ErrorRates) -> None:
        p = np.clip(rates.p_pass(GRID), 1e-12, 1 - 1e-12)
        self.log_posterior = self.log_posterior + np.log(p if passed else 1.0 - p)
        self.checks += 1
        self.passes += int(passed)

    @property
    def weights(self) -> np.ndarray:
        w = np.exp(self.log_posterior - self.log_posterior.max())
        return w / w.sum()

    @property
    def mean(self) -> float:
        return float(self.weights @ GRID)

    def interval(self, level: float = 0.90) -> tuple[float, float]:
        cdf = np.cumsum(self.weights)
        lo = GRID[min(int(np.searchsorted(cdf, (1 - level) / 2)), len(GRID) - 1)]
        hi = GRID[min(int(np.searchsorted(cdf, 1 - (1 - level) / 2)), len(GRID) - 1)]
        return float(lo), float(hi)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.choice(GRID, p=self.weights))

    def estimate(self, label: str) -> Estimate[float]:
        if not self.checks:
            return Estimate(self.mean, assumed("no claims checked yet; uniform prior"),
                            label=label)
        return Estimate(
            self.mean,
            observed(f"{self.passes} of {self.checks} checks passed, weighted by "
                     "each verifier's measured error rates"),
            label=label,
        )


# -- directions ----------------------------------------------------------------


@dataclass
class Direction:
    """A discovered research direction with its priority and its claims."""

    id: str
    label: str
    sector: str
    terms: list[str]
    works: set[str]
    priority: float
    confidence: float  # share of the priority that rests on evidence
    pool: list[Claim] = field(default_factory=list)
    belief: YieldBelief = field(default_factory=YieldBelief)

    @property
    def verified_priority(self) -> float:
        return self.priority * self.belief.mean

    def verified_interval(self) -> tuple[float, float]:
        lo, hi = self.belief.interval()
        return self.priority * lo, self.priority * hi

    def true_yield(self) -> float | None:
        """Share of this direction's claims that are true, where that is known."""
        known = [c.truth for c in self.pool if c.truth is not None]
        return sum(known) / len(known) if known else None


def load_works(corpus: Path | None = None, autoscout_state: Path | None = None) -> list[Work]:
    """The corpus the loop ranks: a sciscout corpus file, or autoscout's reading."""
    if autoscout_state is not None:
        from autoscout.state import StateDir

        knowledge = StateDir(Path(autoscout_state)).load_knowledge()
        return [
            Work(
                id=raw["work"]["id"], title=raw["work"]["title"],
                abstract=raw["work"]["abstract"],
                published=(dt.date.fromisoformat(raw["work"]["published"])
                           if raw["work"]["published"] else None),
                url=raw["work"]["url"], source=raw["work"]["source"],
                disciplines=list(raw["work"]["disciplines"]),
            )
            for raw in knowledge.recent_works if raw["flagged"]
        ]
    result = LocalCorpus(Path(corpus or DEFAULT_CORPUS)).harvest()
    if result.errors and not result.works:
        raise ValueError("; ".join(result.errors))
    return result.works


def find_directions(works: list[Work], source: ClaimSource,
                    asof: dt.date = DEFAULT_ASOF) -> list[Direction]:
    """Discover directions with sciscout, score them, and attach their claims."""
    found = discover(works)
    pipeline = Pipeline(asof=asof)
    available = set(pipeline.priors.sectors())
    tracks = []
    for discovered in found.tracks:
        track = discovered.to_track()
        track.sector = classify(track, available).sector
        tracks.append(track)
    terms = {d.id: d.terms for d in found.tracks}
    directions = []
    for report in pipeline.analyse(tracks, simulate=False):
        ids = {w.id for w in report.track.works}
        directions.append(Direction(
            id=report.track.id, label=report.track.name, sector=report.track.sector,
            terms=terms[report.track.id], works=ids,
            priority=report.assessment.priority, confidence=report.assessment.confidence,
            pool=source.claims_for(ids),
        ))
    return directions


# -- the loop ------------------------------------------------------------------


class Checker:
    """Runs claims through the aisci verifier service, remembering each result.

    The verifiers are deterministic, so a claim is run once however many
    policies or seeds ask about it.
    """

    def __init__(self) -> None:
        verifiers, _, _ = _aisci()
        self.service = verifiers.default_service()
        self.cache: dict[str, tuple[bool, str]] = {}

    def check(self, claim: Claim) -> tuple[bool, str]:
        if claim.id not in self.cache:
            if claim.verifier not in self.service.verifiers:
                self.cache[claim.id] = (False, f"no verifier named {claim.verifier!r}")
            else:
                verdict = self.service.check(claim.verifier, claim.candidate)
                self.cache[claim.id] = (verdict.passed, verdict.reason)
        return self.cache[claim.id]


def _choose(policy: str, open_: list[Direction], naive: dict[str, list[int]],
            rng: np.random.Generator) -> Direction:
    """Pick the direction whose next claim gets checked."""
    if policy == "uniform":
        return open_[int(rng.integers(len(open_)))]
    if policy == "priority":
        weights = np.array([d.priority for d in open_])
        return open_[int(rng.choice(len(open_), p=weights / weights.sum()))]
    if policy == "oracle":
        return max(open_, key=lambda d: d.priority * (d.true_yield() or 0.0))
    if policy == "naive":
        # Believes every pass: a Beta posterior on the raw pass rate.
        return max(open_, key=lambda d: d.priority * rng.beta(
            1 + naive[d.id][0], 1 + naive[d.id][1] - naive[d.id][0]))
    return max(open_, key=lambda d: d.priority * d.belief.sample(rng))


def replay(directions: list[Direction], history: list[Check],
           rates: dict[str, ErrorRates]) -> set[str]:
    """Rebuild each direction's belief from recorded checks.

    Checks are kept per claim and per work, not per direction, so a corpus that
    clusters differently next time still inherits everything already learned.
    """
    by_work: dict[str, Direction] = {w: d for d in directions for w in d.works}
    done = set()
    for check in history:
        done.add(check.claim)
        direction = by_work.get(check.work)
        if direction is not None:
            direction.belief.update(check.passed, rates.get(check.regime, UNCALIBRATED))
    return done


def run_checks(directions: list[Direction], checker: Checker, rates: dict[str, ErrorRates],
               budget: int, rng: np.random.Generator, policy: str = "loop",
               done: set[str] | None = None, cycle: int = 1) -> list[Check]:
    """Spend ``budget`` checks, updating each direction's belief as results arrive."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {', '.join(POLICIES)}")
    done = set() if done is None else done
    queues = {d.id: [c for c in d.pool if c.id not in done] for d in directions}
    naive = {d.id: [d.belief.passes, d.belief.checks] for d in directions}
    checks: list[Check] = []
    for _ in range(budget):
        open_ = [d for d in directions if queues[d.id]]
        if not open_:
            break
        direction = _choose(policy, open_, naive, rng)
        claim = queues[direction.id].pop(0)
        passed, reason = checker.check(claim)
        direction.belief.update(passed, rates.get(claim.regime, UNCALIBRATED))
        naive[direction.id][0] += int(passed)
        naive[direction.id][1] += 1
        done.add(claim.id)
        checks.append(Check(claim.id, claim.work, claim.regime, passed, reason, cycle,
                            claim.truth))
    return checks


def ranking(directions: list[Direction]) -> list[Direction]:
    return sorted(directions, key=lambda d: d.verified_priority, reverse=True)


def ledger_for(directions: list[Direction], rates: dict[str, ErrorRates]) -> ProvenanceLedger:
    """Where each number behind the verified ranking came from."""
    ledger = ProvenanceLedger()
    for d in directions:
        ledger.record(Estimate(d.priority, observed(
            f"sciscout priority; {d.confidence:.0%} of it rests on evidence"),
            label=f"{d.label}: priority"))
        ledger.record(d.belief.estimate(f"{d.label}: yield"))
    used = {c.regime for d in directions for c in d.pool}
    for regime in sorted(used):
        r = rates.get(regime, UNCALIBRATED)
        ledger.record(r.false_pass.relabel(f"{regime}: false pass"))
        ledger.record(r.false_reject.relabel(f"{regime}: false reject"))
    return ledger


# -- steering autoscout --------------------------------------------------------


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def agenda_for(directions: list[Direction], top: int = 5, min_checks: int = 5) -> dict:
    """An autoscout agenda built from the directions that survived checking.

    Only directions with at least ``min_checks`` checks are eligible: an
    unchecked direction sits at the prior mean of one half, which would rank it
    above a direction that has been checked and found middling. Reading more
    because nothing is known is exploration, and that is the allocator's job,
    not the agenda's.
    """
    from autoscout.config import valid_term

    eligible = [d for d in ranking(directions) if d.belief.checks >= min_checks][:top]
    categories: list[str] = []
    topics = {}
    for d in eligible:
        for cat in (SECTOR_CATEGORIES[d.sector].value if d.sector in SECTOR_CATEGORIES else []):
            if cat not in categories:
                categories.append(cat)
        lo, hi = d.belief.interval()
        topics[_slug(d.label)] = {
            "description": (
                f"{d.label}: sciscout priority {d.priority:.2f}, verified yield "
                f"{d.belief.mean:.2f} ({lo:.2f}-{hi:.2f}) from {d.belief.checks} checks."),
            "terms": [t for t in d.terms if valid_term(t)],
        }
    return {
        "name": "directions that survived verification (written by nexus loop)",
        "queries": {"categories": categories},
        "topics": topics,
    }


# -- state ---------------------------------------------------------------------


class LoopState:
    """What the loop remembers between runs: every check, as an auditable chain."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def checks_path(self) -> Path:
        return self.path / "checks.jsonl"

    def history(self) -> list[Check]:
        if not self.checks_path.exists():
            return []
        return [Check(**json.loads(line))
                for line in self.checks_path.read_text().splitlines() if line.strip()]

    def record(self, checks: list[Check], claims: dict[str, Claim]) -> dict:
        """Append checks, then rebuild the experiment store and world model."""
        self.path.mkdir(parents=True, exist_ok=True)
        with self.checks_path.open("a") as handle:
            for check in checks:
                handle.write(json.dumps(check.to_dict(), sort_keys=True) + "\n")
        return self.rebuild(claims)

    def rebuild(self, claims: dict[str, Claim],
                rates: dict[str, ErrorRates] | None = None) -> dict:
        _, datastore, worldmodel = _aisci()
        rates = rates if rates is not None else load_error_rates()
        store = datastore.DataStore()
        world = worldmodel.WorldModel()
        for check in self.history():
            claim = claims.get(check.claim)
            errored = check.reason.startswith(("verifier error", "no verifier"))
            store.append({
                "record_id": store.issue_id("nexus-loop"),
                "campaign_id": "nexus-loop",
                "hypothesis_id": check.claim,
                "protocol": {"verifier": check.regime, "work": check.work},
                "outcome": ("failed_run" if errored else
                            "success" if check.passed else "negative"),
                "result": {"passed": check.passed, "reason": check.reason},
                "agent": "nexus.loop",
                "instrument": check.regime,
                "timestamp_cycle": check.cycle,
            })
            if check.passed and claim is not None:
                r = rates.get(check.regime, UNCALIBRATED)
                # How much a pass is worth before anything is known about the
                # direction: the chance it is right when half of claims are true.
                trust = (1 - r.false_reject.value) / (
                    1 - r.false_reject.value + r.false_pass.value)
                world.add(worldmodel.Claim(
                    id=check.claim, text=claim.statement, type="data", confidence=trust,
                    evidence=[worldmodel.Evidence(
                        "code", f"aisci verifier {check.regime} on {check.claim}",
                        hashlib.sha256(json.dumps(claim.candidate, sort_keys=True,
                                                  default=str).encode()).hexdigest()[:16])],
                    cycle=check.cycle))
        (self.path / "records.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in store.records))
        (self.path / "worldmodel.json").write_text(world.to_json() + "\n")
        return store.audit()


# -- measuring the loop --------------------------------------------------------


def value_found(checks: list[Check], directions: list[Direction]) -> float:
    """Priority-weighted true claims confirmed. Needs the synthetic truth labels."""
    priority = {w: d.priority for d in directions for w in d.works}
    return sum(priority.get(c.work, 0.0) for c in checks if c.truth and c.passed)


def experiment(works: list[Work], source: ClaimSource, rates: dict[str, ErrorRates],
               budget: int = 192, seeds: int = 20, asof: dt.date = DEFAULT_ASOF) -> dict:
    """Compare allocation policies on value found for the same checking budget.

    ``loop`` is this module. ``naive`` is the same sampler believing every
    pass. ``priority`` spends in proportion to sciscout's ranking and never
    looks at a result -- what sciscout alone would do. ``uniform`` spreads
    evenly. ``oracle`` knows every direction's true yield and is the ceiling.
    """
    checker = Checker()
    base = find_directions(works, source, asof)
    values: dict[str, list[float]] = {p: [] for p in POLICIES}
    trap_share: dict[str, list[float]] = {p: [] for p in POLICIES}
    traps = _trap_ids(base, source)
    for seed in range(seeds):
        for policy in POLICIES:
            directions = [Direction(d.id, d.label, d.sector, d.terms, d.works, d.priority,
                                    d.confidence, d.pool) for d in base]
            rng = np.random.default_rng([seed, POLICIES.index(policy)])
            checks = run_checks(directions, checker, rates, budget, rng, policy)
            values[policy].append(value_found(checks, directions))
            by_work = {w: d.id for d in directions for w in d.works}
            trap_share[policy].append(
                sum(by_work.get(c.work) in traps for c in checks) / max(1, len(checks)))
    out = {"budget": budget, "seeds": seeds, "directions": len(base),
           "claims": sum(len(d.pool) for d in base), "policies": {}}
    rng = np.random.default_rng(0)
    for policy in POLICIES:
        v = np.array(values[policy])
        row = {"mean": float(v.mean()), "share_of_oracle": float(
            v.mean() / np.mean(values["oracle"])),
            "trap_share": float(np.mean(trap_share[policy]))}
        if policy != "loop":
            diff = np.array(values["loop"]) - v
            boot = diff[rng.integers(len(diff), size=(4000, len(diff)))].mean(axis=1)
            row["loop_minus"] = float(diff.mean())
            row["loop_minus_90"] = [float(np.quantile(boot, 0.05)),
                                    float(np.quantile(boot, 0.95))]
        out["policies"][policy] = row
    return out


def _trap_ids(directions: list[Direction], source: ClaimSource) -> set[str]:
    """Directions whose claims come mostly from the bank's planted trap tracks."""
    traps = set(getattr(source, "traps", []))
    out = set()
    for d in directions:
        tracks = [c.id.split(":")[1] for c in d.pool if c.id.count(":") >= 2]
        if tracks and sum(t in traps for t in tracks) > len(tracks) / 2:
            out.add(d.id)
    return out


# -- reporting -----------------------------------------------------------------


def render(directions: list[Direction], rates: dict[str, ErrorRates], audit: dict,
           source: ClaimSource, results: dict | None = None) -> str:
    ranked = ranking(directions)
    by_priority = sorted(directions, key=lambda d: d.priority, reverse=True)
    before = {d.id: i + 1 for i, d in enumerate(by_priority)}
    total = sum(d.belief.checks for d in directions)
    lines = [
        "# Closed discovery loop", "",
        f"_{len(directions)} directions, {sum(len(d.pool) for d in directions)} claims "
        f"available, {total} checked._", "",
    ]
    note = getattr(source, "note", "")
    if note:
        lines += [f"**Claim source: {source.name}.** {note} The ranking below shows the "
                  "mechanism working; it says nothing about the real fields these "
                  "directions are named after.", ""]
    lines += [
        "## Directions re-ranked by verified priority", "",
        "Verified priority is sciscout's priority times the share of the direction's "
        "claims estimated to be true. The interval is the 90% credible range on that "
        "share. A direction with no checks sits at the prior mean of one half, with an "
        "interval that says so.", "",
        "| # | Direction | Was | Priority | Checks | Passed | Yield (90%) | Verified priority |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, d in enumerate(ranked, 1):
        lo, hi = d.belief.interval()
        vlo, vhi = d.verified_interval()
        lines.append(
            f"| {i} | {d.label} | {before[d.id]} | {d.priority:.2f} | {d.belief.checks} "
            f"| {d.belief.passes} | {d.belief.mean:.2f} ({lo:.2f}-{hi:.2f}) "
            f"| {d.verified_priority:.2f} ({vlo:.2f}-{vhi:.2f}) |")
    lines += ["", "## How much a pass is worth", "",
              "Error rates measured in `aisci-verifiers/results/i1_calibration.json`. "
              "The last column is the chance a passing claim is true when half of all "
              "claims are.", "",
              "| Verifier | Wrong candidates passed | Correct candidates rejected "
              "| A pass is right |", "| --- | --- | --- | --- |"]
    for regime in sorted({c.regime for d in directions for c in d.pool}):
        r = rates.get(regime, UNCALIBRATED)
        fp, fr = r.false_pass.value, r.false_reject.value
        lines.append(f"| {regime} | {fp:.2f} | {fr:.2f} | {(1 - fr) / (1 - fr + fp):.0%} |")
    lines += ["", "The rates are posterior means under a uniform prior on small "
              "calibration sets (one to forty cases), so they are deliberately not zero "
              "where no error was seen.", ""]
    if audit:
        outcomes = ", ".join(f"{n} {k}" for k, n in sorted(audit["by_outcome"].items()))
        lines += ["## Record", "",
                  f"{audit['written']} checks recorded ({outcomes}); hash chain "
                  f"{'intact' if audit['chain_intact'] else 'BROKEN'}; "
                  f"{len(audit['missing'])} issued but unwritten.", ""]
    if results:
        lines += _render_experiment(results)
    lines += ["## Where the numbers came from", "",
              ledger_for(directions, rates).audit_table(), ""]
    return "\n".join(lines)


def _render_experiment(results: dict) -> list[str]:
    names = {"loop": "loop (this module)", "naive": "loop believing every pass",
             "priority": "sciscout priority only", "uniform": "uniform",
             "oracle": "oracle (knows true yields)"}
    lines = [
        "## Does closing the loop help?", "",
        f"Each policy spends {results['budget']} checks on the same "
        f"{results['claims']} claims across {results['directions']} directions; "
        f"{results['seeds']} seeds. Value is the sciscout priority of each true claim "
        "confirmed, summed. Trap share is the share of checks spent on the two "
        "directions planted with low yield and lenient checking.", "",
        "| Policy | Value found | Share of oracle | Trap share | Loop's lead (90%) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for policy in POLICIES:
        row = results["policies"][policy]
        lead = "-"
        if "loop_minus" in row:
            lo, hi = row["loop_minus_90"]
            lead = f"{row['loop_minus']:+.1f} ({lo:+.1f} to {hi:+.1f})"
        lines.append(f"| {names[policy]} | {row['mean']:.1f} | {row['share_of_oracle']:.0%} "
                     f"| {row['trap_share']:.0%} | {lead} |")
    lines += ["", "The bank's yields and traps were planted by the author of the loop, "
              "so this shows the mechanism does what it claims on a case built to test "
              "it. It is not evidence about real research claims.", ""]
    return lines


# -- command line --------------------------------------------------------------


def cmd_loop(args) -> int:
    try:
        source = BankSource(Path(args.bank))
        rates = load_error_rates(Path(args.calibration))
        works = load_works(Path(args.corpus), args.from_autoscout)
        directions = find_directions(works, source, args.asof)
        checker = Checker()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not directions:
        print("error: no directions found in the corpus", file=sys.stderr)
        return 2

    state = LoopState(Path(args.state))
    if args.reset and state.checks_path.exists():
        state.checks_path.unlink()
    history = state.history()
    done = replay(directions, history, rates)
    first = 1 + max((c.cycle for c in history), default=0)
    claims = {c.id: c for d in directions for c in d.pool}
    audit: dict = {}
    for cycle in range(first, first + args.cycles):
        rng = np.random.default_rng([args.seed, cycle])
        checks = run_checks(directions, checker, rates, args.budget, rng, "loop", done, cycle)
        audit = state.record(checks, claims)
        top = ranking(directions)[0]
        print(f"cycle {cycle}: {len(checks)} checks, {sum(c.passed for c in checks)} passed; "
              f"leading: {top.label} ({top.verified_priority:.2f})")
        if not checks:
            print("no unchecked claims left")
            break

    results = None
    if args.experiment:
        results = experiment(works, source, rates, args.experiment_budget, args.seeds,
                             args.asof)
    report = render(directions, rates, audit, source, results)
    state.path.mkdir(parents=True, exist_ok=True)
    (state.path / "REPORT.md").write_text(report + "\n")
    agenda = agenda_for(directions, args.top)
    (state.path / "agenda.yaml").write_text(yaml.safe_dump(agenda, sort_keys=False))
    if args.out:
        Path(args.out).write_text(report + "\n")
    print(f"\nreport: {state.path / 'REPORT.md'}")
    print(f"agenda for autoscout ({len(agenda['topics'])} topics): {state.path / 'agenda.yaml'}")
    print(f"  python -m autoscout.cli --state <dir> init --agenda {state.path / 'agenda.yaml'}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("loop", help="verify claims per direction and re-rank by the results",
                       description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--from-autoscout", type=Path, default=None, metavar="STATE",
                   help="rank the papers an autoscout state directory has flagged, "
                        "instead of --corpus")
    p.add_argument("--bank", default=str(DEFAULT_BANK), help="claim bank to replay")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    p.add_argument("--state", default=str(DEFAULT_STATE), help="state directory")
    p.add_argument("--reset", action="store_true", help="forget earlier checks")
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--budget", type=int, default=24, help="checks per cycle")
    p.add_argument("--top", type=int, default=5, help="directions written to the agenda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--asof", type=dt.date.fromisoformat, default=DEFAULT_ASOF)
    p.add_argument("--experiment", action="store_true",
                   help="also compare allocation policies (needs truth labels)")
    p.add_argument("--experiment-budget", type=int, default=192)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default=None, help="also write the report here")
    p.set_defaults(func=cmd_loop)
