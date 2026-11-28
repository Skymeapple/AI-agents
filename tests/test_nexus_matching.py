"""Problem-direction matching: the three lists, and the rule that refuses."""

from __future__ import annotations

import datetime as dt

import pytest

from nexus import matching
from sciscout.discovery import discover
from sciscout.models import Work
from sciscout.provenance import Grade, assumed

TOPICS = {
    "battery": "sodium battery electrolyte cathode anode cycling dendrite",
    "qubit": "quantum qubit error correction logical surface code",
    "protein": "protein folding structure prediction residue backbone",
}
FILLER = ["measured", "reported", "studied", "improved", "analysed", "observed"]


def corpus() -> list[Work]:
    works = []
    for topic, words in TOPICS.items():
        vocab = words.split()
        for i in range(8):
            picked = " ".join(vocab[(i + k) % len(vocab)] for k in range(5))
            works.append(Work(
                id=f"{topic}-{i}", title=f"{vocab[0]} {vocab[1]} study {i}",
                abstract=f"{picked} {FILLER[i % len(FILLER)]} {picked}",
                published=dt.date(2026, 1, 1)))
    return works


def problem(pid: str, text: str, catalogue: str = "cat", field: str = "physics"):
    return matching.Problem(pid, catalogue, pid, text, assumed("written for the test"), field)


@pytest.fixture(scope="module")
def found():
    works = corpus()
    result = discover(works)
    assert len(result.tracks) == 3
    return works, result


def direction_about(result, word: str):
    return next(d for d in result.directions if any(word in w.id for w in d.works))


def test_planted_problem_is_matched_to_its_direction(found):
    works, directions = found
    p = problem("battery", "sodium battery electrolyte, cathode cycling")
    result = matching.match([p], directions, works, corpus_fields=frozenset({"physics"}))
    links = result.links_for(p)
    assert len(links) == 1
    assert all(w.id.startswith("battery") for w in links[0].direction.works)
    assert result.matched == [p] and not result.neglected


def test_direction_with_no_problem_is_an_orphan(found):
    works, directions = found
    p = problem("battery", "sodium battery electrolyte, cathode cycling")
    result = matching.match([p], directions, works)
    orphan_topics = {d.works[0].id.split("-")[0] for d in result.orphans}
    assert orphan_topics == {"qubit", "protein"}


def test_unmatched_problem_is_neglected_only_inside_the_corpus_scope(found):
    works, directions = found
    inside = problem("fusion", "tokamak plasma confinement", field="physics")
    outside = problem("dark", "dark matter halo", field="cosmology")
    result = matching.match([inside, outside], directions, works,
                            corpus_fields=frozenset({"physics"}))
    assert result.neglected == [inside]
    assert result.outside == [outside]
    # With no declared scope, nothing may be called neglected.
    undeclared = matching.match([inside, outside], directions, works)
    assert undeclared.neglected == [] and len(undeclared.outside) == 2


def test_direction_between_two_problems_is_linked_to_neither(found):
    works, directions = found
    a = problem("a", "quantum qubit error correction")
    b = problem("b", "quantum qubit logical surface code")
    result = matching.match([a, b], directions, works, margin=1.5)
    qubit = direction_about(result, "qubit")
    assert not [l for l in result.links if l.direction is qubit]
    refusal = next(r for r in result.refusals if r.direction is qubit)
    assert refusal.ambiguous and "runner-up" in refusal.reason
    # Without a rival the same problem is linked.
    alone = matching.match([a], directions, works, margin=1.5)
    assert len(alone.links_for(a)) == 1


def test_margin_applies_within_a_catalogue_not_across_them(found):
    works, directions = found
    a = problem("a", "quantum qubit error correction", catalogue="one")
    b = problem("b", "quantum qubit error correction", catalogue="two")
    result = matching.match([a, b], directions, works, margin=1.5)
    assert len(result.links_for(a)) == 1 and len(result.links_for(b)) == 1


def test_weak_overlap_is_refused_by_the_floor(found):
    works, directions = found
    p = problem("vague", "study of improved materials measured and reported widely")
    result = matching.match([p], directions, works, floor=0.5)
    assert not result.links


def test_removing_a_problem_orphans_its_direction(found):
    works, directions = found
    a = problem("battery", "sodium battery electrolyte")
    b = problem("qubit", "quantum qubit error correction")
    scored = matching.score([a, b], directions, works)
    full = matching.apply_rule(scored)
    dropped = matching.apply_rule(scored, without=b)
    assert len(dropped.orphans) == len(full.orphans) + 1
    assert b not in dropped.problems


def test_rule_constants_and_counts_carry_provenance(found):
    works, directions = found
    assert matching.SIMILARITY_FLOOR.grade is Grade.ASSUMED
    assert matching.MARGIN_OVER_RUNNER_UP.grade is Grade.ASSUMED
    assert matching.FRONTIER_FIELDS.grade is Grade.ASSUMED
    result = matching.match([problem("battery", "sodium battery")], directions, works)
    assert all(e.grade is Grade.OBSERVED for e in result.counts().values())


def test_shipped_catalogues_load_with_graded_matching_text():
    problems = matching.load_catalogues()
    by = {}
    for p in problems:
        by.setdefault(p.catalogue, []).append(p)
    assert len(by["ai100"]) == 100 and len(by["ai2035"]) == 20
    assert set(by) == set(matching.CATALOGUES)
    assert all(p.text.strip() for p in problems)
    # The English text for AI-100 was written for matching, not taken from a source.
    assert {p.provenance.grade for p in by["ai100"]} == {Grade.ASSUMED}
    assert len({p.key for p in problems}) == len(problems)


def test_known_pairs_point_at_real_questions():
    pairs = matching.load_known_pairs()
    ai100 = {p.key for p in matching.load_catalogues(only=["ai100"])}
    ai2035 = {p.id for p in matching.load_catalogues(only=["ai2035"])}
    assert pairs and len(set(pairs.values())) == len(pairs)
    assert set(pairs) <= ai100 and set(pairs.values()) <= ai2035


def test_cli_reports_synthetic_corpus_as_synthetic_and_calls_nothing_neglected(tmp_path, capsys):
    from nexus.cli import main

    out = tmp_path / "report.md"
    code = main(["matching", "--corpus", str(matching.DEMO_CORPUS), "--no-evaluate",
                 "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "SYNTHETIC" in printed and " 0 neglected" in printed
    assert "SYNTHETIC" in out.read_text()
