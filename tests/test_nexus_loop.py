"""The closed loop: claims about what it does, asserted directly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("sympy", reason="nexus loop runs the aisci verifiers")
pytest.importorskip("jsonschema", reason="nexus loop writes to the aisci datastore")

from nexus import loop  # noqa: E402


@pytest.fixture(scope="module")
def rates():
    return loop.load_error_rates()


@pytest.fixture(scope="module")
def source():
    return loop.BankSource()


@pytest.fixture(scope="module")
def base(source):
    return loop.find_directions(loop.load_works(), source)


@pytest.fixture(scope="module")
def checker():
    return loop.Checker()


def fresh(base):
    return [loop.Direction(d.id, d.label, d.sector, d.terms, d.works, d.priority,
                           d.confidence, d.pool) for d in base]


# -- a pass is evidence, not truth ---------------------------------------------


def test_weak_verifier_pass_moves_belief_far_less_than_a_strong_one(rates):
    weak, strong = loop.YieldBelief(), loop.YieldBelief()
    for _ in range(10):
        weak.update(True, rates["python_tests[weak tests]"])
        strong.update(True, rates["symbolic_identity"])
    assert strong.mean > 0.85
    assert weak.mean < 0.60
    assert weak.interval()[0] < 0.2  # ten lenient passes rule almost nothing out


def test_uncalibrated_verifier_teaches_nothing():
    belief = loop.YieldBelief()
    for passed in (True, True, True, False, True):
        belief.update(passed, loop.UNCALIBRATED)
    assert belief.mean == pytest.approx(0.5)
    assert belief.interval() == loop.YieldBelief().interval()


def test_error_rates_are_not_zero_where_no_error_was_seen(rates):
    # 0 wrong passes in 40 is strong evidence, not proof.
    r = rates["matmul_tensor"]
    assert 0 < r.false_pass.value < 0.05
    assert r.false_pass.grade.value == "observed"
    assert "0 of 40" in r.false_pass.provenance.detail


def test_unchecked_direction_is_graded_assumed_and_checked_one_observed(rates):
    belief = loop.YieldBelief()
    assert belief.estimate("y").grade.value == "assumed"
    belief.update(False, rates["symbolic_identity"])
    assert belief.estimate("y").grade.value == "observed"


# -- the loop ------------------------------------------------------------------


def test_bank_candidates_behave_as_planted(source, checker):
    """Strong checkers agree with the planted truth; the weak one does not."""
    wrong_passed = {"weak": 0, "strong": 0}
    for claim in source.claims[:250]:
        passed, _ = checker.check(claim)
        kind = "weak" if claim.variant == "[weak tests]" else "strong"
        if passed and not claim.truth:
            wrong_passed[kind] += 1
        if kind == "strong":
            assert passed == claim.truth, claim.id
    assert wrong_passed["weak"] > 0


def test_loop_beats_priority_alone_and_believing_every_pass(base, checker, rates):
    seeds, budget = 5, 192
    value = {p: [] for p in ("loop", "naive", "priority", "oracle")}
    for seed in range(seeds):
        for policy in value:
            directions = fresh(base)
            rng = np.random.default_rng([seed, loop.POLICIES.index(policy)])
            checks = loop.run_checks(directions, checker, rates, budget, rng, policy)
            assert len(checks) == budget
            value[policy].append(loop.value_found(checks, directions))
    for seed in range(seeds):
        assert value["loop"][seed] > value["priority"][seed]
        assert value["loop"][seed] > value["naive"][seed]
        assert value["loop"][seed] <= value["oracle"][seed]


def test_loop_is_not_captured_by_the_leniently_checked_directions(base, checker, rates, source):
    traps = loop._trap_ids(base, source)
    assert len(traps) == 2
    share = {}
    for policy in ("loop", "naive"):
        directions = fresh(base)
        checks = loop.run_checks(directions, checker, rates, 192,
                                 np.random.default_rng(3), policy)
        by_work = {w: d.id for d in directions for w in d.works}
        share[policy] = sum(by_work[c.work] in traps for c in checks) / len(checks)
        if policy == "loop":
            for d in directions:
                if d.id in traps and d.belief.checks >= 10:
                    # Most of its checks passed, and the loop still does not believe it.
                    assert d.belief.passes / d.belief.checks > d.belief.mean
    assert share["naive"] > 1.5 * share["loop"]


def test_decisions_never_read_the_truth_labels(base, checker, rates):
    def sequence(strip: bool):
        directions = fresh(base)
        if strip:
            for d in directions:
                d.pool = [loop.Claim(c.id, c.work, c.verifier, c.statement, c.candidate,
                                     c.variant, None) for c in d.pool]
        checks = loop.run_checks(directions, checker, rates, 60,
                                 np.random.default_rng(11), "loop")
        return [(c.claim, c.passed) for c in checks]

    assert sequence(strip=True) == sequence(strip=False)


def test_unknown_policy_is_refused(base, checker, rates):
    with pytest.raises(ValueError, match="unknown policy"):
        loop.run_checks(fresh(base), checker, rates, 1, np.random.default_rng(0), "greedy")


# -- memory and audit ----------------------------------------------------------


def test_state_resumes_without_rechecking_and_keeps_failures(tmp_path, base, checker, rates):
    state = loop.LoopState(tmp_path)
    directions = fresh(base)
    claims = {c.id: c for d in directions for c in d.pool}
    first = loop.run_checks(directions, checker, rates, 40, np.random.default_rng(1), cycle=1)
    audit = state.record(first, claims)
    assert audit["chain_intact"] and audit["written"] == 40 and not audit["missing"]
    assert audit["by_outcome"].get("negative", 0) == sum(not c.passed for c in first)

    # A new process: beliefs come back from the record alone.
    resumed = fresh(base)
    done = loop.replay(resumed, state.history(), rates)
    assert done == {c.claim for c in first}
    for before, after in zip(directions, resumed):
        assert after.belief.checks == before.belief.checks
        assert after.belief.mean == pytest.approx(before.belief.mean)

    second = loop.run_checks(resumed, checker, rates, 40, np.random.default_rng(2),
                             done=done, cycle=2)
    assert not {c.claim for c in first} & {c.claim for c in second}
    audit = state.record(second, claims)
    assert audit["written"] == 80 and audit["chain_intact"]

    world = json.loads((tmp_path / "worldmodel.json").read_text())
    passed = [c for c in first + second if c.passed]
    assert set(world) == {c.claim for c in passed}
    assert all(entry["evidence"] and 0 < entry["confidence"] < 1 for entry in world.values())


def test_tampering_with_the_record_is_detected(tmp_path, base, checker, rates):
    state = loop.LoopState(tmp_path)
    directions = fresh(base)
    checks = loop.run_checks(directions, checker, rates, 10, np.random.default_rng(1))
    state.record(checks, {c.id: c for d in directions for c in d.pool})
    _, datastore, _ = loop._aisci()
    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert datastore.DataStore(records=records).audit()["chain_intact"]
    records[3]["result"]["passed"] = not records[3]["result"]["passed"]
    assert not datastore.DataStore(records=records).audit()["chain_intact"]


# -- steering autoscout --------------------------------------------------------


def test_agenda_is_readable_by_autoscout_and_excludes_unchecked_directions(
        tmp_path, base, checker, rates):
    from autoscout.config import AgentConfig

    directions = fresh(base)
    loop.run_checks(directions, checker, rates, 120, np.random.default_rng(0))
    agenda = loop.agenda_for(directions, top=5, min_checks=5)
    path = tmp_path / "agenda.yaml"
    path.write_text(yaml.safe_dump(agenda))
    config = AgentConfig.from_agenda(path)
    assert 1 <= len(config.topics()) <= 5
    assert all(config.profiles[t] for t in config.topics())

    checked = {loop._slug(d.label) for d in directions if d.belief.checks >= 5}
    assert set(agenda["topics"]) <= checked
    best = loop.ranking([d for d in directions if d.belief.checks >= 5])[0]
    assert loop._slug(best.label) in agenda["topics"]
    assert agenda["queries"]["categories"]


def test_agenda_is_empty_before_anything_is_checked(base):
    assert loop.agenda_for(fresh(base))["topics"] == {}


def test_loop_can_rank_what_autoscout_flagged(tmp_path):
    from autoscout.state import Knowledge, StateDir

    state = StateDir(tmp_path)
    work = {"id": "w1", "title": "Tool use in agents", "abstract": "Agents call tools.",
            "published": "2026-10-01", "url": None, "source": "stream", "disciplines": ["cs.AI"]}
    knowledge = Knowledge()
    knowledge.recent_works = [
        {"work": work, "topic": "tool_use", "score": 1.0, "flagged": True},
        {"work": {**work, "id": "w2"}, "topic": None, "score": 0.0, "flagged": False},
    ]
    state.save_knowledge(knowledge)
    works = loop.load_works(autoscout_state=tmp_path)
    assert [w.id for w in works] == ["w1"]
    assert works[0].published.year == 2026


# -- command line --------------------------------------------------------------


def test_cli_runs_resumes_and_writes_report_and_agenda(tmp_path, capsys):
    from nexus.cli import main

    args = ["loop", "--state", str(tmp_path), "--cycles", "2", "--budget", "12"]
    assert main(args) == 0
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "cycle 1:" in out and "cycle 4:" in out
    report = (tmp_path / "REPORT.md").read_text()
    assert "48 checked" in report and "Synthetic" in report and "hash chain intact" in report
    assert Path(tmp_path / "agenda.yaml").exists()
    assert len((tmp_path / "checks.jsonl").read_text().splitlines()) == 48
