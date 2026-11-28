"""Done-criteria tests for the second pass (tasks 1.1, 2.2-2.5, 4.4, 5.3, 5.4, RP1, RP3)."""
import numpy as np

from aisci import agent, evaluation_tools as ev, verifier_map
from aisci.domain_model import run as domain_run


def test_1_1_two_raters_score_20_domains_with_agreement():
    r = verifier_map.run()
    assert len(r["map"]) == 20
    assert r["agreement"]["strength_quadratic_kappa"] > 0.6


def test_2_2_agent_completes_50_cycles_unattended_with_full_traceability():
    r = agent.run(seed=3, cycles=50)
    assert r["cycles"] == 50 and r["claims_written"] > 100 and r["traceability"] == 1.0


def test_2_4_accuracy_curve_per_claim_type():
    r = agent.run(seed=1, cycles=200)
    for typ in ("data", "literature", "synthesis"):
        assert set(r["acc"][typ]) == {10, 50, 200}


def test_2_5_and_rp3_interventional_critic_lifts_synthesis_above_75pct():
    accs = [agent.run(seed=s, cycles=200, critic=True, audit=True, interventional=True)["acc"]["synthesis"][200] for s in range(3)]
    naive = [agent.run(seed=s, cycles=200)["acc"]["synthesis"][200] for s in range(3)]
    assert np.mean(accs) > 0.75 and np.mean(accs) - np.mean(naive) > 0.05


def test_4_4_domain_model_beats_baselines_on_held_out():
    assert domain_run()["beats_all_baselines"]


def test_5_3_review_packets_are_blind_and_balanced():
    items = [{"id": f"a{i}", "source": "agent", "text": f"Our AI agent found effect {i}."} for i in range(6)] + \
            [{"id": f"h{i}", "source": "human", "text": f"We found effect {i}."} for i in range(6)]
    packets, key, rubric = ev.build_review_packets(items, ["r1", "r2", "r3", "r4"], per_item=3)
    assert sum(len(p) for p in packets.values()) == 36
    assert max(len(p) for p in packets.values()) - min(len(p) for p in packets.values()) <= 1
    assert not any("agent" in x["text"].lower() or "AI" in x["text"].split() for p in packets.values() for x in p)
    scores = [{"reviewer": r, "code": x["code"], **{c: (4 if key[x["code"]]["source"] == "agent" else 2) for c in rubric}}
              for r, p in packets.items() for x in p]
    res = ev.analyse_reviews(scores, key, n_perm=500)
    assert res["correctness"]["agent_minus_human"] == 2 and res["correctness"]["p_perm"] < 0.05


def test_5_4_impact_tracker_snapshots():
    data = {"10.1/x": {"cited_by": 5, "replications": 1}, "10.1/y": {"cited_by": 0, "replications": 0}}
    t = ev.ImpactTracker(fetch=lambda doi: data[doi])
    t.register("f1", "10.1/x", "2026-11-01"); t.register("f2", "10.1/y", "2026-11-01")
    t.snapshot("2026Q4")
    assert t.dashboard() == {"findings": 2, "replicated": 1, "total_citations": 5}


def test_rp1_pipeline_extracts_questions_and_recovers_planted_signal():
    q = ev.extract_open_questions("We measure X. The mechanism of Y remains unknown. Z is fine.")
    assert q == ["The mechanism of Y remains unknown."]
    rng = np.random.default_rng(0)
    topics = ["quantum spin liquid", "protein folding kinetics", "dark matter halo", "soil microbiome", "battery anode"]
    hot = {"quantum spin liquid": 50, "protein folding kinetics": 40, "dark matter halo": 5, "soil microbiome": 3, "battery anode": 20}
    recs = [{"year": 2000 + i % 15, "question": f"The origin of {t} behaviour remains unknown.",
             "impact_t10": int(rng.poisson(hot[t])), "early_citations": int(rng.poisson(2))}
            for i, t in enumerate(rng.choice(topics, 300))]
    r = ev.taste_pipeline(recs, split_year=2010)
    assert r["rho_text_model"] > 0.5 > r["rho_early_citation_heuristic"]
