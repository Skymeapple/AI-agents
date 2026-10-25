"""Discovery and sector routing.

The integration test against the demo corpus's planted labels is the one that
matters: an unsupervised step nobody measures is one nobody should trust.
"""

import datetime as dt

import numpy as np
import pytest

from sciscout.classify import CONFIDENCE_FLOOR, classify, coverage_report
from sciscout.commercial.profile import PriorLibrary
from sciscout.discovery import (
    _agglomerate,
    _cohesion,
    discover,
    discover_by_discipline,
    evaluate_against_labels,
    tokenize,
)
from sciscout.models import Track, Work
from sciscout.sources.local import LocalCorpus


def make_works(specs):
    """specs: list of (id, text, disciplines, label)."""
    return [
        Work(
            id=work_id,
            title=text,
            abstract=text,
            published=dt.date(2025, 1, 1),
            disciplines=list(disciplines),
            extra={"track": label},
        )
        for work_id, text, disciplines, label in specs
    ]


def two_topic_corpus(n=6):
    specs = []
    for i in range(n):
        specs.append(
            (f"bat-{i}", "sodium battery electrolyte cathode ionic conductivity cell",
             ["materials science"], "battery")
        )
        specs.append(
            (f"qub-{i}", "qubit coherence superconducting circuit gate fidelity quantum",
             ["quantum physics"], "qubit")
        )
    return make_works(specs)


# -- tokenisation ----------------------------------------------------------


def test_tokenizer_drops_stopwords_and_short_tokens():
    tokens = tokenize("We show that the novel catalyst is a b ok")
    assert "catalyst" in tokens
    for dropped in ("show", "novel", "the", "we", "is"):
        assert dropped not in tokens


# -- clustering behaviour --------------------------------------------------


def test_discovery_separates_two_distinct_topics():
    result = discover(two_topic_corpus(), min_cluster_size=3)
    assert len(result.tracks) == 2
    assert evaluate_against_labels(result, two_topic_corpus())["mean_purity"] == 1.0


def test_discovery_is_deterministic():
    works = two_topic_corpus()
    first = discover(works)
    second = discover(works)
    assert [t.label for t in first.tracks] == [t.label for t in second.tracks]
    assert [t.size for t in first.tracks] == [t.size for t in second.tracks]


def test_small_clusters_are_quarantined_not_promoted():
    """Two similar papers are not a research direction."""
    works = two_topic_corpus(n=6) + make_works(
        [("odd-1", "seismology tomography mantle wave inversion", ["geology"], "odd")]
    )
    result = discover(works, min_cluster_size=3)
    assert all(t.size >= 3 for t in result.tracks)
    assert any(w.id == "odd-1" for w in result.unclustered)
    assert result.coverage() < 1.0


def test_an_empty_corpus_yields_nothing_rather_than_failing():
    result = discover([])
    assert result.tracks == [] and result.coverage() == 0.0


def test_a_corpus_with_no_usable_vocabulary_is_handled():
    """Every work identical: no term distinguishes anything."""
    works = make_works([(f"w{i}", "identical text here", ["x"], "a") for i in range(5)])
    result = discover(works)
    assert result.tracks == []
    assert len(result.unclustered) == 5


def test_a_higher_threshold_never_yields_fewer_clusters():
    """Raising the similarity floor splits, it does not merge."""
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    loose = discover(works, threshold=0.10)
    tight = discover(works, threshold=0.22)
    assert len(tight.tracks) >= len(loose.tracks)


def _naive_average_linkage(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    """Textbook O(n^3) average linkage: rescan every cluster pair on every merge."""
    clusters = [[i] for i in range(similarity.shape[0])]
    while len(clusters) > 1:
        best, pair = -np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                value = similarity[np.ix_(clusters[a], clusters[b])].mean()
                if value > best:
                    best, pair = value, (a, b)
        if best < threshold:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return clusters


def test_fast_agglomeration_matches_the_textbook_algorithm():
    """The cached, lazily refreshed merge loop must not change any partition."""
    rng = np.random.default_rng(3)
    for trial in range(25):
        points = rng.normal(size=(40, 6)) + rng.integers(0, 4, size=(40, 1)) * 2.0
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        similarity = points @ points.T
        threshold = float(rng.uniform(0.0, 0.6))
        fast = _agglomerate(similarity.copy(), threshold)
        slow = _naive_average_linkage(similarity, threshold)
        assert sorted(map(sorted, fast)) == sorted(map(sorted, slow)), trial


def test_cohesion_from_sums_equals_mean_pairwise_similarity():
    rng = np.random.default_rng(5)
    matrix = rng.random((30, 8))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    members = [0, 3, 4, 9, 17, 22]
    block = (matrix @ matrix.T)[np.ix_(members, members)]
    expected = (block.sum() - np.trace(block)) / (len(members) * (len(members) - 1))
    assert _cohesion(matrix, members) == pytest.approx(expected)


def test_large_corpora_take_the_sampled_path_and_still_find_the_directions():
    """Above max_exact, discovery samples instead of refusing, and must not
    lose quality doing it. The demo corpus is forced down the sampled path."""
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    result = discover(works, max_exact=400, sample_size=300)
    scores = evaluate_against_labels(result, works)
    assert scores["directions_recovered"] == scores["true_directions"] == 16
    assert scores["mean_purity"] >= 0.99


def test_a_direction_too_rare_to_be_sampled_is_still_found():
    """Sampling must not cost the emerging directions discovery exists to catch."""
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    rare = [
        Work(
            id=f"rare-{i}",
            title="Magnetocaloric refrigeration regenerator",
            abstract="Gadolinium magnetocaloric regenerator beds cool without "
            "refrigerant gases using permanent magnet field cycles",
            published=dt.date(2026, 1, 1),
            extra={"track": "rare"},
        )
        for i in range(5)
    ]
    result = discover(works + rare, max_exact=400, sample_size=300)
    homes = {id(t) for t in result.tracks for w in t.works if w.id.startswith("rare-")}
    assert len(homes) == 1
    track = next(t for t in result.tracks if id(t) in homes)
    assert sum(w.id.startswith("rare-") for w in track.works) == 5
    assert track.size == 5


def test_partitioned_discovery_keeps_disciplines_apart():
    works = two_topic_corpus(n=12)
    result = discover_by_discipline(works, min_partition=5, min_cluster_size=3)
    for track in result.tracks:
        disciplines = {d for w in track.works for d in w.disciplines}
        assert len(disciplines) == 1


# -- the integration test that actually validates discovery ----------------


def test_discovery_recovers_the_planted_directions_in_the_demo_corpus():
    """The corpus plants 16 directions; discovery must find them unaided.

    Deliberately includes adjacent pairs -- quantum error correction beside
    topological qubits, protein design beside climate emulators -- so that a
    method which merely separates obviously-different vocabularies would not
    pass.
    """
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    scores = evaluate_against_labels(discover(works), works)
    assert scores["recovery_rate"] == 1.0
    assert scores["mean_purity"] > 0.95
    assert scores["coverage"] > 0.95


# -- sector routing --------------------------------------------------------


def track_from(text: str, disciplines: list[str]) -> Track:
    return Track(
        id="t",
        name="t",
        works=[Work(id="w", title=text, abstract=text, disciplines=disciplines)],
    )


def test_routing_recognises_each_modelled_sector():
    cases = [
        ("biopharma", "phase ii clinical trial patient dose efficacy therapeutic", ["medicine"]),
        ("energy_hardware", "battery electrolyte energy density kwh cell grid", ["materials science"]),
        ("semiconductors", "qubit wafer fabrication transistor lithography chip", ["quantum physics"]),
        ("software_ai", "neural model training dataset benchmark inference", ["computer science"]),
        ("agriculture_food", "crop field trial seed harvest soil cultivar", ["agronomy"]),
    ]
    for expected, text, disciplines in cases:
        assert classify(track_from(text, disciplines)).sector == expected


def test_an_unrecognised_field_routes_to_generic_and_says_so():
    """The important behaviour: admit the gap rather than pick a best guess."""
    match = classify(track_from("mantle tomography seismic anisotropy", ["geophysics"]))
    assert match.sector == "generic"
    estimate = match.as_estimate()
    assert "no commercialisation archetype matched" in estimate.provenance.detail


def test_a_genuine_tie_routes_to_generic():
    """Vocabulary split evenly across sectors must not be resolved by a coin flip."""
    match = classify(
        track_from(
            "clinical patient tumour therapeutic qubit wafer transistor photonic", []
        )
    )
    assert match.sector == "generic"


def test_a_lone_discipline_label_does_not_win_against_real_vocabulary():
    """The corroboration rule, pinned.

    "Electrical engineering" covers battery packs and lithography alike. A sector
    matched only by such a label, with none of its vocabulary present, must not
    outrank or suppress a sector the text is actually saturated in -- that was
    routing batteries, protein design and quantum error correction to generic.
    """
    match = classify(
        track_from(
            "sodium battery electrolyte anode cathode energy density kwh electrode",
            ["electrical engineering", "materials science"],
        )
    )
    assert match.sector == "energy_hardware"


def test_cross_sector_directions_are_still_allowed_to_be_ambiguous():
    """Corroboration must not make everything confident.

    Some directions genuinely span archetypes, and saying so beats forcing a
    choice that silently picks the cost model.
    """
    match = classify(
        track_from(
            "crop harvest soil cultivar germplasm qubit wafer transistor lithography",
            [],
        )
    )
    assert match.sector == "generic"


def test_routing_is_never_graded_as_observed():
    """Lexical routing is a guess and must be graded as one."""
    from sciscout.provenance import Grade

    match = classify(track_from("battery electrolyte cell energy density", ["materials science"]))
    assert match.as_estimate().grade is Grade.ASSUMED


def test_routing_will_not_name_a_sector_with_no_model_behind_it():
    """A sector we can name but cannot cost is worse than an honest generic."""
    match = classify(
        track_from("crop field trial seed harvest soil", ["agronomy"]),
        sectors_available={"biopharma", "software_ai"},
    )
    assert match.sector in {"generic", "biopharma", "software_ai"}
    assert match.sector != "agriculture_food"


def test_coverage_report_names_the_directions_it_cannot_model():
    matches = {
        "Batteries": classify(track_from("battery electrolyte cell kwh", ["materials science"])),
        "Seismology": classify(track_from("mantle tomography seismic", ["geophysics"])),
    }
    report = coverage_report(matches)
    assert "Seismology" in report
    assert "no commercialisation model" in report


def test_every_demo_direction_routes_to_a_modelled_sector():
    """End-to-end: unlabelled corpus in, costable directions out."""
    works = LocalCorpus("data/demo_corpus.json").harvest().works
    available = set(PriorLibrary().sectors())
    found = discover(works)
    matches = {
        d.label: classify(d.to_track(), available) for d in found.tracks
    }
    assert all(m.is_confident for m in matches.values())
