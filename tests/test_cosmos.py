"""Checks that the cosmos simulations reproduce known results, not just that they run."""
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from cosmos.common import Cosmo, measurements  # noqa: E402


def test_background_matches_planck():
    c = Cosmo(h=0.6736, omega_b=0.02237, omega_bc=0.14237)
    assert 100 * c.theta_star() == pytest.approx(1.0411, rel=3e-4)
    assert c.sound_horizon(1059.94) == pytest.approx(147.09, rel=2e-3)
    assert c.age_gyr() == pytest.approx(13.80, rel=3e-3)


def test_every_measurement_has_provenance():
    def leaves(node):
        if isinstance(node, dict):
            if "provenance" in node:
                yield node
            else:
                for v in node.values():
                    yield from leaves(v)
        elif isinstance(node, list):
            for v in node:
                yield from leaves(v)

    entries = list(leaves(measurements()))
    assert len(entries) > 50
    for e in entries:
        if e["provenance"] == "sourced":
            assert e["ref"] and e["checked"] in {"document", "search-summary", "recalled"}
        else:
            assert e["provenance"] == "assumed" and e["rationale"]


def test_dark_energy_fit_reproduces_desi():
    from cosmos.p02_dark_energy import fit
    _, chi_l = fit(False, use_cmb=False, use_sn=False)
    _, chi_w = fit(True, use_cmb=False, use_sn=False)
    published = measurements()["dark_energy_published"]["desi_only"]["value"]
    assert chi_w - chi_l == pytest.approx(published, abs=0.4)


def test_page_curve_is_symmetric_and_below_thermal():
    from cosmos.p06_black_hole_information import page_entropy
    assert page_entropy(4, 64) == pytest.approx(page_entropy(64, 4))
    assert page_entropy(16, 16) < np.log(16)


def test_neutrino_minimum_sums():
    from cosmos.p10_neutrino_mass import masses
    assert sum(masses(0.0, "normal")) == pytest.approx(0.059, abs=0.002)
    assert sum(masses(0.0, "inverted")) == pytest.approx(0.099, abs=0.002)


def test_leptogenesis_forgets_initial_conditions_in_strong_washout():
    from cosmos.p05_baryogenesis import efficiency
    assert efficiency(46.0, True)[1] == pytest.approx(efficiency(46.0, False)[1], rel=0.02)
