"""Mehrstufig: Konsistenz mit der einstufigen Fassung, Gewichte, Vehikel über die Wochengrenze."""

import numpy as np
import pytest

import ope_constants as C
import ope_estimators as E
import ope_evaluation as EV
import ope_multistep as M


@pytest.fixture(scope="module")
def star(pool):
    return EV.best_actions(pool)


def test_trajectories_have_the_right_shapes_and_valid_values(star):
    x, a, r = M.simulate_trajectories(20, 3, ("protocol", E.protocol(star, 0.5)), 7)
    assert x.shape == a.shape == r.shape == (20 * C.N_SITES, 3)
    assert set(np.unique(x)) <= set(range(C.NX)) and set(np.unique(a)) <= set(range(C.NA))
    assert ((r > 0.3) & (r < 1.0)).all()
    xs, as_, _ = M.simulate_trajectories(20, 3, ("target", star), 7)
    assert (as_ == star[xs]).all()


def test_trajectories_are_reproducible(star):
    a = M.simulate_trajectories(10, 2, ("protocol", E.protocol(star, 0.5)), 3)
    b = M.simulate_trajectories(10, 2, ("protocol", E.protocol(star, 0.5)), 3)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_one_step_multistep_estimators_equal_the_cell_estimators(star):
    p0 = E.protocol(star, 0.5)
    x, a, r = M.simulate_trajectories(30 * 5, 1, ("protocol", p0), 11)
    nds = 5
    ds = np.repeat(np.arange(nds), len(x) // nds)
    res, _ = M.estimators_multi(x, a, r, ds, nds, p0, star, 1)
    n = np.zeros((nds, C.NC))
    s = np.zeros((nds, C.NC))
    for d in range(nds):
        sel = ds == d
        cell = x[sel, 0] * C.NA + a[sel, 0]
        n[d] = np.bincount(cell, minlength=C.NC)
        s[d] = np.bincount(cell, weights=r[sel, 0], minlength=C.NC)
    ref = E.estimators(n, s, E.fit_model(n, s, "voll"), E.det_policy(star), p0)
    assert res["IPS"] == pytest.approx(ref["ips"], abs=1e-10)
    assert res["SNIPS"] == pytest.approx(ref["snips"], abs=1e-10)
    assert res["PDIS"] == pytest.approx(ref["ips"], abs=1e-10)
    assert res["DM_voll"] == pytest.approx(ref["dm"], abs=1e-10)
    assert res["DR_voll"] == pytest.approx(ref["dr"], abs=1e-10)


def test_stock_carries_over_the_week_boundary():
    """Ein Wechsel von z = 1,5 auf z = 0,5 in Woche 2 lässt den hohen Bestand noch eine Weile wirken: die Reichweite der Woche
    nach dem Wechsel ist größer als in einer Welt, die dauerhaft z = 0,5 fährt."""
    import ope_world as W
    demand = W.generate_demand(300, 1.0, 2, 5)
    _, cover_switch = W.simulate_weeks(demand, np.array([1.5, 0.5])[None, None, :], 1.0)
    _, cover_const = W.simulate_weeks(demand, 0.5, 1.0)
    assert cover_switch[:, :, 1].mean() > cover_const[:, :, 1].mean() + 0.05


def test_stored_weight_statistics_match_the_formulas():
    import json
    cells = json.loads((M.OUT).read_text(encoding="utf-8"))["cells"]
    for c in cells:
        assert c["w_mean"] == pytest.approx(1.0, abs=0.15 if c["hit_share"] > 0 else 1.0) or c["no_hit"] > 0
        if c["no_hit"] < 0.01:
            assert c["w_var"] == pytest.approx(c["w_var_formula"], rel=0.1)
        assert c["hit_share"] == pytest.approx(c["hit_share_formula"], abs=max(0.002, 0.2 * c["hit_share_formula"]))
