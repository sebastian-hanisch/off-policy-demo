"""Schätzer: von Hand an der Mini-Instanz, Zell-Fassung gegen Zeilen-Fassung, Protokoll und Modelle."""

import numpy as np
import pytest

import ope_constants as C
import ope_estimators as E


def cell_arrays(x, a, r):
    n = np.zeros((1, C.NC))
    s = np.zeros((1, C.NC))
    for xi, ai, ri in zip(x, a, r):
        n[0, xi * C.NA + ai] += 1
        s[0, xi * C.NA + ai] += ri
    return n, s


def test_row_estimators_by_hand(hand):
    res = E.row_estimators(hand["x"], hand["a"], hand["r"], hand["q"], hand["pi"], hand["p0"])
    assert res["ips"] == pytest.approx(0.6875)
    assert res["snips"] == pytest.approx(2.75 / 3.25)
    assert res["dm"] == pytest.approx(0.65)
    assert res["dr"] == pytest.approx(0.78125)


def test_cell_estimators_equal_the_row_estimators(hand):
    n, s = cell_arrays(hand["x"], hand["a"], hand["r"])
    out = E.estimators(n, s, hand["q"].ravel()[None, :], hand["pi"], hand["p0"])
    ref = E.row_estimators(hand["x"], hand["a"], hand["r"], hand["q"], hand["pi"], hand["p0"])
    for k in ("ips", "snips", "dm", "dr"):
        assert out[k][0] == pytest.approx(ref[k], abs=1e-12), k


def test_cell_and_row_estimators_agree_on_random_logs():
    rng = np.random.default_rng(3)
    star = np.array([1, 2, 1, 2, 1, 2])
    pi, p0 = E.det_policy(star), E.protocol(star, 0.3)
    for _ in range(5):
        x = rng.integers(0, C.NX, 400)
        a = np.array([rng.choice(3, p=p0[xi]) for xi in x])
        r = rng.normal(0.9, 0.05, 400)
        q = rng.uniform(0.8, 1.0, (C.NX, C.NA))
        n, s = cell_arrays(x, a, r)
        out = E.estimators(n, s, q.ravel()[None, :], pi, p0)
        ref = E.row_estimators(x, a, r, q, pi, p0)
        for k in ("ips", "snips", "dm", "dr"):
            assert out[k][0] == pytest.approx(ref[k], abs=1e-10), k


def test_snips_is_undefined_without_a_matching_row(hand):
    n, s = cell_arrays([0, 0], [0, 0], [0.5, 0.7])                # nie die Aktion der neuen Regel
    out = E.estimators(n, s, hand["q"].ravel()[None, :], hand["pi"], hand["p0"])
    assert np.isnan(out["snips"][0]) and out["ips"][0] == 0.0 and out["_hits"][0] == 0


def test_protocol_rows_sum_to_one_and_hit_the_target_with_probability_rho():
    star = np.array([1, 2, 1, 2, 1, 2])
    for rho in (0.01, 0.2, 0.5):
        p = E.protocol(star, rho)
        assert p.sum(axis=1) == pytest.approx(1.0)
        assert p[np.arange(C.NX), star] == pytest.approx(rho)
        assert (p >= 0).all()


def test_protocol_leans_to_high_z_under_high_load_and_low_z_under_low_load():
    star = np.array([1, 2, 1, 2, 1, 2])
    p = E.protocol(star, 0.2)
    assert p[0, 0] > p[0, 2]                              # niedrige Last: eher z = 0,5 als 1,5
    assert p[1, 1] > p[1, 0]                              # hohe Last, Zielaktion 1,5: der Rest bevorzugt 1,0 vor 0,5


def test_weights_are_ratio_where_the_target_chooses_and_zero_elsewhere(hand):
    w = E.weights(hand["pi"], hand["p0"]).reshape(C.NX, C.NA)
    assert w[0, 1] == pytest.approx(2.0) and w[1, 0] == pytest.approx(1.25)
    assert w[0, 0] == 0.0 and w[1, 1] == 0.0


def test_saturated_design_reproduces_the_cell_means():
    rng = np.random.default_rng(4)
    n = rng.integers(1, 30, (3, C.NC)).astype(float)
    s = n * rng.uniform(0.7, 1.0, (3, C.NC))
    q = E.fit_table(n, s, np.eye(C.NC))
    assert q == pytest.approx(s / n, abs=1e-4)


def test_models_are_nested_in_quality():
    """Das gute Modell enthält das halbgute, dieses das falsche: der gewichtete Anpassungsfehler fällt in dieser Reihenfolge."""
    rng = np.random.default_rng(5)
    n = rng.integers(20, 60, (1, C.NC)).astype(float)
    truth = rng.uniform(0.7, 1.0, (1, C.NC))
    s = n * truth
    errs = []
    for m in ("falsch", "halb", "gut"):
        q = E.fit_model(n, s, m)
        errs.append(float((n * (q - truth) ** 2).sum()))
    assert errs[0] >= errs[1] >= errs[2]


def test_dr_equals_dm_when_the_model_is_saturated(hand):
    rng = np.random.default_rng(6)
    star = np.array([1, 2, 1, 2, 1, 2])
    pi, p0 = E.det_policy(star), E.protocol(star, 0.3)
    n = rng.integers(5, 40, (2, C.NC)).astype(float)
    s = n * rng.uniform(0.7, 1.0, (2, C.NC))
    q = E.fit_table(n, s, np.eye(C.NC))
    out = E.estimators(n, s, q, pi, p0)
    assert out["dr"] == pytest.approx(out["dm"], abs=1e-5)
