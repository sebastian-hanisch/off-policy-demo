"""Jede Zahl des README (Abschnitt "Befunde") wird hier nachgerechnet.

Einstufig: 3 000 simulierte Protokolle je Zelle, gerechnet mit denselben Funktionen wie in der App (Seeds fest). Mehrstufig: der
vorgerechnete Zahlenblock `precomputed_multistep.json`."""

import json
from pathlib import Path

import pytest

import ope_constants as C
import ope_evaluation as EV
import ope_multistep as M

ROOT = Path(__file__).resolve().parent.parent
R = 3000


@pytest.fixture(scope="module")
def runs(pool):
    cache = {}

    def get(n, rho_index, model):
        key = (n, rho_index, model)
        if key not in cache:
            cache[key] = EV.summarize(pool, n, rho_index, model, reps=R)
        return cache[key]

    return get


def test_truth_and_gap(pool, runs):
    s = runs(2000, 3, "falsch")
    assert s["value"] == pytest.approx(89.54, abs=0.01)
    assert s["value_const"] == pytest.approx(89.01, abs=0.01)
    assert s["gap"] == pytest.approx(0.54, abs=0.01) and s["const_action"] == 2
    for rho_i, v in {0: 84.64, 3: 85.58, 4: 87.07}.items():
        assert runs(2000, rho_i, "falsch")["value_log"] == pytest.approx(v, abs=0.02)
    r = 100 * pool["r"]
    assert r[0].tolist() == pytest.approx([90.31, 90.94, 89.89], abs=0.02)         # klein, niedrige Last
    assert r[1].tolist() == pytest.approx([73.64, 82.32, 87.55], abs=0.02)         # klein, hohe Last


def test_ips_is_unbiased_and_noisy(runs):
    for rho_i, rmse in {0: 20.0, 1: 8.5, 2: 6.1, 3: 4.0, 4: 2.0}.items():
        s = runs(2000, rho_i, "falsch")["rows"]["ips"]
        assert s["rmse"] == pytest.approx(rmse, rel=0.04), rho_i
        assert abs(s["bias"]) < 3 * s["sd"] / R**0.5 + 0.02, rho_i


def test_snips_is_nearly_as_good_as_the_model_based_estimators(runs):
    for rho_i, rmse in {0: 1.81, 1: 0.79, 2: 0.57, 3: 0.39, 4: 0.25}.items():
        assert runs(2000, rho_i, "falsch")["rows"]["snips"]["rmse"] == pytest.approx(rmse, abs=0.03), rho_i


def test_direct_method_with_the_wrong_model_is_biased_by_more_than_the_gap(runs):
    for rho_i, (bias, rmse) in {0: (-3.21, 3.24), 1: (-2.63, 2.65), 2: (-2.15, 2.17), 3: (-1.54, 1.57), 4: (-0.66, 0.70)}.items():
        s = runs(2000, rho_i, "falsch")["rows"]["dm"]
        assert s["bias"] == pytest.approx(bias, abs=0.05), rho_i
        assert s["rmse"] == pytest.approx(rmse, abs=0.05), rho_i
        assert abs(s["bias"]) > runs(2000, rho_i, "falsch")["gap"]


def test_the_half_good_model_biases_upwards(runs):
    for rho_i, bias in {0: 1.10, 1: 0.73, 2: 0.47, 3: 0.21}.items():
        assert runs(2000, rho_i, "halb")["rows"]["dm"]["bias"] == pytest.approx(bias, abs=0.05), rho_i


def test_doubly_robust_pays_in_variance_where_overlap_is_thin(runs):
    expected = {
        "falsch": {0: 2.95, 1: 1.10, 2: 0.68, 3: 0.41, 4: 0.25},
        "halb": {0: 1.80, 1: 0.78, 2: 0.57, 3: 0.38, 4: 0.25},
        "gut": {0: 1.82, 1: 0.78, 2: 0.56, 3: 0.38, 4: 0.25},
    }
    for model, per in expected.items():
        for rho_i, v in per.items():
            assert runs(2000, rho_i, model)["rows"]["dr"]["rmse"] == pytest.approx(v, abs=0.05), (model, rho_i)
    # bei ρ = 0,01 ist DR mit falschem Modell schlechter als SNIPS, ab ρ = 0,05 gleichauf
    thin = runs(2000, 0, "falsch")["rows"]
    assert thin["dr"]["rmse"] > thin["snips"]["rmse"] + 0.5
    ok = runs(2000, 3, "falsch")["rows"]
    assert ok["dr"]["rmse"] == pytest.approx(ok["snips"]["rmse"], abs=0.05)


def test_dr_equals_dm_when_the_model_uses_all_the_data(runs):
    for rho_i in (0, 3, 4):
        s = runs(2000, rho_i, "voll")["rows"]
        assert s["dr"]["rmse"] == pytest.approx(s["dm"]["rmse"], abs=0.03)


def test_the_decision_test(runs):
    for rho_i, expected in {3: {"dm": 0, "ips": 56, "snips": 100, "dr": 99}, 4: {"dm": 0, "ips": 56, "snips": 100, "dr": 97}}.items():
        rows = runs(2000, rho_i, "falsch")["rows"]
        for k, v in expected.items():
            assert 100 * rows[k]["correct_order"] == pytest.approx(v, abs=3), (rho_i, k)
    assert 100 * runs(2000, 3, "halb")["rows"]["dm"]["correct_order"] == pytest.approx(79, abs=3)
    assert 100 * runs(2000, 4, "halb")["rows"]["dm"]["correct_order"] == pytest.approx(55, abs=4)
    assert 100 * runs(2000, 3, "gut")["rows"]["dm"]["correct_order"] == pytest.approx(100, abs=1)
    assert 100 * runs(2000, 0, "falsch")["rows"]["dm"]["correct_order"] == pytest.approx(100, abs=1)   # das Vorzeichen des Fehlers dreht


def test_data_size_helps_ips_dr_but_not_the_biased_dm(pool, runs):
    expected = {250: (11.3, 1.16, 1.77, 1.22), 500: (8.3, 0.81, 1.67, 0.84), 1000: (5.7, 0.56, 1.61, 0.59), 2000: (4.0, 0.39, 1.57, 0.41),
                5000: (2.6, 0.25, 1.56, 0.26)}
    for n, (ips, snips, dm, dr) in expected.items():
        r = runs(n, 3, "falsch")["rows"]
        assert (r["ips"]["rmse"], r["snips"]["rmse"], r["dm"]["rmse"], r["dr"]["rmse"]) == pytest.approx((ips, snips, dm, dr), rel=0.05)


def test_thin_overlap_with_small_data(runs):
    s = runs(250, 0, "falsch")
    assert 100 * s["no_hit"] == pytest.approx(8.4, abs=1.0)
    assert s["rows"]["dr"]["rmse"] == pytest.approx(8.2, rel=0.08) and s["rows"]["dm"]["rmse"] == pytest.approx(3.4, rel=0.05)
    assert s["rows"]["snips"]["defined"] == pytest.approx(0.916, abs=0.015)
    big = runs(5000, 0, "falsch")["rows"]
    assert big["dr"]["rmse"] == pytest.approx(1.9, rel=0.08) and big["dm"]["rmse"] == pytest.approx(3.2, rel=0.05)


def test_multistep_numbers():
    cells = json.loads(M.OUT.read_text(encoding="utf-8"))["cells"]
    by = {(c["H"], c["rho"]): c for c in cells}
    assert by[(1, 0.5)]["truth"] == pytest.approx(89.55, abs=0.05) and by[(10, 0.5)]["truth"] == pytest.approx(89.27, abs=0.05)
    assert by[(3, 0.5)]["w_var"] == pytest.approx(7.0, rel=0.05) and by[(5, 0.5)]["w_var"] == pytest.approx(31.0, rel=0.05)
    assert 100 * by[(10, 0.5)]["no_hit"] == pytest.approx(13.0, abs=3.0)
    assert 100 * by[(5, 0.2)]["no_hit"] == pytest.approx(55.5, abs=5.0)
    assert 100 * by[(10, 0.2)]["no_hit"] == 100.0
    assert by[(10, 0.5)]["ess"] == pytest.approx(1.9, abs=0.5)
    e5 = by[(5, 0.5)]["ests"]
    assert e5["IPS"]["rmse"] == pytest.approx(11.1, rel=0.1) and e5["PDIS"]["rmse"] == pytest.approx(4.7, rel=0.1)
    assert e5["SNIPS"]["rmse"] == pytest.approx(0.47, abs=0.08) and e5["DR_voll"]["rmse"] == pytest.approx(0.28, abs=0.06)
    assert e5["DM_voll"]["bias"] == pytest.approx(-0.56, abs=0.06) and e5["DM_falsch"]["bias"] == pytest.approx(-1.03, abs=0.08)
    e10 = by[(10, 0.5)]["ests"]
    assert e10["IPS"]["rmse"] == pytest.approx(62.5, rel=0.15) and e10["DR_voll"]["rmse"] == pytest.approx(0.86, abs=0.2)
    assert by[(10, 0.2)]["ests"]["SNIPS"]["defined"] == 0.0


def test_default_seed_protocol(pool):
    r = EV.run_one(pool, C.DEFAULT_N, C.DEFAULT_RHO_INDEX, C.MODELS[C.DEFAULT_MODEL_INDEX], C.DEFAULT_SEED)
    assert r["n"].sum() == C.DEFAULT_N
    assert r["hits"] == pytest.approx(400, abs=50)


def test_default_seed_values(pool):
    r = EV.run_one(pool, C.DEFAULT_N, C.DEFAULT_RHO_INDEX, "falsch", C.DEFAULT_SEED)
    v = 100 * EV.value(pool, EV.target_policy(pool))
    err = {k: 100 * x - v for k, x in r["est"].items()}
    assert (err["ips"], err["snips"], err["dm"], err["dr"]) == pytest.approx((3.77, -0.03, -1.59, 0.07), abs=0.03)
    assert r["hits"] == 417
