"""Pool, Wahrheit, Protokolle und Zusammenfassungen: von Hand und als Eigenschaften."""

import numpy as np
import pytest

import ope_constants as C
import ope_estimators as E
import ope_evaluation as EV


def synthetic_pool(r_table):
    rewards = np.repeat(np.asarray(r_table, dtype=float).reshape(C.NC, 1), 50, axis=1)
    return {"rewards": rewards, "r": np.asarray(r_table, dtype=float)}


def test_value_by_hand():
    r = np.zeros((C.NX, C.NA))
    r[:, 0], r[:, 1], r[:, 2] = 0.5, 0.8, 0.6
    pool = synthetic_pool(r)
    assert EV.value(pool, E.det_policy(np.array([1] * 6))) == pytest.approx(0.8)
    mix = np.zeros((C.NX, C.NA))
    mix[:, 1], mix[:, 2] = 0.5, 0.5
    assert EV.value(pool, mix) == pytest.approx(0.7)
    assert EV.best_actions(pool).tolist() == [1] * 6
    assert EV.best_constant(pool) == (1, pytest.approx(0.8))


def test_contexts_and_classes_sum_to_one():
    assert C.ctx_p().sum() == pytest.approx(1.0)
    assert C.ctx_p().shape == (C.NX,)
    assert sum(C.CLASS_P) == pytest.approx(1.0) and sum(C.LOAD_P) == pytest.approx(1.0)


def test_pool_truth_matches_the_documented_structure(pool):
    r = pool["r"]
    assert r.shape == (C.NX, C.NA)
    assert EV.best_actions(pool).tolist() == [1, 2, 1, 2, 1, 2]       # niedrige Last z = 1,0, hohe Last z = 1,5
    assert (pool["fill"] > 0.7).all() and (pool["cover"] > 0).all()
    # höhere Sicherheit: mehr Servicegrad und mehr Reichweite
    assert (np.diff(pool["fill"], axis=1) > 0).all() and (np.diff(pool["cover"], axis=1) > 0).all()
    assert pool["r"] == pytest.approx(pool["fill"] - C.COST * pool["cover"], abs=0.003)


def test_sample_logs_have_fixed_class_sizes_and_the_right_cell_shares(pool):
    star = EV.best_actions(pool)
    p0 = E.protocol(star, 0.2)
    n, s = EV.sample_logs(pool, p0, 2000, 300, 5)
    assert (n.sum(axis=1) == 2000).all()
    per_class = n.reshape(300, 3, 2, 3).sum(axis=(2, 3))
    assert (per_class == np.array([500, 1000, 500])).all()
    share_target = n.reshape(300, C.NX, C.NA)[:, np.arange(C.NX), star].sum(axis=(0)) / n.reshape(300, C.NX, C.NA).sum(axis=(0, 2))
    assert share_target == pytest.approx([0.2] * 6, abs=0.03)
    assert (s >= 0).all() and (s <= n).all()                       # Belohnungen liegen in [0, 1]


def test_sample_logs_are_reproducible_and_independent_of_the_number_of_protocols(pool):
    p0 = E.protocol(EV.best_actions(pool), 0.2)
    a = EV.sample_logs(pool, p0, 500, 100, 9)
    b = EV.sample_logs(pool, p0, 500, 250, 9)
    assert np.array_equal(a[0], b[0][:100]) and np.array_equal(a[1], b[1][:100])
    assert not np.array_equal(a[0], EV.sample_logs(pool, p0, 500, 100, 10)[0])


def test_summaries_expose_the_expected_structure(pool):
    s = EV.summarize(pool, 1000, 3, "falsch", reps=200)
    assert set(s["rows"]) == set(C.ESTIMATORS)
    assert s["value"] == pytest.approx(100 * EV.value(pool, EV.target_policy(pool)))
    assert s["value"] > s["value_const"] > s["value_log"]
    assert s["gap"] == pytest.approx(s["value"] - s["value_const"])
    assert s["weight_var"] == pytest.approx(1 / 0.2 - 1)


def test_run_one_is_reproducible(pool):
    a = EV.run_one(pool, 1000, 3, "falsch", 35)
    b = EV.run_one(pool, 1000, 3, "falsch", 35)
    assert a["est"] == b["est"] and (a["n"] == b["n"]).all() and a["n"].sum() == 1000


def test_the_count_chart_marks_the_target_action_of_the_new_rule(pool):
    import ope_visualization as V
    star = EV.best_actions(pool)
    n = np.arange(C.NC, dtype=float).reshape(C.NX, C.NA)
    fig = V.build_counts(n, star)
    text = fig.data[0].text
    for x in range(C.NX):
        marked = [a for a in range(C.NA) if "◆" in text[x][a]]
        assert marked == [int(star[x])]
