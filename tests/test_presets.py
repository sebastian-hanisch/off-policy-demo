"""Presets: Vollständigkeit, gültige Werte, Einrasten der Permalink-Werte."""

import ope_constants as C
import ope_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_grid():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and (preset["n"] - C.N_MIN) % C.N_STEP == 0
        assert 0 <= preset["rho"] < len(C.RHO_CHOICES)
        assert 0 <= preset["model"] < len(C.MODELS)
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_labels_are_complete():
    assert set(C.MODEL_LABELS) == set(C.MODELS)
    assert set(C.ESTIMATOR_LABELS) == set(C.ESTIMATOR_SHORT) == set(C.ESTIMATOR_COLORS) == set(C.ESTIMATORS)
    assert len(C.CLASS_OF_SITE) == C.N_SITES == len(C.SIZES)


def test_problem_dimensions_are_consistent():
    assert C.NC == C.NX * C.NA and C.NX == 3 * len(C.LOADS)
    assert max(C.CLASS_OF_SITE) + 1 == len(C.CLASS_NAMES) == len(C.CLASS_P)


def test_permalink_values_are_clamped_and_snapped():
    assert P.clamp_to_spec(P.SETTING_SPECS["n_slider"], 1100) == 1000
    assert P.clamp_to_spec(P.SETTING_SPECS["n_slider"], 99999) == 5000
    assert P.clamp_to_spec(P.SETTING_SPECS["rho_radio"], 9) == 4
    assert P.clamp_to_spec(P.SETTING_SPECS["model_radio"], -1) == 0
