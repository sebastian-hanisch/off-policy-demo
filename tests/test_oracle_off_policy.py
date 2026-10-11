"""Orakel-Tests: der Kern gegen einen unabhängigen Rechenweg (ORACLE-PLAYBOOK §2).

1. IPS ist erwartungstreu: vollständige Aufzählung aller Logs der Länge 1 bis 3 (exakte Bruchrechnung) liefert E[IPS] = V.
2. Gewichtsvarianz = 1/rho - 1 (einstufig) und rho^-H - 1 (mehrstufig, Produkt unabhängiger Gewichte).
3. Doppelt robust ist erwartungstreu bei beliebigem Modell (Enumeration); DM nicht.
4. Servicegrad des Vehikels gegen die exakte Formel der Order-up-to-Regel (unabhängig erzeugte Nachfrage)."""

import itertools
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import nbinom

import ope_constants as C
import ope_estimators as E
import ope_world as W


def tiny_problem():
    """Zwei Kontexte (P = 1/2), zwei Aktionen. Belohnung r(x, a) deterministisch, Protokoll und Regel als Brüche."""
    ctx = [Fraction(1, 2), Fraction(1, 2)]
    p0 = [[Fraction(1, 4), Fraction(3, 4)], [Fraction(3, 5), Fraction(2, 5)]]
    pi = [[Fraction(0), Fraction(1)], [Fraction(1, 2), Fraction(1, 2)]]
    r = [[Fraction(1, 5), Fraction(4, 5)], [Fraction(3, 10), Fraction(1, 10)]]
    q = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(0), Fraction(9, 10)]]                    # absichtlich falsches Modell
    return ctx, p0, pi, r, q


def enumerate_expectation(n, estimator):
    """E[estimator] über alle Logs der Länge n (Kontext, Aktion) als exakter Bruch."""
    ctx, p0, pi, r, q = tiny_problem()
    total = Fraction(0)
    outcomes = [(x, a) for x in range(2) for a in range(2)]
    for combo in itertools.product(outcomes, repeat=n):
        prob = Fraction(1)
        for x, a in combo:
            prob *= ctx[x] * p0[x][a]
        total += prob * estimator(combo)
    return total


def ips_of(combo):
    _, p0, pi, r, _ = tiny_problem()
    return sum(pi[x][a] / p0[x][a] * r[x][a] for x, a in combo) / len(combo)


def dm_of(combo):
    _, _, pi, _, q = tiny_problem()
    return sum(sum(pi[x][b] * q[x][b] for b in range(2)) for x, _ in combo) / len(combo)


def dr_of(combo):
    _, p0, pi, r, q = tiny_problem()
    n = len(combo)
    return dm_of(combo) + sum(pi[x][a] / p0[x][a] * (r[x][a] - q[x][a]) for x, a in combo) / n


def value_exact():
    ctx, _, pi, r, _ = tiny_problem()
    return sum(ctx[x] * sum(pi[x][a] * r[x][a] for a in range(2)) for x in range(2))


def test_ips_is_unbiased_by_exact_enumeration():
    for n in (1, 2, 3):
        assert enumerate_expectation(n, ips_of) == value_exact()


def test_doubly_robust_is_unbiased_with_a_wrong_model_and_dm_is_not():
    v = value_exact()
    for n in (1, 2):
        assert enumerate_expectation(n, dr_of) == v
        assert enumerate_expectation(n, dm_of) != v


def test_weight_variance_is_one_over_rho_minus_one():
    star = np.array([1, 2, 1, 2, 1, 2])
    pi = E.det_policy(star)
    for rho in (0.1, 0.2, 0.5):
        p0 = E.protocol(star, rho)
        w = E.weights(pi, p0).reshape(C.NX, C.NA)
        prob = C.ctx_p()[:, None] * p0
        mean = (prob * w).sum()
        second = (prob * w**2).sum()
        assert mean == pytest.approx(1.0)
        assert second - mean**2 == pytest.approx(1 / rho - 1)
        for horizon in (3, 5):                           # unabhängige Wochen: das Produkt der Gewichte
            assert second**horizon - mean**(2 * horizon) == pytest.approx(rho**-horizon - 1)


def fill_rate_formula(mean, disp, lead, z):
    """Exakter Servicegrad (Typ 2) der Order-up-to-Regel mit täglicher Prüfung und i.i.d. negativ-binomialer Nachfrage."""
    sd = np.sqrt(lead * (mean + mean**2 / disp))
    s = int(np.round(lead * mean + z * sd))

    def excess(n):
        if n == 0:
            return max(0.0, -s)
        nn, mm = n * disp, n * mean
        p = nn / (nn + mm)
        k = np.arange(s, s + 2000)
        return float(nbinom.sf(k, nn, p).sum())

    return 1.0 - (excess(lead) - excess(lead - 1)) / mean


def test_weekly_service_level_matches_the_exact_formula_without_weekday_and_load_effects():
    rng = np.random.default_rng(1)
    worlds, weeks = 1200, 6
    mu = C.site_means()
    total = C.BURN_DAYS + 7 * weeks
    demand = np.empty((worlds, C.N_SITES, total), dtype=np.int32)
    for s in range(C.N_SITES):
        demand[:, s, :] = nbinom.rvs(C.DISP, C.DISP / (C.DISP + mu[s]), size=(worlds, total), random_state=rng)
    for z in (0.5, 1.5):
        fill, cover = W.simulate_weeks(demand, z, 1.0)
        week_demand = demand[:, :, C.BURN_DAYS:].reshape(worlds, C.N_SITES, weeks, 7).sum(axis=3)
        pooled = (fill * week_demand).sum(axis=(0, 2)) / week_demand.sum(axis=(0, 2))      # Quote der Summen
        for s in (0, 3, 7):
            assert pooled[s] == pytest.approx(fill_rate_formula(mu[s], C.DISP, C.LEAD, z), abs=0.005), (z, s)
        assert (cover >= 0).all() and cover.mean() > 0.2
