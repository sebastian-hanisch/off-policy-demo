"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ope_constants as C  # noqa: E402
import ope_evaluation as EV  # noqa: E402


@pytest.fixture(scope="session")
def pool():
    return EV.load_pool()


@pytest.fixture
def hand():
    """Mini-Instanz: vier Zeilen in zwei Kontexten und zwei genutzten Aktionen.

    Zeilen (x, a, r): (0, 1, 1,0), (0, 0, 0,4), (1, 0, 0,6), (1, 1, 0,2). Neue Regel: x0 -> a1, x1 -> a0. Protokoll p0: x0 = (0,5; 0,5; 0),
    x1 = (0,8; 0,2; 0). Modell q: x0 = (0,3; 0,8; 0), x1 = (0,5; 0,1; 0).
    Handrechnung: Gewichte 2, 0, 1,25, 0.
      IPS = (2 * 1,0 + 1,25 * 0,6) / 4 = 0,6875;  SNIPS = 2,75 / 3,25 = 0,846154
      DM = mittleres Modell unter der Regel = (0,8 + 0,8 + 0,5 + 0,5) / 4 = 0,65
      DR = 0,65 + (2 * (1,0 - 0,8) + 1,25 * (0,6 - 0,5)) / 4 = 0,65 + 0,13125 = 0,78125"""
    x = np.array([0, 0, 1, 1])
    a = np.array([1, 0, 0, 1])
    r = np.array([1.0, 0.4, 0.6, 0.2])
    pi = np.zeros((C.NX, C.NA))
    pi[0, 1] = 1.0
    pi[1, 0] = 1.0
    pi[2:, 0] = 1.0
    p0 = np.full((C.NX, C.NA), 1.0 / 3.0)
    p0[0] = [0.5, 0.5, 0.0]
    p0[1] = [0.8, 0.2, 0.0]
    q = np.zeros((C.NX, C.NA))
    q[0] = [0.3, 0.8, 0.0]
    q[1] = [0.5, 0.1, 0.0]
    return {"x": x, "a": a, "r": r, "pi": pi, "p0": p0, "q": q}
