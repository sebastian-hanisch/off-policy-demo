"""Konstanten: Vehikel, Entscheidungsproblem, Regler-Grenzen, Farben, Presets. Das Vehikel (Standorte mit Order-up-to-Bestand) ist
das der ganzen Linie."""

import numpy as np

# Vehikel: Standorte mit täglicher Nachbestellung (Order-up-to), Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt.
BASE_MEAN = 20.0
SIZES = (0.6, 0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.5)          # acht Standorte; Größenklassen klein / mittel / groß
DOW = (1.15, 1.10, 1.00, 1.00, 1.05, 0.90, 0.80)
DISP = 5
LEAD = 2
BURN_DAYS = 28
N_SITES = 8
CLASS_OF_SITE = (0, 0, 1, 1, 1, 1, 2, 2)
CLASS_NAMES = ("klein", "mittel", "groß")
CLASS_P = (0.25, 0.5, 0.25)                                 # Anteil der Entscheidungen je Größenklasse (2 / 4 / 2 Standorte)

# Stück 12: Off-Policy-Evaluation
ACTIONS = (0.5, 1.0, 1.5)                                   # Aktion = Sicherheitsfaktor z der Woche
LOADS = (0.8, 1.2)                                          # Kontext Wochenlast: niedrig / hoch (Faktor auf die Nachfrage)
LOAD_NAMES = ("niedrig", "hoch")
LOAD_P = (0.5, 0.5)
NX, NA = 6, 3                                               # Kontexte x = Größenklasse * 2 + Last; Aktionen
NC = NX * NA
COST = 0.05                                                 # Belohnung = Servicegrad - COST * Reichweite (Tage)
POOL_WORLDS = 5000                                          # Welten je (Last, Aktion) im Pool (8 Standorte je Welt)
TAU = 1.0                                                   # wie stark das Protokoll Last und Aktion koppelt

N_MIN, N_MAX, DEFAULT_N, N_STEP = 250, 5000, 2000, 250     # Zahl der Logzeilen (Entscheidungen)
RHO_CHOICES = (0.01, 0.05, 0.1, 0.2, 0.5)                   # Wahrscheinlichkeit, mit der das Protokoll die Zielaktion wählt
DEFAULT_RHO_INDEX = 3
MODELS = ("falsch", "halb", "gut", "voll")
MODEL_LABELS = {
    "falsch": "Falsch: Last und Aktion nur additiv",
    "halb": "Halb gut: Last × z als lineare Wechselwirkung",
    "gut": "Gut: Last × Aktion vollständig",
    "voll": "Zellmittel (jede Zelle ihr eigener Wert)",
}
DEFAULT_MODEL_INDEX = 0
SEED_MAX, DEFAULT_SEED = 999999, 35
WORLDS_LIVE = 1000                                          # simulierte Protokolle (Wiederholungen) in der Live-Auswertung

ESTIMATORS = ("ips", "snips", "dm", "dr")
ESTIMATOR_LABELS = {
    "ips": "IPS (Gewichtung mit der Wahrscheinlichkeit)",
    "snips": "SNIPS (normierte Gewichtung)",
    "dm": "Direkte Methode (Modell der Belohnung)",
    "dr": "Doppelt robust (Modell + gewichtete Korrektur)",
}
ESTIMATOR_SHORT = {"ips": "IPS", "snips": "SNIPS", "dm": "Direkte Methode", "dr": "Doppelt robust"}
ESTIMATOR_COLORS = {"ips": "#c0392b", "snips": "#d95f02", "dm": "#4c78a8", "dr": "#2F6B65"}

COLOR_TRUTH = "#d95f02"
COLOR_LOG = "#7f7f7f"

PRESETS = {
    "Gutes Protokoll, falsches Modell (Voreinstellung)": {"n": DEFAULT_N, "rho": 3, "model": 0, "seed": DEFAULT_SEED},
    "Knappe Überlappung": {"n": 1000, "rho": 0, "model": 0, "seed": DEFAULT_SEED},
    "Gutes Modell": {"n": DEFAULT_N, "rho": 3, "model": 2, "seed": DEFAULT_SEED},
    "Reichlich Überlappung": {"n": DEFAULT_N, "rho": 4, "model": 0, "seed": DEFAULT_SEED},
}
PRESET_ORDER = tuple(PRESETS)
PRESET_HELP = {
    "Gutes Protokoll, falsches Modell (Voreinstellung)": "2 000 Zeilen; das Protokoll wählt die Zielaktion in 20 % der Fälle; das Belohnungsmodell kennt die Wechselwirkung von Last und Aktion nicht.",
    "Knappe Überlappung": "1 000 Zeilen, Zielaktion nur in 1 % der Fälle: kaum ein Datenpunkt gehört zur neuen Regel.",
    "Gutes Modell": "Dasselbe Protokoll, aber das Modell kennt die Wechselwirkung: wozu dann noch die Korrektur?",
    "Reichlich Überlappung": "Zielaktion in der Hälfte der Fälle: die Gewichtung wird zahm.",
}


def class_p():
    return np.array(CLASS_P, dtype=float)


def ctx_p():
    """Wahrscheinlichkeit je Kontext x = Klasse * 2 + Last: (6,)."""
    return np.outer(class_p(), np.array(LOAD_P, dtype=float)).ravel()


def site_means():
    return BASE_MEAN * np.array(SIZES, dtype=float)


def dow_factors():
    d = np.array(DOW, dtype=float)
    return d / d.mean()


def fmt_pct(x, digits=1):
    """Anteil (0..1) als Prozentzahl mit Dezimalkomma, z. B. 0,934 -> '93,4 %'."""
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def fmt_pp(x, digits=2):
    """Differenz in Prozentpunkten (aus Anteilen 0..1) mit Vorzeichen und Dezimalkomma."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def fmt_pkt(x, digits=1):
    """Kurzform für enge Kacheln: '+4,0 Pkt.' (Differenz in Prozentpunkten, aus Anteilen 0..1)."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Pkt."


def fmt_num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(x):,}".replace(",", " ")
