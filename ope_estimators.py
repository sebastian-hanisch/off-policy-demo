"""Schätzer für den Wert einer neuen Regel aus Logdaten, vektorisiert über viele Protokolle.

Kontexte x = Größenklasse * 2 + Last (6), Aktionen a = Sicherheitsfaktor (3), Zellen c = x * 3 + a (18). Bei diskretem Kontext
hängen alle Schätzer nur von den Zell-Zählern n_c und den Zell-Summen s_c der Belohnung ab; das ist keine Näherung, sondern eine
exakte Umformung der Zeilensummen (die Zeilen-Fassung `row_estimators` ist die Referenz in den Tests).

    IPS    = 1/N  sum_i w_i r_i                      w = pi(a|x) / p0(a|x)
    SNIPS  = sum_i w_i r_i / sum_i w_i
    DM     = 1/N  sum_i sum_a pi(a|x_i) q(x_i, a)
    DR     = DM + 1/N sum_i w_i (r_i - q(x_i, a_i))"""

import numpy as np

import ope_constants as C

NX, NA, NC = C.NX, C.NA, C.NC

_cls = np.repeat(np.arange(NX) // 2, NA)
_lst = np.repeat(np.arange(NX) % 2, NA)
_act = np.tile(np.arange(NA), NX)
_z = np.array(C.ACTIONS)[_act]


def _onehot(v, k):
    return (v[:, None] == np.arange(k)[None, :]).astype(float)


# Designmatrizen der Belohnungsmodelle über die 18 Zellen
Z_GOOD = np.hstack([_onehot(_lst * 3 + _act, 6), _onehot(_cls, 3)[:, [0, 2]]])                        # Last x Aktion voll + Größe additiv
Z_WRONG = np.hstack([np.ones((NC, 1)), _onehot(_cls, 3)[:, [0, 2]], _lst[:, None], _onehot(_act, 3)[:, [1, 2]]])   # alles additiv
Z_HALF = np.hstack([np.ones((NC, 1)), _onehot(_cls, 3)[:, [0, 2]], _lst[:, None], _z[:, None], (_lst * _z)[:, None]])  # Last x z linear
DESIGNS = {"falsch": Z_WRONG, "halb": Z_HALF, "gut": Z_GOOD}


def fit_table(n, s, design, lam=1e-6):
    """Gewichtete Kleinste-Quadrate-Anpassung über die Zellen. n, s: (R, 18) -> Vorhersage je Zelle (R, 18)."""
    ztz = np.einsum("rc,cp,cq->rpq", n, design, design) + lam * np.eye(design.shape[1])[None]
    zty = np.einsum("cp,rc->rp", design, s)
    beta = np.linalg.solve(ztz, zty[:, :, None])[:, :, 0]
    return beta @ design.T


def fit_model(n, s, model, smooth=1.0):
    """Belohnungsmodell `model` aus Zählern und Summen. 'voll' = Zellmittel, mit `smooth` Pseudozeilen zum guten Modell geglättet
    (damit leere Zellen definiert sind)."""
    if model == "voll":
        return (s + smooth * fit_table(n, s, Z_GOOD)) / (n + smooth)
    return fit_table(n, s, DESIGNS[model])


def weights(pi, p0):
    """Gewicht je Zelle (18,): pi(a|x) / p0(a|x); 0 dort, wo pi nichts wählt."""
    pif, p0f = pi.ravel(), p0.ravel()
    return np.where(pif > 0, pif / np.where(p0f > 0, p0f, 1.0), 0.0)


def estimators(n, s, q, pi, p0):
    """n, s, q: (R, 18); pi, p0: (6, 3). Rückgabe: dict Name -> (R,); SNIPS ist NaN, wenn kein Datenpunkt zur Zielregel gehört."""
    total = n.sum(axis=1)
    w = weights(pi, p0)
    out = {"ips": (s * w).sum(axis=1) / total}
    denom = (n * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["snips"] = np.where(denom > 0, (s * w).sum(axis=1) / denom, np.nan)
    dm = ((n.reshape(-1, NX, NA).sum(axis=2) / total[:, None]) * (q.reshape(-1, NX, NA) * pi[None]).sum(axis=2)).sum(axis=1)
    out["dm"] = dm
    out["dr"] = dm + ((s - n * q) * w).sum(axis=1) / total
    out["_hits"] = (n * (w > 0)).sum(axis=1)
    out["_ess"] = np.where((n * w**2).sum(axis=1) > 0, denom**2 / np.maximum((n * w**2).sum(axis=1), 1e-300), 0.0)
    return out


def row_estimators(x, a, r, q_table, pi, p0):
    """Zeilen-Fassung (Referenz): x, a Indexvektoren, r Belohnung, q_table (6, 3)."""
    w = pi[x, a] / p0[x, a]
    dm_rows = (q_table[x] * pi[x]).sum(axis=1)
    return {"ips": float(np.mean(w * r)), "snips": float(np.sum(w * r) / np.sum(w)), "dm": float(np.mean(dm_rows)),
            "dr": float(np.mean(dm_rows + w * (r - q_table[x, a])))}


def protocol(target_actions, rho, tau=C.TAU):
    """Protokoll (6, 3): Zielaktion je Kontext mit Wahrscheinlichkeit rho; der Rest verteilt sich lastabhängig: bei hoher Last
    bevorzugt höheres z (Gewichte exp(+tau z)), bei niedriger Last niedrigeres z (exp(-tau z))."""
    p = np.zeros((NX, NA))
    for x in range(NX):
        t = target_actions[x]
        sign = 1.0 if x % 2 == 1 else -1.0
        wgt = np.exp(sign * tau * np.array(C.ACTIONS))
        wgt[t] = 0.0
        p[x] = (1.0 - rho) * wgt / wgt.sum()
        p[x, t] = rho
    return p


def det_policy(actions_per_x):
    pi = np.zeros((NX, NA))
    pi[np.arange(NX), actions_per_x] = 1.0
    return pi
