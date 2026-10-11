"""Mehrstufig: eine Trajektorie ist ein Standort über H Wochen hintereinander (Bestand und Pipeline tragen über die Wochengrenze).

Je Woche: Last unabhängig 50/50, Aktion aus dem Protokoll (Zielaktion mit Wahrscheinlichkeit rho). Die Simulation läuft als ein
Vehikel-Lauf über 7 H Tage mit wochenweise wechselndem z und wechselnder Last: der Bestand trägt wirklich über. Wert einer Regel =
mittlere Wochenbelohnung der Trajektorie. Wahrheit: die Zielregel direkt gefahren (auf-Policy). Schätzer: IPS mit dem Produkt der
Gewichte, SNIPS, PDIS (je Schritt), direkte Methode (einstufiges Modell, H-mal angewendet) und doppelt robust (sequentiell).

Das Ergebnis ist ein Zahlenblock, der einmal vorgerechnet in `precomputed_multistep.json` liegt (`python generate_precomputed.py
--multi`); die App zeigt ihn an."""

import json
from pathlib import Path

import numpy as np

import ope_constants as C
import ope_estimators as E
import ope_evaluation as EV
import ope_world as W
from ope_rng import uniform_block

OUT = Path(__file__).resolve().parent / "precomputed_multistep.json"
HORIZONS = (1, 3, 5, 10)
RHOS = (0.8, 0.5, 0.2)
TRAJ_PER_DATASET = 2000                  # Trajektorien je Datensatz (250 Welten x 8 Standorte)
DATASETS = 200
WORLDS_PER_DATASET = TRAJ_PER_DATASET // C.N_SITES
CHUNK_DATASETS = 20


def simulate_trajectories(worlds, horizon, policy, seed):
    """Trajektorien aus `worlds` Welten (je 8 Standorte). policy: ('target', best_actions) oder ('protocol', p0).
    Rückgabe: x, a, r je (worlds * 8, horizon)."""
    classes = np.tile(np.array(C.CLASS_OF_SITE), (worlds, 1))
    u_load = uniform_block(seed, worlds * C.N_SITES * horizon).reshape(worlds, C.N_SITES, horizon)
    load_idx = (u_load >= C.LOAD_P[0]).astype(int)
    x = classes[:, :, None] * 2 + load_idx
    if policy[0] == "target":
        a = policy[1][x]
    else:
        u = uniform_block(seed + 17, worlds * C.N_SITES * horizon).reshape(worlds, C.N_SITES, horizon)
        cum = np.cumsum(policy[1][x], axis=3)
        a = (u[..., None] > cum).sum(axis=3).clip(0, C.NA - 1)
    z = np.array(C.ACTIONS)[a]
    load = np.array(C.LOADS)[load_idx]
    demand = W.generate_demand(worlds, load, horizon, seed + 29)
    fill, cover = W.simulate_weeks(demand, z, load)
    r = W.rewards(fill, cover)
    return x.reshape(-1, horizon), a.reshape(-1, horizon), r.reshape(-1, horizon)


def estimators_multi(x, a, r, ds, nds, p0, star, horizon):
    """Schätzer je Datensatz. x, a, r: (T, H); ds: Datensatz-Index je Trajektorie (T,)."""
    cell = x * C.NA + a
    key = (ds[:, None] * C.NC + cell).ravel()
    n = np.bincount(key, minlength=nds * C.NC).reshape(nds, C.NC).astype(float)
    s = np.bincount(key, weights=r.ravel(), minlength=nds * C.NC).reshape(nds, C.NC)
    models = {"voll": E.fit_model(n, s, "voll"), "falsch": E.fit_model(n, s, "falsch")}
    match = a == star[x]
    wstep = np.where(match, 1.0 / p0[x, a], 0.0)
    cumw = np.cumprod(wstep, axis=1)
    g = r.sum(axis=1)
    cnt = np.bincount(ds, minlength=nds).astype(float)

    def dmean(v):
        return np.bincount(ds, weights=v, minlength=nds) / cnt

    wt = cumw[:, -1]
    res = {"IPS": dmean(wt * g) / horizon}
    sw = np.bincount(ds, weights=wt, minlength=nds)
    with np.errstate(invalid="ignore", divide="ignore"):
        res["SNIPS"] = np.bincount(ds, weights=wt * g, minlength=nds) / sw / horizon
    res["PDIS"] = dmean((cumw * r).sum(axis=1)) / horizon
    for name, qt in models.items():
        qd = qt[ds]
        q_xa = np.take_along_axis(qd, cell, axis=1)
        q_xpi = np.take_along_axis(qd, x * C.NA + star[x], axis=1)
        m_d = np.bincount(ds, weights=q_xpi.sum(axis=1), minlength=nds) / (cnt * horizon)
        m_t = m_d[ds]
        res[f"DM_{name}"] = m_d
        dr = q_xpi[:, 0] + (horizon - 1) * m_t
        for h in range(horizon):
            qq = q_xa[:, h] + (m_t - q_xpi[:, h + 1] if h < horizon - 1 else 0.0)
            dr = dr + cumw[:, h] * (r[:, h] - qq)
        res[f"DR_{name}"] = dmean(dr) / horizon
    w2 = np.bincount(ds, weights=wt**2, minlength=nds)
    with np.errstate(invalid="ignore", divide="ignore"):
        res["ESS"] = np.where(w2 > 0, sw**2 / w2, 0.0)
    res["matched"] = np.bincount(ds, weights=(wt > 0).astype(float), minlength=nds)
    return res, wt


def truth(star, horizon, worlds=20_000, seed=9_100_000):
    """Wert der Zielregel bei Horizont H, direkt gefahren (Mittel der mittleren Wochenbelohnung)."""
    vals = []
    done = 0
    k = 0
    while done < worlds:
        w = min(5000, worlds - done)
        _, _, r = simulate_trajectories(w, horizon, ("target", star), seed + 1_000_003 * k)
        vals.append(r.mean(axis=1))
        done += w
        k += 1
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def run_config(horizon, rho, star, seed):
    """Alle Datensätze einer Zelle (H, rho). Rückgabe: dict Schätzer -> (DATASETS,), Gewichte (Trajektorien)."""
    p0 = E.protocol(star, rho)
    acc = {}
    weights = []
    for ch in range(DATASETS // CHUNK_DATASETS):
        x, a, r = simulate_trajectories(CHUNK_DATASETS * WORLDS_PER_DATASET, horizon, ("protocol", p0), seed + 1_000_003 * ch)
        ds = np.repeat(np.arange(CHUNK_DATASETS), TRAJ_PER_DATASET)
        res, wt = estimators_multi(x, a, r, ds, CHUNK_DATASETS, p0, star, horizon)
        for k, v in res.items():
            acc.setdefault(k, []).append(v)
        weights.append(wt)
    return {k: np.concatenate(v) for k, v in acc.items()}, np.concatenate(weights)


def generate():
    pool = EV.load_pool()
    star = EV.best_actions(pool)
    cells = []
    for horizon in HORIZONS:
        v, v_se = truth(star, horizon)
        for rho in RHOS:
            out, wts = run_config(horizon, rho, star, 31_000_000 + 10_007 * horizon + 131 * int(round(rho * 10)))
            row = {"H": horizon, "rho": rho, "truth": 100 * v, "truth_se": 100 * v_se,
                   "w_mean": float(wts.mean()), "w_var": float(wts.var()), "w_var_formula": float(rho**-horizon - 1.0),
                   "hit_share": float((wts > 0).mean()), "hit_share_formula": float(rho**horizon),
                   "no_hit": float((out["matched"] == 0).mean()), "ess": float(out["ESS"].mean()), "ests": {}}
            for name in ("IPS", "SNIPS", "PDIS", "DM_falsch", "DM_voll", "DR_voll", "DR_falsch"):
                x = out[name]
                ok = np.isfinite(x)
                err = 100 * (x[ok] - v)
                row["ests"][name] = {
                    "bias": float(err.mean()) if ok.any() else None,
                    "sd": float(100 * x[ok].std(ddof=1)) if ok.sum() > 1 else None,
                    "rmse": float(np.sqrt((err**2).mean())) if ok.any() else None,
                    "defined": float(ok.mean()),
                }
            cells.append(row)
    meta = {"datasets": DATASETS, "trajectories": TRAJ_PER_DATASET, "horizons": list(HORIZONS), "rhos": list(RHOS)}
    OUT.write_text(json.dumps({"meta": meta, "cells": cells}, separators=(",", ":")), encoding="utf-8")
