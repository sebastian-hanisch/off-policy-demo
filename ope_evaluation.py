"""Auswertung: Off-Policy-Evaluation mit bekannter Wahrheit.

Der Pool (je Zelle 10 000 Wochenbelohnungen aus dem Vehikel) ist die Wahrheit: r(x, a) ist das Mittel der Zelle, der Wert einer
Regel pi ist V(pi) = sum_x P(x) sum_a pi(a|x) r(x, a). Ein Protokoll (Log) zieht N Entscheidungen: Größenklassen fest, Last 50/50,
Aktion aus dem Protokoll p0; die Belohnung wird aus dem Pool der Zelle gezogen (mit Zurücklegen). Gefragt ist der Wert der
Zielregel, die niemand gefahren ist."""

import json
from pathlib import Path

import numpy as np

import ope_constants as C
import ope_estimators as E
from ope_rng import uniform_block

POOL_PATH = Path(__file__).resolve().parent / "precomputed_pool.json"


def load_pool(path=POOL_PATH):
    """Rückgabe: dict(rewards (18, 10000), r (6, 3) Mittel, fill (6, 3), cover (6, 3))."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rewards = np.empty((C.NC, data["meta"]["rows_per_cell"]))
    for x in range(C.NX):
        for a in range(C.NA):
            rewards[x * C.NA + a] = data["cells"][f"{x}_{a}"]
    return {"rewards": rewards, "r": rewards.mean(axis=1).reshape(C.NX, C.NA),
            "fill": np.array(data["meta"]["fill_mean"]), "cover": np.array(data["meta"]["cover_mean"])}


def value(pool, pi):
    """Wahrer Wert einer Regel pi (6, 3)."""
    return float((C.ctx_p()[:, None] * pi * pool["r"]).sum())


def best_actions(pool):
    """Beste Aktion je Kontext (die Zielregel pi*)."""
    return pool["r"].argmax(axis=1)


def target_policy(pool):
    return E.det_policy(best_actions(pool))


def constant_policy(a):
    return E.det_policy(np.full(C.NX, a))


def best_constant(pool):
    """Index der besten konstanten Regel und ihr Wert."""
    vals = [value(pool, constant_policy(a)) for a in range(C.NA)]
    return int(np.argmax(vals)), max(vals)


def cell_probabilities(p0):
    """Wahrscheinlichkeit je Zelle (x, a) bei Last 50/50 und Aktion aus p0, bezogen auf die Zeilen der Größenklasse (18,)."""
    return p0.ravel()


def sample_logs(pool, p0, n_rows, reps, seed):
    """`reps` Protokolle mit je `n_rows` Entscheidungen. Rückgabe: Zähler n (reps, 18) und Belohnungssummen s (reps, 18).
    Die Zeilenzahl je Größenklasse ist fest (n_rows * CLASS_P); Last und Aktion sind zufällig. Je Protokoll-Block von 100 ein
    eigener Zufallsstrom: die ersten k Protokolle hängen nicht von der Gesamtzahl ab."""
    n = np.zeros((reps, C.NC))
    s = np.zeros((reps, C.NC))
    rewards = pool["rewards"]
    per_cell = rewards.shape[1]
    class_rows = [int(round(n_rows * p)) for p in C.CLASS_P]
    for c_idx, start in enumerate(range(0, reps, 100)):
        r = min(100, reps - start)
        base = seed + 1_000_003 * c_idx
        for k in range(3):
            nk = class_rows[k]
            probs = np.concatenate([C.LOAD_P[ld] * p0[k * 2 + ld] for ld in range(2)])        # 6 Zellen (Last, Aktion)
            cum = np.cumsum(probs)
            cum[-1] = 1.0
            u = uniform_block(base + 31 * (k + 1), r * nk).reshape(r, nk)
            cat = np.searchsorted(cum, u, side="right").clip(0, 5)
            cell = (k * 2 + cat // 3) * C.NA + cat % 3                                          # (r, nk)
            u2 = uniform_block(base + 31 * (k + 1) + 11, r * nk).reshape(r, nk)
            val = rewards[cell, np.minimum((u2 * per_cell).astype(np.int64), per_cell - 1)]
            rid = np.repeat(np.arange(r), nk)
            key = rid * C.NC + cell.ravel()
            n[start:start + r] += np.bincount(key, minlength=r * C.NC).reshape(r, C.NC)
            s[start:start + r] += np.bincount(key, weights=val.ravel(), minlength=r * C.NC).reshape(r, C.NC)
    return n, s


def run_seed(n_rows, rho_index, tag=0):
    return 21_000_000 + 10_007 * n_rows + 131 * rho_index + 13 * tag


def summarize(pool, n_rows, rho_index, model, reps=C.WORLDS_LIVE, tag=0, seed=None):
    """Viele Protokolle bei (N, rho, Modell): je Schätzer Verzerrung, Streuung, RMSE gegen den wahren Wert der Zielregel;
    dazu Gewichts-Kennzahlen und der Entscheidungstest (Zielregel gegen die beste konstante Regel)."""
    rho = C.RHO_CHOICES[rho_index]
    star = best_actions(pool)
    pi = E.det_policy(star)
    p0 = E.protocol(star, rho)
    seed = run_seed(n_rows, rho_index, tag) if seed is None else seed
    n, s = sample_logs(pool, p0, n_rows, reps, seed)
    q = E.fit_model(n, s, model)
    est = E.estimators(n, s, q, pi, p0)
    v = value(pool, pi)
    v_log = value(pool, p0)
    a_c, v_c = best_constant(pool)
    pi_c = constant_policy(a_c)
    est_c = E.estimators(n, s, q, pi_c, p0)
    rows = {}
    for name in C.ESTIMATORS:
        x = est[name]
        ok = np.isfinite(x)
        err = 100 * (x[ok] - v)
        rows[name] = {
            "bias": float(err.mean()) if ok.any() else float("nan"),
            "sd": float(100 * x[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
            "rmse": float(np.sqrt((err**2).mean())) if ok.any() else float("nan"),
            "mean": float(100 * x[ok].mean()) if ok.any() else float("nan"),
            "defined": float(ok.mean()),
        }
        # Entscheidungstest: sagt der Schätzer richtig, dass die Zielregel besser ist als die beste konstante Regel?
        both = ok & np.isfinite(est_c[name])
        rows[name]["correct_order"] = float((est[name][both] > est_c[name][both]).mean()) if both.any() else float("nan")
    hits = est["_hits"]
    w = E.weights(pi, p0)
    return {"n": n_rows, "rho": rho, "model": model, "reps": reps, "value": 100 * v, "value_log": 100 * v_log,
            "value_const": 100 * v_c, "const_action": a_c, "gap": 100 * (v - v_c), "rows": rows,
            "no_hit": float((hits == 0).mean()), "mean_hits": float(hits.mean()), "ess": float(est["_ess"].mean()),
            "weight_max": float(w.max()), "weight_var": float(1.0 / rho - 1.0)}


def run_one(pool, n_rows, rho_index, model, seed):
    """Das eine Protokoll der Seitenleiste: Zähler je Zelle, Schätzungen und Modell."""
    rho = C.RHO_CHOICES[rho_index]
    star = best_actions(pool)
    pi = E.det_policy(star)
    p0 = E.protocol(star, rho)
    n, s = sample_logs(pool, p0, n_rows, 1, 77_000_000 + 7_919 * int(seed))
    q = E.fit_model(n, s, model)
    est = E.estimators(n, s, q, pi, p0)
    return {"n": n[0].reshape(C.NX, C.NA), "s": s[0].reshape(C.NX, C.NA), "q": q[0].reshape(C.NX, C.NA),
            "est": {k: float(est[k][0]) for k in C.ESTIMATORS}, "p0": p0, "pi": pi, "hits": float(est["_hits"][0]),
            "ess": float(est["_ess"][0])}
