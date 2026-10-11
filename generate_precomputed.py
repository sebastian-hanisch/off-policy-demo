"""Erzeugt `precomputed_pool.json`: je Zelle (Kontext, Aktion) 10 000 Wochenbelohnungen aus dem Vehikel (SplitMix64), dazu die
Mittelwerte von Servicegrad und Reichweite als Wahrheit. Und `precomputed_multistep.json`: die mehrstufige Messreihe (Trajektorien
über H Wochen, Bestand trägt über die Wochengrenze) mit Gewichts-Kennzahlen und Fehlern der Schätzer.

    python generate_precomputed.py            # Pool: etwa eine halbe Minute
    python generate_precomputed.py --multi    # nur die mehrstufige Messreihe (Pool muss vorhanden sein)"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import ope_constants as C
import ope_world as W

POOL_OUT = Path(__file__).resolve().parent / "precomputed_pool.json"
ROWS_PER_CELL = 10_000


def make_pool():
    classes = np.array(C.CLASS_OF_SITE)
    cells = {}
    fill_mean = np.zeros((C.NX, C.NA))
    cover_mean = np.zeros((C.NX, C.NA))
    for li in range(2):
        for a in range(C.NA):
            fill, cover = W.weekly_rows(a, li, C.POOL_WORLDS, 7_000_003 * (li + 1) + 977 * (a + 1))
            r = W.rewards(fill, cover)
            for k in range(3):
                x = k * 2 + li
                sel = classes == k
                rr = r[:, sel].ravel()[:ROWS_PER_CELL]
                assert len(rr) == ROWS_PER_CELL
                cells[f"{x}_{a}"] = [round(float(v), 5) for v in rr]
                fill_mean[x, a] = fill[:, sel].ravel()[:ROWS_PER_CELL].mean()
                cover_mean[x, a] = cover[:, sel].ravel()[:ROWS_PER_CELL].mean()
    meta = {"rows_per_cell": ROWS_PER_CELL, "worlds": C.POOL_WORLDS, "cost": C.COST, "actions": list(C.ACTIONS),
            "loads": list(C.LOADS), "fill_mean": fill_mean.round(6).tolist(), "cover_mean": cover_mean.round(6).tolist()}
    POOL_OUT.write_text(json.dumps({"meta": meta, "cells": cells}, separators=(",", ":")), encoding="utf-8")


def main():
    t0 = time.time()
    if "--multi" not in sys.argv:
        make_pool()
        print(f"{POOL_OUT.name}: {C.NC} Zellen à {ROWS_PER_CELL}, {time.time() - t0:.0f} s", file=sys.stderr)
    if "--multi" in sys.argv or "--all" in sys.argv:
        import ope_multistep as M
        M.generate()
        print(f"mehrstufig: {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
