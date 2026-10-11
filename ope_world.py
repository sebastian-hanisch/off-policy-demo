"""Das Vehikel: acht Standorte mit Order-up-to-Bestand und Wochenlast; Ergebnis je Standort-Woche sind Servicegrad und Reichweite.

Zeitablauf je Tag (alle Standorte und Wiederholungen gleichzeitig, numpy): Wareneingang (Bestellung vom Tag t - LEAD), Nachfrage
(negativ-binomial, Standortgröße x Wochentag x Wochenlast), Bedienung aus dem Bestand, Rückstand bei Fehlmenge, Bestellung bis zum
Niveau S = round(LEAD * mu + z * sd); das Zentrallager liefert unbegrenzt. Belohnung einer Standort-Woche:

    R = Servicegrad - COST * Reichweite,    Reichweite = mittlerer Bestand am Tagesende / mittlere Tagesnachfrage der Woche.

Aller Zufall kommt aus SplitMix64 (`ope_rng.uniform_block`); die negativ-binomiale Inversion läuft über `scipy.stats.nbinom.ppf`."""

import numpy as np
from scipy.stats import nbinom

import ope_constants as C
from ope_rng import uniform_block


def generate_demand(reps, load, weeks, seed):
    """Nachfrage (reps, N_SITES, BURN_DAYS + 7 * weeks). load: broadcastbar auf (reps, N_SITES, weeks) (Faktor je Woche; die
    Einschwingtage nutzen die Last der ersten Woche)."""
    total = C.BURN_DAYS + 7 * weeks
    ld = np.broadcast_to(np.asarray(load, dtype=float), (reps, C.N_SITES, weeks))
    ld = np.concatenate([np.repeat(ld[:, :, :1], C.BURN_DAYS // 7, axis=2), ld], axis=2).repeat(7, axis=2)
    dow = np.tile(C.dow_factors(), total // 7 + 1)[:total]
    mean = C.site_means()[None, :, None] * dow[None, None, :] * ld
    u = uniform_block(seed, reps * C.N_SITES * total).reshape(reps, C.N_SITES, total)
    return nbinom.ppf(u, C.DISP, C.DISP / (C.DISP + mean)).astype(np.int32)


def simulate_weeks(demand, z, load):
    """Servicegrad und Reichweite je Standort und Woche, Form (reps, N_SITES, weeks). z: Sicherheitsfaktor, broadcastbar auf
    (reps, N_SITES, weeks) (je Woche konstant); load wie in `generate_demand`."""
    reps, s, total = demand.shape
    weeks = (total - C.BURN_DAYS) // 7
    mu = C.site_means()
    sd = np.sqrt(C.LEAD * (mu + mu**2 / C.DISP))
    zw = np.broadcast_to(np.asarray(z, dtype=float), (reps, s, weeks))
    zfull = np.concatenate([np.repeat(zw[:, :, :1], C.BURN_DAYS // 7, axis=2), zw], axis=2).repeat(7, axis=2)
    s_order = np.round(C.LEAD * mu[None, :, None] + zfull * sd[None, :, None])
    ni = np.round(C.LEAD * mu)[None, :].repeat(reps, axis=0) + np.round(zfull[:, :, 0] * sd[None, :])
    pipe = [np.zeros((reps, s)) for _ in range(C.LEAD)]
    short = np.empty((reps, s, weeks * 7))
    stock = np.empty((reps, s, weeks * 7))
    for t in range(total):
        ni = ni + pipe[0]
        d = demand[:, :, t].astype(float)
        served = np.minimum(np.maximum(ni, 0.0), d)
        ni = ni - d
        order = np.maximum(0.0, s_order[:, :, t] - (ni + sum(pipe[1:], start=np.zeros_like(ni))))
        pipe = pipe[1:] + [order]
        if t >= C.BURN_DAYS:
            k = t - C.BURN_DAYS
            short[:, :, k] = d - served
            stock[:, :, k] = np.maximum(ni, 0.0)
    dem = demand[:, :, C.BURN_DAYS:].astype(float)
    fill = 1.0 - short.reshape(reps, s, weeks, 7).sum(axis=3) / dem.reshape(reps, s, weeks, 7).sum(axis=3)
    ld = np.broadcast_to(np.asarray(load, dtype=float), (reps, s, weeks))
    cover = stock.reshape(reps, s, weeks, 7).mean(axis=3) / (mu[None, :, None] * ld)
    return fill, cover


def weekly_rows(action, load_idx, worlds, seed):
    """Entscheidungszeilen für eine Zelle (Last, Aktion): acht Standorte in `worlds` Welten, eine Woche nach dem Einschwingen mit
    demselben z und derselben Last. Rückgabe: fill, cover je (worlds, N_SITES)."""
    load = C.LOADS[load_idx]
    demand = generate_demand(worlds, load, 1, seed)
    fill, cover = simulate_weeks(demand, C.ACTIONS[action], load)
    return fill[:, :, 0], cover[:, :, 0]


def rewards(fill, cover):
    return fill - C.COST * cover
