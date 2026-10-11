"""Auswertung aus Logdaten: Off-Policy-Evaluation - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwölftes und letztes Stück der Konzepte-Linie "Experimente und Wirkungsmessung": Eine neue Bestandsregel wurde nie gefahren; es
gibt nur Logdaten einer anderen Regel. Wie gut lässt sich der Wert der neuen Regel daraus schätzen (Gewichtung, Modell, beides),
und wann reicht die Überlappung nicht? Mit bekannter Wahrheit und mehrstufig gemessen. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

import ope_constants as C
from ope_evaluation import best_actions, load_pool, run_one, summarize
from ope_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)
from ope_visualization import build_counts, build_estimates, build_models, build_multistep, build_spread, build_sweep

st.set_page_config(page_title="Off-Policy-Evaluation – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _pool():
    return load_pool()


@st.cache_data(show_spinner=False)
def _multistep():
    data = json.loads((Path(__file__).resolve().parent / "precomputed_multistep.json").read_text(encoding="utf-8"))
    return data["cells"]


@st.cache_data(show_spinner=False)
def _one(n, rho_index, model, seed):
    return run_one(_pool(), n, rho_index, model, seed)


@st.cache_data(show_spinner=False)
def _many(n, rho_index, model):
    return summarize(_pool(), n, rho_index, model, reps=C.WORLDS_LIVE)


@st.cache_data(show_spinner=False)
def _sweep_n(rho_index, model):
    return {n: summarize(_pool(), n, rho_index, model, reps=400, tag=5) for n in (250, 500, 1000, 2000, 5000)}


@st.cache_data(show_spinner=False)
def _sweep_rho(n, model):
    return {i: summarize(_pool(), n, i, model, reps=400, tag=7) for i in range(len(C.RHO_CHOICES))}


@st.cache_data(show_spinner=False)
def _sweep_model(n, rho_index):
    return {m: summarize(_pool(), n, rho_index, m, reps=400, tag=9) for m in C.MODELS}


st.title("📒 Auswertung aus Logdaten: Off-Policy-Evaluation")
st.markdown(
    """
Eine neue Bestandsregel soll bewertet werden, aber **sie wurde nie gefahren**. Was es gibt, sind Logdaten der **bisherigen**
Regel: Standort-Wochen mit Kontext (Größe, Last), gewählter Sicherheitsstufe und Belohnung. Daraus den Wert der neuen Regel zu
schätzen, ohne sie zu testen, heißt **Off-Policy-Evaluation**. Es gibt zwei Wege: das Ergebnis der Zeilen **gewichten**, deren
Aktion zur neuen Regel passt, oder ein **Modell** der Belohnung fragen. Der dritte Weg, **doppelt robust**, nimmt beides. Hier
laufen alle mit bekannter Wahrheit gegeneinander.
"""
)
st.caption(
    "Zwölftes Stück der Linie „Experimente und Wirkungsmessung“, aufbauend auf "
    "[ab-test-demo](https://sebastianhanisch-ab-test-demo.streamlit.app/) (gleiches Vehikel: Bestandsregel mit Servicegrad). "
    "Nicht zu verwechseln mit „off-policy“ im Q-Learning (anderes Verfahren, anderer Zweck)."
)

with st.expander("So ist das Entscheidungsproblem aufgebaut", expanded=True):
    st.markdown(
        """
- **Entscheidung:** ein Standort in einer Woche. **Kontext:** Größenklasse (klein, mittel, groß) und Wochenlast (niedrig 0,8, hoch
  1,2 der üblichen Nachfrage). **Aktion:** Sicherheitsfaktor z = 0,5, 1,0 oder 1,5.
- **Belohnung:** Servicegrad der Woche minus 0,05 mal Reichweite in Tagen (Bestand im Mittel / Tagesnachfrage). Mehr Sicherheit
  hilft beim Servicegrad, kostet aber Bestand.
- **Neue Regel π\\*:** in jedem Kontext die beste Aktion (niedrige Last z = 1,0, hohe Last z = 1,5). Sie ist **nie gefahren**
  worden.
- **Protokoll:** die bisherige Regel wählt die Aktion von π\\* mit Wahrscheinlichkeit ρ (der Überlappung); den Rest verteilt sie so,
  dass sie bei hoher Last eher viel und bei niedriger Last eher wenig Sicherheit fährt.
- **Wahrheit:** Mittel von je 10 000 simulierten Standort-Wochen je Kontext und Aktion (Servicegrad und Reichweite aus dem Vehikel).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{i}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])
st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Logdaten**")
    n = st.slider("Zeilen im Log (Standort-Wochen)", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="So viele Entscheidungen hat das Protokoll aufgezeichnet.")
    rho_index = st.radio("Überlappung ρ: Anteil der Fälle, in denen das Protokoll die Aktion der neuen Regel wählt",
                         options=list(range(len(C.RHO_CHOICES))), key="rho_radio",
                         format_func=lambda i: f"ρ = {C.fmt_num(C.RHO_CHOICES[i], 2)}",
                         help="Je kleiner, desto weniger Zeilen gehören zur neuen Regel; die Gewichte werden größer (1/ρ).")
    st.markdown("**Modell der Belohnung**")
    model_index = st.radio("Wie gut ist das Modell?", options=list(range(len(C.MODELS))), key="model_radio",
                           format_func=lambda i: C.MODEL_LABELS[C.MODELS[i]],
                           help="Das Modell ist eine Regression der Belohnung auf Größe, Last und Aktion. Das falsche kennt keine "
                                "Wechselwirkung von Last und Aktion, die in den Daten steckt.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt das eine Protokoll unten.")
    st.button("🎲 Neues Protokoll würfeln", on_click=randomize_seed)

n, rho_index, model_index, seed = int(n), int(rho_index), int(model_index), int(seed)
model = C.MODELS[model_index]
rho = C.RHO_CHOICES[rho_index]
sync_query_params({"n_slider": n, "rho_radio": rho_index, "model_radio": model_index, "seed_input": seed})

pool = _pool()
many = _many(n, rho_index, model)
truth, truth_log = many["value"], many["value_log"]

# --- Ein Protokoll --------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧪 Ein Protokoll")
one = _one(n, rho_index, model, seed)
st.caption(
    f"{C.fmt_int(n)} Zeilen, ρ = {C.fmt_num(rho, 2)}; Belohnungsmodell: {C.MODEL_LABELS[model]}. Wahrer Wert der neuen Regel: "
    f"**{C.fmt_num(truth, 2)}**, der gefahrenen Regel: {C.fmt_num(truth_log, 2)} (Punkte Belohnung)."
)
st.plotly_chart(build_counts(one["n"], best_actions(pool)), width="stretch", key=f"counts_{n}_{rho_index}_{seed}")
st.plotly_chart(build_estimates(one["est"], truth, truth_log), width="stretch", key=f"est_{n}_{rho_index}_{model}_{seed}")
st.caption(
    f"Zur neuen Regel passen {C.fmt_int(one['hits'])} der {C.fmt_int(n)} Zeilen; die effektive Stichprobengröße der Gewichte "
    f"(ESS) ist {C.fmt_num(one['ess'], 0)}. Die neue Regel ist im Wert nur {C.fmt_num(many['gap'], 2)} Punkte besser als die beste "
    f"konstante Regel (z = {C.fmt_num(C.ACTIONS[many['const_action']], 1)} überall): ein kleiner Abstand, der verlangt, dass "
    "die Schätzung genauer ist als er."
)

# --- Viele Protokolle -----------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 Tausend Protokolle: wer trifft den Wert?")
rows = many["rows"]
st.plotly_chart(build_spread(rows, truth), width="stretch", key=f"spread_{n}_{rho_index}_{model}")
tab = ["| Schätzer | Verzerrung | Streuung | RMSE | Sagt richtig „neue Regel besser“ |", "|---|---|---|---|---|"]
for k in C.ESTIMATORS:
    r = rows[k]
    rmse = C.fmt_num(r["rmse"], 2) if np.isfinite(r["rmse"]) else "–"
    extra = "" if r["defined"] >= 0.999 else f" (nur in {C.fmt_pct(r['defined'], 0)} der Protokolle definiert)"
    tab.append(f"| {C.ESTIMATOR_SHORT[k]} | {C.fmt_num(r['bias'], 2) if np.isfinite(r['bias']) else '–'} | "
               f"{C.fmt_num(r['sd'], 2) if np.isfinite(r['sd']) else '–'} | {rmse}{extra} | "
               f"{C.fmt_pct(r['correct_order'], 0) if np.isfinite(r['correct_order']) else '–'} |")
st.markdown("\n".join(tab))
st.caption(
    f"Punkte Belohnung. Wahrheit {C.fmt_num(truth, 2)}; in {C.fmt_pct(many['no_hit'], 1)} der Protokolle gibt es keine einzige Zeile, "
    f"die zur neuen Regel passt; im Mittel {C.fmt_num(many['mean_hits'], 0)} Zeilen. „Sagt richtig“: der Anteil der Protokolle, in "
    "denen der Schätzer die neue Regel höher bewertet als die beste konstante Regel (in Wahrheit ist sie es)."
)
ips, snips, dm, dr = rows["ips"], rows["snips"], rows["dm"], rows["dr"]
st.error(
    f"**IPS ist im Mittel richtig, aber streut {C.fmt_num(ips['sd'] / max(dr['sd'], 1e-9), 1)}-mal so stark wie Doppelt "
    f"robust** (RMSE {C.fmt_num(ips['rmse'], 2)} gegen {C.fmt_num(dr['rmse'], 2)}). Die Direkte Methode streut "
    f"wenig, liegt aber {C.fmt_num(dm['bias'], 2)} Punkte daneben (RMSE {C.fmt_num(dm['rmse'], 2)}): bei einem Abstand der Regeln von "
    f"{C.fmt_num(many['gap'], 2)} Punkten reicht das, um die Reihenfolge zu kippen."
    if model == "falsch" else
    f"**IPS ist im Mittel richtig, streut aber {C.fmt_num(ips['sd'] / max(dr['sd'], 1e-9), 1)}-mal so stark wie die "
    f"modellbasierten Schätzer** (RMSE {C.fmt_num(ips['rmse'], 2)} gegen {C.fmt_num(dr['rmse'], 2)} bei Doppelt robust). Mit "
    f"diesem Modell ist die Direkte Methode ({C.fmt_num(dm['rmse'], 2)}) praktisch so gut wie Doppelt robust."
)

# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Wie viele Daten, wie viel Überlappung?")
with st.spinner("Rechne die Sweeps …"):
    sn = _sweep_n(rho_index, model)
    sr = _sweep_rho(n, model)
    sm = _sweep_model(n, rho_index)
ns = list(sn)
st.plotly_chart(build_sweep(ns, {k: [sn[x]["rows"][k]["rmse"] for x in ns] for k in C.ESTIMATORS},
                            f"RMSE gegen die Zahl der Zeilen (ρ = {C.fmt_num(rho, 2)}, Modell: {model})", "Zeilen im Log", xlog=True),
                width="stretch", key=f"sweep_n_{rho_index}_{model}")
rhos = list(C.RHO_CHOICES)
st.plotly_chart(build_sweep(rhos, {k: [sr[i]["rows"][k]["rmse"] for i in range(len(rhos))] for k in C.ESTIMATORS},
                            f"RMSE gegen die Überlappung ρ ({C.fmt_int(n)} Zeilen, Modell: {model})", "Überlappung ρ", xlog=True),
                width="stretch", key=f"sweep_rho_{n}_{model}")
st.plotly_chart(
    build_models([C.MODEL_LABELS[m].split(":")[0] for m in C.MODELS], [sm[m]["rows"]["dm"]["rmse"] for m in C.MODELS],
                 [sm[m]["rows"]["dr"]["rmse"] for m in C.MODELS], [sm[m]["rows"]["snips"]["rmse"] for m in C.MODELS]),
    width="stretch", key=f"models_{n}_{rho_index}")
st.info(
    f"Die Korrektur verwendet die Gewichte nur für den **Fehler des Modells**. Beim falschen Modell senkt sie den Fehler von "
    f"{C.fmt_num(sm['falsch']['rows']['dm']['rmse'], 2)} auf {C.fmt_num(sm['falsch']['rows']['dr']['rmse'], 2)}; beim Zellmittel-"
    f"Modell, das die Daten bereits voll ausschöpft, bleibt sie ohne Wirkung ({C.fmt_num(sm['voll']['rows']['dm']['rmse'], 2)} "
    f"gegen {C.fmt_num(sm['voll']['rows']['dr']['rmse'], 2)}): **doppelter Schutz nur dort, wo das Modell die Daten nicht ausschöpft**."
)

# --- Mehrstufig -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("⛓️ Mehrstufig: Gewichte multiplizieren sich")
cells = _multistep()
hs = sorted({c["H"] for c in cells})
by = {(c["H"], c["rho"]): c for c in cells}
series = {k: [by[(h, 0.5)]["ests"][k]["rmse"] for h in hs] for k in ("IPS", "SNIPS", "PDIS", "DM_falsch", "DM_voll", "DR_voll")}
st.plotly_chart(build_multistep(hs, series, "RMSE gegen den Horizont, ρ = 0,5 je Woche (2 000 Trajektorien)"), width="stretch",
                key="multistep_chart")
mt = ["| ρ je Woche | Horizont | Varianz der Gewichte | Anteil der Datensätze ohne Treffer | ESS (von 2 000) |", "|---|---|---|---|---|"]
for c in cells:
    mt.append(f"| {C.fmt_num(c['rho'], 1)} | {c['H']} Wochen | {C.fmt_num(c['w_var'], 1)} (Formel ρ^−H − 1 = "
              f"{C.fmt_num(c['w_var_formula'], 1)}) | {C.fmt_pct(c['no_hit'], 0)} | {C.fmt_num(c['ess'], 1)} |")
st.markdown("\n".join(mt))
c10, c5 = by[(10, 0.5)], by[(5, 0.2)]
st.info(
    f"Das Gewicht einer ganzen Trajektorie ist das Produkt der Wochen-Gewichte: seine Varianz wächst wie ρ^−H − 1. Bei ρ = 0,5 und "
    f"10 Wochen haben {C.fmt_pct(c10['no_hit'], 0)} der Datensätze keine einzige passende Trajektorie, bei ρ = 0,2 und 5 Wochen "
    f"{C.fmt_pct(c5['no_hit'], 0)}. Die direkte Methode bleibt nutzbar, trägt aber einen **strukturellen** Fehler, weil sie "
    "die Nachwirkung des Bestands von Woche zu Woche nicht kennt."
)

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Überlappung: jede Aktion der neuen Regel kommt im Log vor** | Wählt das Protokoll eine Aktion nie, gibt es keine Zeile, die darüber etwas sagt; IPS und die Korrektur sind blind, und das Modell extrapoliert. | (dieses Stück) |
| **Das Protokoll ist bekannt (protokollierte Wahrscheinlichkeiten)** | Ohne die Wahrscheinlichkeiten fehlen die Gewichte; sie müssen geschätzt werden, und der Fehler geht in jede Schätzung ein. | (dieses Stück) |
| **Alle Einflüsse auf Aktion und Belohnung stehen im Kontext** | Verdeckte Einflüsse (etwa ein Disponent, der Sonderfälle anders fährt) machen jede Gewichtung schief; das ist mit Daten allein nicht prüfbar. | (dieses Stück) |
| **Entscheidungen sind unabhängig** | Bestand und Pipeline tragen über Wochen; mehrstufige Gewichte werden unbrauchbar, und die direkte Methode trägt einen strukturellen Fehler. | (dieses Stück) |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen oder Zielregeln vervielfachen die Vergleiche. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |
"""
)
st.caption(
    "Die Zahlen stammen aus dem erfundenen Vehikel dieses Stücks (acht Standorte, sechs Kontexte, drei Aktionen). Das Protokoll ist "
    "eine Modellannahme."
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Wert einer Regel.** Mit Kontext $x$, Aktion $a$ und Belohnung $r$ ist $V(\pi) = E[\sum_a \pi(a|x)\, r(x,a)]$. Gesucht ist
$V(\pi^*)$ für eine Regel $\pi^*$, die im Log nicht vorkommt; das Log stammt aus dem Protokoll $p_0$.

**IPS.** $\hat V_{\mathrm{IPS}} = \frac1N \sum_i w_i r_i$ mit $w_i = \pi^*(a_i|x_i)/p_0(a_i|x_i)$. Unverzerrt, wenn $p_0(a|x) > 0$,
wo $\pi^*(a|x) > 0$. Die Varianz der Gewichte ist bei deterministischem $\pi^*$ und Treffer-Wahrscheinlichkeit $\rho$ gleich
$1/\rho - 1$.

**SNIPS.** $\hat V_{\mathrm{SNIPS}} = \sum_i w_i r_i / \sum_i w_i$: schwach verzerrt, aber deutlich ruhiger; ohne Treffer
undefiniert.

**Direkte Methode.** $\hat V_{\mathrm{DM}} = \frac1N \sum_i \sum_a \pi^*(a|x_i)\, \hat q(x_i,a)$ mit einem Modell $\hat q$ der Belohnung.
Streut wenig, trägt aber den ganzen Modellfehler.

**Doppelt robust.** $\hat V_{\mathrm{DR}} = \hat V_{\mathrm{DM}} + \frac1N \sum_i w_i\,(r_i - \hat q(x_i,a_i))$. Unverzerrt, wenn das Modell
oder die Gewichte stimmen. Ist $\hat q$ saturiert (Zellmittel), verschwindet die Korrektur und DR = DM.

**Mehrstufig.** Für eine Trajektorie über $H$ Schritte ist $w = \prod_{h} w_h$; bei Treffer-Wahrscheinlichkeit $\rho$ je Schritt ist
die Varianz $\rho^{-H} - 1$. PDIS gewichtet jede Belohnung nur mit den Gewichten bis zu ihrem Schritt.

Implementiert in `ope_estimators.py` (Schätzer), `ope_evaluation.py` (Protokolle, Wahrheit, Zusammenfassungen), `ope_world.py`
(Vehikel) und `ope_multistep.py` (Trajektorien).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Experimente und Wirkungsmessung: A/B bis Logdaten](https://sebastianhanisch.net/konzepte-experimente-wirkungsmessung.html)."
)
