"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Scrollen auf dem Handy nicht zoomt."""

import numpy as np
import plotly.graph_objects as go

import ope_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=360, legend=dict(orientation="h", y=-0.25))


def _finish(fig, title, xtitle=None, ytitle=None, **extra):
    fig.update_layout(title=dict(text=title, x=0), xaxis=dict(title=xtitle, fixedrange=True),
                      yaxis=dict(title=ytitle, fixedrange=True), **{**_LAYOUT, **extra})
    return fig


def context_label(x):
    return f"{C.CLASS_NAMES[x // 2]}, Last {C.LOAD_NAMES[x % 2]}"


def build_counts(n, star):
    """Wie oft hat das Protokoll in jedem Kontext welche Aktion gewählt? Zählung (Zahl); `star` = Aktion der neuen Regel je Kontext."""
    star = [int(a) for a in star]
    labels = [f"z = {a:.1f}".replace(".", ",") for a in C.ACTIONS]
    text = [[f"{int(n[x, a])}" + (" ◆" if a == star[x] else "") for a in range(C.NA)] for x in range(C.NX)]
    fig = go.Figure(go.Heatmap(z=n, x=labels, y=[context_label(x) for x in range(C.NX)], text=text, texttemplate="%{text}",
                               colorscale="Blues", showscale=False, xgap=2, ygap=2,
                               hovertemplate="%{y}, %{x}: %{z} Zeilen<extra></extra>"))
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    fig.update_xaxes(fixedrange=True, side="top")
    fig.update_layout(title=dict(text="Das Protokoll: Zeilen je Kontext und Aktion (◆ = Aktion der neuen Regel)", x=0), height=330,
                      margin=dict(l=10, r=10, t=70, b=10))
    return fig


def build_estimates(est, truth, truth_log):
    """Geschätzter Wert der neuen Regel je Schätzer (Prozent) gegen den wahren Wert und den Wert des Protokolls selbst."""
    names = [k for k in C.ESTIMATORS]
    vals = [100 * est[k] if np.isfinite(est[k]) else None for k in names]
    fig = go.Figure(go.Bar(x=[C.ESTIMATOR_SHORT[k] for k in names], y=vals, marker_color=[C.ESTIMATOR_COLORS[k] for k in names],
                           text=[("%.2f" % v).replace(".", ",") if v is not None else "undefiniert" for v in vals],
                           textposition="outside"))
    fig.add_hline(y=truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="wahrer Wert der neuen Regel",
                  annotation_position="top left")
    fig.add_hline(y=truth_log, line=dict(color=C.COLOR_LOG, dash="dot"), annotation_text="Wert der gefahrenen Regel",
                  annotation_position="bottom left")
    lo = min([v for v in vals if v is not None] + [truth, truth_log]) - 3
    hi = max([v for v in vals if v is not None] + [truth, truth_log]) + 3
    fig.update_yaxes(range=[lo, hi])
    return _finish(fig, "Geschätzter Wert der neuen Regel in diesem Protokoll", None, "Belohnung in Punkten")


def build_spread(rows, truth):
    """Mittel ± Standardabweichung über viele Protokolle je Schätzer, mit der Wahrheit."""
    fig = go.Figure()
    for k in C.ESTIMATORS:
        r = rows[k]
        if not np.isfinite(r["mean"]):
            continue
        fig.add_trace(go.Scatter(x=[C.ESTIMATOR_SHORT[k]], y=[r["mean"]], mode="markers", showlegend=False,
                                 marker=dict(color=C.ESTIMATOR_COLORS[k], size=12),
                                 error_y=dict(type="data", array=[r["sd"]], thickness=3, width=9, color=C.ESTIMATOR_COLORS[k])))
    fig.add_hline(y=truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="wahrer Wert",
                  annotation_position="top left")
    return _finish(fig, "Geschätzter Wert über viele Protokolle: Mittel ± Standardabweichung", None, "Belohnung in Punkten")


def build_sweep(xs, series, title, xtitle, xlog=False, ytitle="RMSE in Punkten"):
    """RMSE (oder anderes Maß) je Schätzer gegen einen Regler. series: {Schätzer: [y]}."""
    fig = go.Figure()
    for k, ys in series.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=C.ESTIMATOR_SHORT.get(k, k),
                                 line=dict(color=C.ESTIMATOR_COLORS.get(k, "#555"), width=2.5)))
    if xlog:
        fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _finish(fig, title, xtitle, ytitle)


def build_models(model_names, dm, dr, snips):
    """RMSE von direkter Methode und doppelt robust je Modellgüte (SNIPS als Bezug)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Direkte Methode", x=model_names, y=dm, marker_color=C.ESTIMATOR_COLORS["dm"]))
    fig.add_trace(go.Bar(name="Doppelt robust", x=model_names, y=dr, marker_color=C.ESTIMATOR_COLORS["dr"]))
    fig.add_trace(go.Scatter(name="SNIPS (kein Modell)", x=model_names, y=snips, mode="lines+markers",
                             line=dict(color=C.ESTIMATOR_COLORS["snips"], width=2, dash="dot")))
    fig.update_layout(barmode="group")
    return _finish(fig, "Wie viel hilft die Korrektur, je nach Güte des Modells? (RMSE)", None, "RMSE in Punkten")


def build_multistep(horizons, series, title):
    fig = go.Figure()
    colors = {"IPS": C.ESTIMATOR_COLORS["ips"], "SNIPS": C.ESTIMATOR_COLORS["snips"], "PDIS": "#8c564b",
              "DM_falsch": C.ESTIMATOR_COLORS["dm"], "DM_voll": "#9ecae1", "DR_voll": C.ESTIMATOR_COLORS["dr"]}
    labels = {"IPS": "IPS", "SNIPS": "SNIPS", "PDIS": "PDIS (je Schritt)", "DM_falsch": "Direkte Methode, falsches Modell",
              "DM_voll": "Direkte Methode, Zellmittel", "DR_voll": "Doppelt robust, Zellmittel"}
    for k, ys in series.items():
        fig.add_trace(go.Scatter(x=horizons, y=ys, mode="lines+markers", name=labels.get(k, k),
                                 line=dict(color=colors.get(k, "#555"), width=2.5)))
    fig.update_yaxes(type="log")
    fig.update_xaxes(tickvals=list(horizons))
    return _finish(fig, title, "Horizont in Wochen", "RMSE in Punkten (logarithmisch)")
