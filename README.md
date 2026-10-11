# Auswertung aus Logdaten: Off-Policy-Evaluation – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-off-policy-demo.streamlit.app/)**

Interaktive Demo zur **Bewertung einer nie gefahrenen Regel aus Logdaten**. **Zwölftes und letztes Stück der Konzepte-Linie
„Experimente und Wirkungsmessung“** im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und
Machine Learning). Das Vehikel ist die Bestandsregel der Vorgänger (tägliche Nachbestellung, Servicegrad), aber nicht als Test,
sondern als Entscheidungsproblem: jede Standort-Woche hat Kontext und Aktion, und die neue Regel soll aus den Aufzeichnungen der
alten bewertet werden. Nicht zu verwechseln mit „off-policy“ im Q-Learning (anderes Verfahren, anderer Zweck).

## Kernfrage

Eine neue Regel wurde nie gefahren; was gibt es sonst an Daten? Wie gut lässt sich ihr Wert aus Logdaten der alten Regel schätzen
(Gewichten, Modellieren, beides), und wann reicht die Überlappung nicht?

## Modell und Methodik

- **Entscheidung:** ein Standort in einer Woche (`ope_world.py`). **Kontext:** Größenklasse (klein / mittel / groß, 2 / 4 / 2 von
  8 Standorten) und Wochenlast (niedrig 0,8 / hoch 1,2 der Nachfrage). **Aktion:** Sicherheitsfaktor z = 0,5 / 1,0 / 1,5.
- **Belohnung:** Servicegrad der Woche minus 0,05 · Reichweite (mittlerer Bestand / mittlere Tagesnachfrage, in Tagen).
- **Wahrheit:** je Zelle (Kontext, Aktion) 10 000 simulierte Standort-Wochen (SplitMix64, `precomputed_pool.json`); der Wert einer Regel
  ist das mit den Kontext-Anteilen gewichtete Zellmittel. Die **neue Regel π\*** wählt in jedem Kontext die beste Aktion (niedrige
  Last z = 1,0, hohe Last z = 1,5).
- **Protokoll** (`ope_estimators.protocol`): wählt die Aktion von π\* mit Wahrscheinlichkeit ρ; den Rest verteilt es so, dass bei
  hoher Last eher viel, bei niedriger Last eher wenig Sicherheit gefahren wird.
- **Vier Schätzer** (`ope_estimators.py`): IPS (Gewichtung mit π\*/p₀), SNIPS (normiert), Direkte Methode (Belohnungsmodell),
  Doppelt robust (Modell plus gewichtete Korrektur). **Vier Modelle:** falsch (Last und Aktion additiv), halb gut (Last × z linear),
  gut (Last × Aktion vollständig), Zellmittel (saturiert).
- **Mehrstufig** (`ope_multistep.py`): eine Trajektorie ist ein Standort über H Wochen; Bestand und Pipeline tragen über die
  Wochengrenze (ein Vehikel-Lauf über 7 H Tage). Dazu PDIS (je Schritt gewichtet).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (einstufig je Zelle 3 000 Protokolle mit 2 000 Zeilen, wenn nicht anders genannt;
mehrstufig 200 Datensätze mit je 2 000 Trajektorien). Belohnung in Punkten (Prozentpunkte der Belohnung).

| Frage | Befund |
|---|---|
| Wie viel besser ist die neue Regel? | **89,54** Punkte gegen **89,01** für die beste konstante Regel (z = 1,5 überall): Abstand **0,54**. Die gefahrene Regel liegt bei **84,64 / 85,58 / 87,07** (ρ = 0,01 / 0,2 / 0,5). |
| Ist IPS richtig? | **Im Mittel ja, im Einzelfall nein:** RMSE **20,0 / 8,5 / 6,1 / 4,0 / 2,0** Punkte bei ρ = 0,01 / 0,05 / 0,1 / 0,2 / 0,5, also das Mehrfache des Abstands von 0,54. |
| Und SNIPS? | RMSE **1,81 / 0,79 / 0,57 / 0,39 / 0,25**: 8- bis 11-mal ruhiger als IPS, mit kleiner Verzerrung. |
| Direkte Methode mit falschem Modell? | **Verzerrt um mehr als der Abstand:** −**3,21 / −2,63 / −2,15 / −1,54 / −0,66** Punkte (RMSE 3,24 … 0,70). Das Modell kennt die Wechselwirkung von Last und Aktion nicht. Mit dem halb guten Modell +1,10 / +0,73 / +0,47 / +0,21: das Vorzeichen hängt vom Modell ab. |
| Doppelt robust mit falschem Modell? | RMSE **2,95 / 1,10 / 0,68 / 0,41 / 0,25**: ab ρ = 0,05 etwa so gut wie SNIPS und 2,4- bis 3,8-mal besser als die Direkte Methode (0,25 gegen 0,70 bei ρ = 0,5). Bei ρ = 0,01 mit **2,95** schlechter als SNIPS (1,81). |
| Mit gutem Modell? | Direkte Methode und Doppelt robust sind gleich gut (RMSE 0,38 bei ρ = 0,2); mit dem Zellmittel-Modell **verschwindet die Korrektur** (DR = DM, 0,38). |
| Entscheidungstest: sagt der Schätzer „neue Regel besser als beste konstante Regel“? | Bei ρ = 0,2: IPS **56 %**, SNIPS **100 %**, Doppelt robust **99 %**, Direkte Methode mit falschem Modell **0 %** (in keinem Protokoll richtig), mit halb gutem Modell **79 %**, mit gutem **100 %**. Bei ρ = 0,5 und halb gutem Modell nur **55 %**. Bei ρ = 0,01 sagt die Direkte Methode mit falschem Modell dagegen in **100 %** das Richtige: die Richtung ihres Fehlers hängt vom Protokoll ab, ein richtiges Ergebnis ist dort ein Zufall des Protokolls. |
| Hilft mehr Daten? | Bei ρ = 0,2, falsches Modell, 250 / 500 / 1 000 / 2 000 / 5 000 Zeilen: IPS **11,3 / 8,3 / 5,7 / 4,0 / 2,6**, SNIPS 1,16 / 0,81 / 0,56 / 0,39 / 0,25, Doppelt robust 1,22 / 0,84 / 0,59 / 0,41 / 0,26, Direkte Methode **1,77 / 1,67 / 1,61 / 1,57 / 1,56** (bleibt wegen der Verzerrung stehen). |
| Knappe Überlappung bei wenigen Daten? | ρ = 0,01 und 250 Zeilen: in **8,4 %** der Protokolle keine einzige passende Zeile (SNIPS dort undefiniert); RMSE Doppelt robust **8,2** gegen Direkte Methode **3,4**. Mit 5 000 Zeilen dreht es sich: **1,9** gegen **3,2**. |
| Mehrstufig: wie wachsen die Gewichte? | Varianz des Trajektorien-Gewichts **7,0** bei 3 und **31,0** bei 5 Wochen (ρ = 0,5 je Woche), genau ρ^−H − 1. |
| Wann gibt es gar keinen Treffer? | ρ = 0,5, 10 Wochen: **13 %** der Datensätze ohne passende Trajektorie (ESS **1,9** von 2 000); ρ = 0,2, 5 Wochen: **56 %**, 10 Wochen: **100 %**. |
| Wie schneiden die Schätzer mehrstufig ab (ρ = 0,5, 5 Wochen)? | RMSE IPS **11,1**, PDIS **4,7**, SNIPS **0,47**, Doppelt robust **0,28**; die Direkte Methode ist um **−1,03** (falsch) bzw. **−0,56** (Zellmittel) verzerrt, weil sie die Nachwirkung des Bestands nicht kennt. Bei 10 Wochen: IPS **62,5**, Doppelt robust **0,86**. |
| Default-Seed 35 (ρ = 0,2, falsches Modell) | Fehler gegen die Wahrheit: IPS **+3,77**, SNIPS **−0,03**, Direkte Methode **−1,59**, Doppelt robust **+0,07**; 417 passende Zeilen. |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zielregel ist kaum besser als eine einfache Vergleichsregel.** Der Plan sprach von „Größe × Last“ als Wechselwirkung; die
  Messung zeigt: die Größe spielt kaum eine Rolle, **Last × Aktion** trägt die Wechselwirkung. Der Abstand von 0,54 Punkten zur
  besten konstanten Regel ist der Maßstab für jede Schätzung.
- **Doppelt robust schützt vor einem schlechten Modell und zahlt mit Varianz.** Bei dünner Überlappung (ρ = 0,01) ist es **schlechter**
  als SNIPS, ab ρ = 0,05 gleichauf. Der Plan-Satz „doppelt robust ist robust“ steht deshalb nicht so im Text.
- **Doppelter Schutz nur dort, wo das Modell die Daten nicht ausschöpft.** Mit saturiertem Modell (Zellmittel) ist die Korrektur
  null und DR = DM.
- **Gekapptes IPS ist keine Abhilfe:** in der Vorab-Messreihe war es um 45 bis 81 Punkte verzerrt. Die Demo zeigt es nicht.
- **Schwache Fehlspezifikation:** bei dem halb guten Modell ist die Direkte Methode (RMSE 0,40 bei ρ = 0,2) praktisch so gut wie
  Doppelt robust (0,38): dort bringt die Korrektur nichts.
- **Mehrstufig ist der eigentliche Härtetest:** Gewichte multiplizieren sich, und bei 5 bis 10 Wochen hat ein beträchtlicher Teil der
  Datensätze keine passende Trajektorie. SNIPS bleibt brauchbar, wo es definiert ist; IPS ist unbrauchbar.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Überlappung: jede Aktion der neuen Regel kommt im Log vor** | Wählt das Protokoll eine Aktion nie, gibt es keine Zeile, die darüber etwas sagt. | (dieses Stück) |
| **Das Protokoll ist bekannt (protokollierte Wahrscheinlichkeiten)** | Ohne die Wahrscheinlichkeiten müssen die Gewichte geschätzt werden. | (dieses Stück) |
| **Alle Einflüsse auf Aktion und Belohnung stehen im Kontext** | Verdeckte Einflüsse verfälschen jede Gewichtung; mit Daten allein nicht prüfbar. | (dieses Stück) |
| **Entscheidungen sind unabhängig** | Bestand trägt über Wochen; mehrstufige Gewichte werden unbrauchbar. | (dieses Stück) |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen oder Zielregeln vervielfachen die Vergleiche. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |

Die Zahlen gelten für dieses erfundene Entscheidungsproblem (acht Standorte, sechs Kontexte, drei Aktionen); das Protokoll ist eine
Modellannahme.

## Bezug zu OR

Wer eine neue Disposition einführen will, hat Logdaten der alten, aber meist keine Möglichkeit, die neue ohne Risiko zu testen.
Off-Policy-Evaluation sagt, wie weit diese Daten tragen: wenn die alte Regel die Aktionen der neuen oft genug gewählt hat und der
Abstand größer ist als die Streuung der Schätzung. Sonst ist ein kleiner Test der ehrlichere Weg.

## Verwandte Demos im Portfolio

- [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo): derselbe Servicegrad aus der Bestandsregel, bei zufälliger Zuteilung.
- [bandit-vs-ab-demo](https://github.com/sebastian-hanisch/bandit-vs-ab-demo): Zuteilung nach den bisherigen Ergebnissen und deren
  Gewichtung.
- [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo): die Base-Stock-Regel des Vehikels.

## Verifikation

- **Schätzer** (`tests/test_estimators.py`): von Hand an einer Mini-Instanz (IPS 0,6875, SNIPS 0,846, DM 0,65, DR 0,78125); die
  Zell-Fassung gegen die Zeilen-Fassung auf Zufallslogs; SNIPS undefiniert ohne Treffer; Protokoll-Zeilen summieren auf 1;
  saturiertes Modell reproduziert die Zellmittel, DR = DM.
- **Orakel** (`tests/test_oracle_off_policy.py`): IPS erwartungstreu durch vollständige Aufzählung (exakte Brüche, Logs der Länge 1
  bis 3); Doppelt robust erwartungstreu bei falschem Modell, Direkte Methode nicht; Gewichtsvarianz 1/ρ − 1 und ρ^−H − 1;
  Servicegrad des Vehikels gegen die exakte Formel der Order-up-to-Regel.
- **Pool, Protokolle, Mehrstufig** (`tests/test_evaluation.py`, `tests/test_multistep.py`): Wert von Hand, feste Klassengrößen,
  Reproduzierbarkeit, mehrstufig bei H = 1 gleich der einstufigen Fassung, Bestand trägt über die Wochengrenze.
- **Befunde:** jede Zahl dieses READMEs in `tests/test_claims.py`.
- **Presets, Oberfläche, Footer:** vollständig, Permalink eingerastet, AppTest-Rauchtest, Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: ein Protokoll, tausend Protokolle, Sweeps, mehrstufig, Grenzen |
| `ope_constants.py` | Vehikel, Entscheidungsproblem, Regler-Grenzen, Presets |
| `ope_rng.py` | SplitMix64 |
| `ope_world.py` | Nachfrage, Tagesablauf, Servicegrad und Reichweite je Standort-Woche |
| `ope_estimators.py` | IPS, SNIPS, Direkte Methode, Doppelt robust, Modelle, Protokoll |
| `ope_evaluation.py` | Pool, Wahrheit, Protokolle ziehen, Zusammenfassungen, ein Protokoll |
| `ope_multistep.py` | Trajektorien über H Wochen, mehrstufige Schätzer, Zahlenblock |
| `ope_presets.py`, `ope_visualization.py` | Permalink, Presets; Plotly-Abbildungen (Achsen gesperrt) |
| `generate_precomputed.py`, `precomputed_pool.json`, `precomputed_multistep.json` | Pool (18 Zellen à 10 000), mehrstufiger Zahlenblock |
| `tests/` | Schätzer, Orakel, Pool, Mehrstufig, Presets, Befunde, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Kreuz-Anpassung** (Cross-Fitting) des Modells und **Kappung** der Gewichte: nicht gebaut.
- **Stetige Kontexte** und gelernte Modelle (Boosting, neuronale Netze): nicht gebaut, der Kontext ist diskret.
- **Geschätzte Protokolle** (Gewichte aus einem Klassifikator): hier ist das Protokoll bekannt.
- **Konfidenzintervalle** für die Schätzer: nicht gebaut.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (Verteilungsfunktionen).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Experimente und Wirkungsmessung: A/B bis Logdaten](https://sebastianhanisch.net/konzepte-experimente-wirkungsmessung.html).
