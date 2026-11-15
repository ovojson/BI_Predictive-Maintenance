# Versuchsprotokoll · Verschleißkurve und Wechselwirkungen

**15. November 2026 · Europe/Berlin.** Der Versuch lief von **20:32:48 bis 20:33:21 Uhr** mit den 8.000 Vorgängen aus `modeling.csv`. Regelgrenzen, Modellparameter und Alarmschwellen wurden innerhalb der Trainings- und inneren Validierungsfalten bestimmt. Der Wiederholungslauf von **20:33:53 bis 20:34:17 Uhr** reproduzierte alle Kennzahlen und beide CSV-Ranglisten.

## Aufbau

- Der physikalische Kern wurde je Trainingsfalte wie im bisherigen Hybridmodell gelernt. Seine Alarme gelten in allen Varianten. Die neue Logistik trainiert auf Fällen ohne Regelalarm.
- Der erste Abschnitt beginnt am je Falte gelernten Verschleißbeginn. Zwanzig Minuten später kann sich die Steigung des Logits ändern. Die Produktvariante bleibt als Grundmerkmal erhalten.
- Getestet wurden Kurve allein, Kurve mit Standzeit × Drehmoment, Kurve mit Standzeit × Produktvariante sowie Kurve mit beiden Wechselwirkungen. Die Drehmomentvarianten enthalten zusätzlich Drehmoment als Haupteffekt. Interaktionen nutzen die Standzeit oberhalb des gelernten Verschleißbeginns. Drehmoment und Produktstufe werden mit Mittelwerten aus der jeweiligen Trainingsfalte zentriert.
- Vier Regularisierungen je Merkmalsgruppe ergeben **16 neue Konfigurationen**. Die innere stratifizierte 3-Fold-CV wählt C (0,01 / 0,1 / 1 / 10) und Alarmschwelle anhand von FP + 6 × FN. Die äußere Bewertung verwendet dieselben fünf stratifizierten 5-Fold-Wiederholungen wie der vorherige Versuch. Als Kontrollen dienen die bisherige Schwelle von 1/7 und die innerlich optimierte Schwelle des bisherigen Modells.

## Ranking

Die Kosten gelten jeweils pro Wiederholung mit 8.000 Vorhersagen. Bei den neuen Varianten wurden C und Schwelle in jeder äußeren Trainingsfalte neu gewählt. Bei der gemeinsamen Auswahl kam zusätzlich die Wahl der Merkmalsgruppe hinzu.

| Rang | Version | Kosten Ø | Fehlalarme Ø | Übersehen Ø |
|---:|---|---:|---:|---:|
| 1 | **Bisheriges Hybridmodell** | **237,6** | 25,2 | 35,4 |
| 2 | Bisheriges Hybridmodell mit optimierter Schwelle | 248,2 | 19,0 | 38,2 |
| 3 | Kurve + Produktvariante | 258,2 | 38,6 | 36,6 |
| 4 | Kurve allein | 259,8 | 34,2 | 37,6 |
| 5 | Kurve + beide Wechselwirkungen | 262,6 | 28,6 | 39,0 |
| 6 | Verschleißkurve mit gemeinsamer innerer Auswahl | 266,4 | 36,0 | 38,4 |
| 7 | Kurve + Drehmoment | 266,6 | 37,4 | 38,2 |

Die beste einzelne neue Konfiguration in der nachträglichen Sichtung verwendet beide Wechselwirkungen und C = 0,1. Sie liegt bei 254,2 Einheiten. Die Auswahlstrategie, die Merkmalsgruppe, C und Schwelle vollständig in den inneren Falten bestimmt, liegt bei 266,4 Einheiten.

## Was bei den Fehlfällen passiert

Auf der ursprünglichen Testaufteilung (Seed 2006) erkennt die gemeinsame Auswahl **keinen der 36** zuvor übersehenen Ausfälle zusätzlich. Sie verliert vier bisherige Treffer und löst netto einen weiteren Fehlalarm aus. Die Kosten steigen damit von **240 auf 265**.

Über alle Wiederholungen gewinnt diese Auswahl durchschnittlich **0,8** Treffer und verliert **3,8**. Bei 200–219 Minuten gewinnt sie im Schnitt **0,2** Treffer. Das ist ein zusätzlicher Treffer in einer der fünf Aufteilungen, die sich dieselben 271 beschrifteten Ausfälle teilen. Die beste neue Merkmalsgruppe, Produktvariante, gewinnt in diesem Standzeitband in keiner Aufteilung einen Treffer hinzu.

Die Tests stützen den bisherigen Hybrid für die 6:1-Wertung. Der Befund gilt für die getestete Kurvenform und Interaktionen. Die Challenge-Datei floss in diesen Versuch nicht ein.

## Reproduktion

Im Projektordner `python -X utf8 verschleissversuch.py` ausführen. `rangliste.csv` zeigt auch die fallweisen Gewinne und Verluste im Mittel, `konfigurationen.csv` alle 16 neuen Konfigurationen und den Kontrolllauf. `details.json` enthält Zeiten, Konfigurationen, Faltenergebnisse und die gewählten Schwellen.
