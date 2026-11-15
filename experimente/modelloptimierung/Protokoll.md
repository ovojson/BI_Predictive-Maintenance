# Versuchsprotokoll · Modelloptimierung

**15. November 2026, Uhrzeit Europe/Berlin.** Verwendet wurden die 8.000 Vorgänge und 271 Ausfälle aus `modeling.csv`. Die Kostenfunktion lautet FP + 6 × FN.

| Zeit | Was durchgeführt wurde | Wie |
|---|---|---|
| 20:05:25–20:05:27 | Referenz reproduziert und Fehlfälle untersucht | Für jede äußere Testfalte wurden die physikalischen Grenzen aus ihrer Trainingsfalte geschätzt. Verglichen wurden übersehene Ausfälle und korrekt freigegebene Vorgänge. |
| 20:08:10–20:09:39 | Schwellen getestet, Restmodelle verglichen und Ranking erstellt | Fünf stratifizierte 5-Fold-Wiederholungen mit Seeds 2006, 2026, 2007, 888 und 1024. Innere 3-Fold-CV wählte Parameter und Alarmschwellen. Bei jedem Fit blieben die Alarme des physikalischen Kerns aktiv. |
| 20:09:39 | Sieger auf allen beschrifteten Daten trainiert und Challenge vorhergesagt | Eingabereihenfolge geprüft. Die CSV enthält 2.000 Zeilen und 63 Alarme. |
| 20:10:48–20:12:11 | Gesamten Vergleich erneut ausgeführt und geprüft | Alle vier Ergebnistabellen wurden bytegenau reproduziert. Die erneute Challenge-CSV ist bytegleich mit der ursprünglichen Abgabedatei. |

## Fehlfälle

Bei Seed 2006 reproduziert die Referenz **235 Treffer, 24 Fehlalarme und 36 übersehene Ausfälle**. Unter den übersehenen Ausfällen liegt **keiner** innerhalb der untersuchten physikalischen Grenzbänder. Die Bänder umfassen bis zu 100 W Abstand zu einer Leistungsgrenze, gleichzeitig höchstens 0,1 K und 25 U/min Abstand zur Kühlungsregion oder höchstens 500 Nm·min Abstand zur typabhängigen Belastungsgrenze. Bei doppelter Bandbreite liegen **2 übersehene Ausfälle und 291 korrekt freigegebene Vorgänge** in mindestens einem Band. Das wiederholt sich bei allen fünf Seeds.

Bei Seed 2006 liegen 23 übersehene Ausfälle bei 200–219 Minuten Werkzeugstandzeit. 20 der 36 haben einen Verschleißscore von höchstens 1 %. Eine pauschale Vorverlegung des Verschleißalarms um zehn Minuten hätte bei sonst gleichen Werten sechs Ausfälle zusätzlich erfasst und 71 weitere Fehlalarme verursacht. Das ergibt 35 zusätzliche Kosteneinheiten.

## Getestete Konfigurationen

- **Schwellenvergleich.** Das bestehende Hybridmodell mit fester Schwelle 1/7 wurde gegen dieselbe Verschleißlogistik mit kostenoptimierter Schwelle aus den inneren Falten getestet.
- **Logistik mit Grenzabständen.** Zwölf Konfigurationen aus vier Regularisierungen (C = 0,01 / 0,1 / 1 / 10) und drei Merkmalsgruppen. Zur Werkzeugstandzeit und Produktvariante kommen entweder der Belastungsabstand, alle relevanten Abstände oder eine Transformation, die Grenznähe betont.
- **Histogram Gradient Boosting mit Grenzabständen.** Zwölf Konfigurationen aus zwei Merkmalsgruppen (Belastung oder alle relevanten Abstände), drei Tiefen (1 / 2 / 3) und zwei Mindestblattgrößen (20 / 60). Lernrate 0,06 und 90 Iterationen.

## Ranking bei 6:1

Mittelwerte pro Wiederholung mit jeweils 8.000 Testvorhersagen. Für Zeilen 2, 4 und 5 bilden die äußeren Testfalten die vollständige innere Auswahlstrategie ab.

| Rang | Version | Mittlere Kosten | Kosten je Seed | Fehlalarme Ø | Übersehen Ø |
|---:|---|---:|---|---:|---:|
| 1 | Hybrid aktuell · Logistik C = 1, Schwelle 1/7 | **237,6** | 240 / 237 / 242 / 235 / 234 | 25,2 | 35,4 |
| 2 | Hybrid mit optimierter Schwelle | 248,2 | 240 / 259 / 251 / 239 / 252 | 19,0 | 38,2 |
| 3 | Physikalische Regeln | 252,0 | 252 / 252 / 252 / 252 / 252 | 0,0 | 42,0 |
| 4 | Gradient Boosting mit Grenzabständen | 261,4 | 258 / 260 / 275 / 255 / 259 | 13,0 | 41,4 |
| 5 | Logistik mit Grenzabständen | 263,8 | 261 / 270 / 268 / 258 / 262 | 32,2 | 38,6 |

Das bestehende Hybridmodell liegt in allen fünf Wiederholungen vorn, einmal gleichauf mit der optimierten Schwelle. Unter den 24 neuen Einzelkonfigurationen liegt HGB mit Belastungsabstand, Tiefe 2 und Mindestblattgröße 60 bei 252,6 Einheiten vorn. Diese Platzierung beruht auf der Sichtung derselben äußeren Ergebnisse. Alle Wiederholungen verwenden dieselben 271 Ausfälle in unterschiedlichen Aufteilungen.

## Recall-Ziele

Die Schwelle für jedes Ziel wurde aus inneren Out-of-Fold-Scores gewählt. Die Tabelle zeigt den **tatsächlich erreichten** Recall und die Kosten auf den äußeren Testfalten für die bestehende Verschleißlogistik mit der jeweiligen Zielschwelle. Die Ergebnisse für die beiden Varianten mit Grenzabständen stehen in `recall_ziele.csv`.

| Inneres Recall-Ziel | Recall außen Ø | Fehlalarme Ø | Kosten Ø |
|---:|---:|---:|---:|
| 90 % | 90,18 % | 159,6 | 319,2 |
| 95 % | 95,42 % | 655,4 | 729,8 |
| 100 % | 99,78 % | 7.349,8 | 7.353,4 |

Die 100-%-Zielschwelle erkennt alle Ausfälle in drei der fünf Wiederholungen. In den übrigen zwei Wiederholungen übersieht sie zwei beziehungsweise einen Ausfall. Die nachträgliche Schwellenkurve für Seed 2006 erreicht vollständigen Recall bei mindestens **6.882 Fehlalarmen**. Dieser Wert beschreibt die Scores auf dieser Aufteilung. Um bei vollständigem Recall die Referenzkosten von 240 zu unterbieten, wären höchstens 239 Fehlalarme zulässig gewesen.

## Nachvollziehen

`rangliste.csv` enthält die fünf Modellversionen, `konfigurationen.csv` alle 25 trainierbaren Varianten mit innerer Schwellenwahl. `details.json` enthält Zeitstempel, die ausgewählten Konfigurationen pro Falte, Schwellen und Einzelkosten. Die Diagnose der übersehenen Ausfälle steht in `fehleranalyse.json`. Über `schwellenkurve_seed2006.csv` lässt sich der nachträgliche Schwellenverlauf nachvollziehen.
