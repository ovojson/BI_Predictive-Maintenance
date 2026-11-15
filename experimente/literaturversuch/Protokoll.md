# Literaturversuch · Regelbäume, Betriebspunkte und Merkmalskombinationen

**15. November 2026 · Europe/Berlin.** Die Aufgabenstellung `bi-group-work-classification.pdf`, insbesondere S. 4, legt die Bewertung mit FP + 6 × FN fest. Sämtliche Trainings- und Validierungsdaten stammen aus `modeling.csv`. Die zwei Veröffentlichungen dienen als Methodenquellen. Für die Versuche wurden keine Zahlenwerte, Tabellenzeilen, Labels, Generatorregeln oder Schwellen aus ihnen übernommen. Die Challenge-Datei blieb außerhalb der Auswertung.

## Übertragene Ideen

| Quelle | Idee im vorliegenden Versuch |
|---|---|
| Stephan Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications*, § III-B, gedruckte S. 71 | Flache Entscheidungsbäume auf verschiedenen Merkmalsgruppen. Jeder Baum lernt seine Grenzwerte und Zusatzalarme auf den jeweiligen Trainingsdaten. |
| Matzka, § IV, S. 71–73 | Standardisierte Abweichungen helfen bei der Diagnose. Im eigenen Versuch wurden Abweichungen von einem **bedingten** Betriebspunkt gebildet, mit einem Regressionsfit auf normalen, freigegebenen Trainingsvorgängen. |
| Ahmed Samet, *Data Mining Driven Approaches for Predictive Maintenance, Inventive Design Modelling and Energy Storage Systems*, Habilitationsschrift, § 4.1 und § 4.2.2–4.2.3, S. 58 und 67–70 | Übertragung der Mustersuche auf gleichzeitig beobachtete Merkmale innerhalb einer Zeile. Chronicle Mining im engeren Sinn benötigt Zeitstempel und geordnete Ereignisse. Die CSV enthält Einzelvorgänge ohne diese Zuordnung. |

## Kleine Bäume und bedingte Betriebspunkte

Der Lauf von **20:52:40 bis 20:53:25 Uhr** testete 24 Varianten. Vier Merkmalsgruppen (Kühlung, Belastung, bedingter Betriebspunkt, Kombination) wurden mit drei Baumtiefen (1 / 2 / 3) und zwei Mindestblattgrößen (30 / 75) kombiniert. Diese Einstellungen sind lokale Versuchsparameter. Die Zusatzmodelle trainieren auf Vorgängen ohne Hybridalarm und können bestehende Alarme nur ergänzen.

Die erste Betriebspunktabweichung misst, wie weit die beobachtete Drehzahl von der bei gegebenem Drehmoment und Produkttyp erwarteten Drehzahl entfernt liegt. Die zweite berechnet dasselbe für den Temperaturabstand bei gegebener Lufttemperatur, Drehzahl und Produkttyp. Für beide werden Trend und robuste Streuung in jeder Falte aus dem normalen Trainingsrest geschätzt.

| Version | Mittlere Kosten | Fehlalarme Ø | Übersehen Ø |
|---|---:|---:|---:|
| Bisheriges Hybridmodell | **237,6** | 25,2 | 35,4 |
| Kühlung, Belastung oder Kombination bei separater innerer Auswahl | 237,6 | 25,2 | 35,4 |
| Betriebspunkt bei separater innerer Auswahl | 241,8 | 30,6 | 35,2 |
| Freie innere Auswahl über alle 24 Bäume | 241,8 | 30,6 | 35,2 |

Merkmalsgruppe, Baumparameter und Schwelle werden in der inneren stratifizierten 3-Fold-CV gewählt. Die äußere Auswertung nutzt dieselben fünf stratifizierten 5-Fold-Wiederholungen wie das bisherige Modell. In diesen Wiederholungen erkennt die freie Auswahl insgesamt einen zusätzlichen Ausfall und erzeugt 27 zusätzliche Fehlalarme. Der Kontrolllauf von **20:54:15 bis 20:55:07 Uhr** reproduzierte die Ranglisten und Faltenergebnisse. Die Einzelergebnisse und ausgewählten Baumregeln stehen in `details.json`.

## Statische Merkmalsregeln

Der Lauf von **20:54:37 bis 20:54:56 Uhr** prüft, ob eine kurze UND-Kombination mehr leistet als ein kleiner Baum, der seine Splits nacheinander lernt. Eine Regel besteht aus bis zu zwei gleichzeitig erfüllten Bedingungen. Die Grenzwerte folgen aus Quantilen des freigegebenen Trainingsrests. Die Suche maximiert die eingesparten Kosten `6 × zusätzliche Treffer − zusätzliche Fehlalarme`. Die innere CV vergleicht Regelkomplexität und Mindestanzahl abgedeckter Trainingsausfälle mit der Referenz.

In **allen 25 äußeren Falten** fiel die innere Wahl auf das unveränderte Hybridmodell. Der Mittelwert blieb bei **237,6** Kosteneinheiten. Bei fester Vorgabe einer Suche mit bis zu zwei Bedingungen und mindestens fünf Trainingsausfällen wurden über die fünf Wiederholungen neun zusätzliche Treffer und 115 zusätzliche Fehlalarme gezählt. Diese Einstellung kostete im Mittel **249,8**. Der Kontrolllauf von **20:55:33 bis 20:55:46 Uhr** reproduzierte die Regeln, Rangliste und Faltenergebnisse. Die vollständigen Ergebnisse liegen unter `experimente/merkmalregeln`.

## Erklärbarkeit für Bericht und Pitch

Die technische Dokumentation kann Matzkas Erklärungsansatz auch bei beibehaltener Prognose nutzen. Für jeden Hybridalarm lässt sich benennen, welche tatsächlich geprüfte Regel gegriffen hat und wie weit der beobachtete Messwert von der in der Trainingsfalte gelernten Grenze entfernt war. Ist nur die Verschleißstufe aktiv, lässt sich ihr Score mit der Alarmschwelle ausgeben. Eine große Abweichung kann Anlass zur Inspektion sein. Ob sie zur Entscheidung beigetragen hat, ergibt sich aus der tatsächlich ausgelösten Regel oder dem verwendeten Score.
