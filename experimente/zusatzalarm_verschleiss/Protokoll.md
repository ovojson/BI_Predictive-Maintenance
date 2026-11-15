# Versuchsprotokoll · Zusatzalarme für verbleibende Ausfälle

**15. November 2026 · Europe/Berlin.** Die Experimente verwenden die 8.000 beschrifteten Vorgänge aus `modeling.csv`. In allen fünf äußeren 5-Fold-Wiederholungen behalten beide Ansätze sämtliche Alarme des bisherigen Hybrids bei. Nur bisher freigegebene Vorgänge erhalten einen möglichen Zusatzalarm. Die innere 3-Fold-CV wählt Merkmale, Einstellungen und Schwellen nach FP + 6 × FN. Bei Gleichstand ist eine Schwelle ohne Zusatzalarm zulässig.

## 1. Glatte empirische Ausfallrate

Der Lauf von **20:39:03 bis 20:39:14 Uhr** betrachtet für jede Standzeit die Fall- und Ausfallzahlen bei ähnlichen Standzeiten. Ein Gauß-Gewicht glättet diese Zahlen mit Breiten von 1,5 / 3 / 6 / 12 Minuten. In vier der acht Konfigurationen zählen Beobachtungen anderer Produktvarianten halb. Über acht Pseudobeobachtungen stabilisiert die mittlere Ausfallrate des jeweiligen Trainingsrests die Schätzung in dünn besetzten Bereichen. Der Wiederholungslauf von **20:40:05 bis 20:40:13 Uhr** bestätigte alle Kennzahlen.

Die vollständige innere Auswahl erreichte **241,6** statt **237,6** Kosteneinheiten. Über die fünf Wiederholungen ergaben sich **ein zusätzlicher Treffer und 26 zusätzliche Fehlalarme**. Die im Nachhinein beste feste Breite (6 Minuten, Standzeit allein) ergab **237,4**. Ihre gesamte Verbesserung stammt aus einer Aufteilung, in der sie einen Ausfall zusätzlich erkannte und fünf Fehlalarme hinzufügte. In den vier anderen Aufteilungen blieben die Entscheidungen mit dieser Breite identisch zur Referenz. Im Standzeitbereich 200–219 Minuten gab es mit dieser Breite keinen zusätzlichen Treffer.

## 2. Ähnliche Betriebspunkte

Der Lauf von **20:40:34 bis 20:41:14 Uhr** schätzt das Risiko aus ähnlichen Vorgängen. Das Nachbarschaftsmodell verknüpft Standzeit, Drehmoment, Produkttyp und Temperatur mit Drehzahl oder einem Leistungsmerkmal aus Drehmoment × Drehzahl. Die Merkmale werden in jeder Trainingsfalte skaliert. Getestet wurden 8 / 24 / 64 Nachbarn, drei Standzeitgewichte und beide Merkmalsgruppen. Der beste Kandidat und seine Schwelle werden in den inneren Falten gewählt.

Die vollständige innere Auswahl erreichte **243,0** Kosteneinheiten mit **einem zusätzlichen Treffer und 33 zusätzlichen Fehlalarmen** über die fünf Wiederholungen. Die besten festen Konfigurationen lagen gleichauf mit 237,6 und lösten dabei keine zusätzlichen Alarme aus.

## Abgabeentscheidung

Die beobachtete Verbesserung von 0,2 Kosteneinheiten im Mittel beruht auf der nachträglichen Wahl einer Glättungsbreite und einem einzelnen wiederholt untersuchten Datensatz. Für 6:1 bleibt der bisherige Hybrid die belastbarere Wahl. Zeitreihen über Werkzeugzyklen oder zusätzliche Sensormessungen könnten helfen, die verbleibenden Ausfälle gezielter zu erkennen.
