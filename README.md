# Predictive Maintenance – Gruppenarbeit Classification

Das Hybridmodell kombiniert gelernte physikalische Alarmregeln mit einem logistischen Verschleißmodell. In fünf Wiederholungen einer stratifizierten 5-fachen Kreuzvalidierung entstehen im Mittel **237,6 Kosteneinheiten** gegenüber **1.626** bei durchgehender Freigabe. Auf den 2.000 Challenge-Vorgängen erzeugt das Modell **63 Alarme**.

## Modell lokal ausführen

Für den Ablauf im Stil der Vorlesung gibt es `classification_hybrid.ipynb`. Das Notebook liegt neben `modell.py` und `physik.py` und nutzt deren Modellcode. Mit `python -m pip install -r requirements-notebook.txt` installierst du die Pakete für den Python-Kernel. Öffne das Notebook im Projektordner, wähle diesen Kernel aus und führe die Zellen der Reihe nach aus. Gruppenname und Dateipfade stehen in der ersten Codezelle. Die Notebook-Prognose landet standardmäßig in `results/`.

Python 3.11 oder neuer. Die Resultate wurden mit Python 3.13, NumPy 2.5.1, pandas 3.0.5 und scikit-learn 1.9.0 erzeugt.

```powershell
python -m pip install -r requirements.txt
python -X utf8 modell.py --gruppe "MeinGruppenname"
```

Die beiden Eingabedateien werden standardmäßig im persönlichen Ordner `Downloads` gesucht. Abweichende Speicherorte lassen sich mit `--modeling` und `--challenge` angeben. Der Ausgabepfad ist standardmäßig `abgabe`. Mit `--nur-prognose` wird das Modell auf allen 8.000 beschrifteten Zeilen trainiert und schreibt die Vorhersage direkt.

```powershell
python -X utf8 modell.py --gruppe "MeinGruppenname" --nur-prognose
```

Die Ausgabedatei hat genau die Spalte `Machine failure`. Die Werte 0 und 1 entsprechen zeilenweise der Eingabereihenfolge. Der vollständige Lauf schreibt zusätzlich `validierung.json` mit Kosten, Fehlertypen, Ablationen und Prognosebereich.

Der erweiterte Vergleich enthält 16 Baumvarianten. Mit `--alle` kommen der Präzisionsvergleich, das Hybridmodell, die Varianten des physikalischen Regelmodells und die aus dem früheren Chat rekonstruierbaren Kandidaten hinzu. `--liste` zeigt die Namen und `--modell` testet eine Variante gezielt.

```powershell
python -X utf8 modellvergleich.py "$env:USERPROFILE/Downloads/modeling.csv" --erweitert
python -X utf8 modellvergleich.py "$env:USERPROFILE/Downloads/modeling.csv" --family rf
```

CatBoost und LightGBM benötigen die Zusatzpakete aus `requirements-vergleich.txt`. Die früheren Trainingsparameter und Ensemblegewichte liegen im Chat-Auszug nicht vor. Die lokalen Rekonstruktionen verwenden dokumentierte Einstellungen im Code und ein gleich gewichtetes Ensemble. Der Random Forest mit Schwelle `0.399` verwendet die vorhandene lokale RF-Konfiguration mit physikalischen Merkmalen.

```powershell
python -m pip install -r requirements-vergleich.txt
$daten = "$env:USERPROFILE/Downloads/modeling.csv"
python -X utf8 modellvergleich.py $daten --alle --liste
python -X utf8 modellvergleich.py $daten --alle --modell "CatBoost + LightGBM"
python -X utf8 modellvergleich.py $daten --alle
```

## Modelloptimierung reproduzieren

Die wiederholte verschachtelte Validierung vergleicht das bestehende Hybridmodell, eine aus den Trainingsfalten gewählte Schwelle und zwei Restmodelle mit Grenzabständen. Die getesteten Parameter stehen in `GRIDS` in `modelloptimierung.py`.

```powershell
python -X utf8 fehleranalyse.py
python -X utf8 modelloptimierung.py --challenge "$env:USERPROFILE/Downloads/challenge.csv"
```

Die Ergebnisse und das Protokoll liegen unter `experimente/modelloptimierung`. Mit `--ausgabe` lässt sich für weitere Versuche ein eigener Ordner angeben. Ein Lauf ohne `--challenge` erstellt nur die Validierung und wählt die finale Konfiguration.

## Verschleißkurve und Wechselwirkungen testen

`verschleissversuch.py` vergleicht die bisherige Verschleißlogistik mit einer Kurve, deren Steigung sich ab 20 Minuten nach dem gelernten Verschleißbeginn ändern kann. Getestet werden die Kurve allein und jeweils mit Wechselwirkungen zwischen Standzeit und Drehmoment, Standzeit und Produktvariante oder beiden. `VARIANTS`, `KNOT_OFFSET` und `GRID` stehen am Anfang der Datei.

```powershell
python -X utf8 verschleissversuch.py
```

Der Lauf verwendet `modeling.csv` und schreibt sein Protokoll sowie die Ergebnisse nach `experimente/verschleisskombinationen`. In der Rangliste zeigen `new_hits` und `lost_hits`, welche Treffer gegenüber der Referenz gewonnen oder verloren wurden. Der Zusatz `200_219` beschränkt die Zählung auf diesen Standzeitbereich.

Zwei weitere Versuche prüfen ergänzende Alarme durch geglättete Verschleißraten und ähnliche Betriebspunkte. Sie behalten die Alarme des bisherigen Hybrids bei.

```powershell
python -X utf8 zusatzalarm_verschleiss.py
python -X utf8 zusatzalarm_nachbarschaft.py
```

Die gemeinsame Auswertung steht in `experimente/zusatzalarm_verschleiss/Protokoll.md`.

## Literaturversuch wiederholen

Die Literaturversuche prüfen kleine Regelbäume, Abweichungen von typischen Betriebspunkten und kurze Regeln aus gleichzeitig vorliegenden Messwerten. Sie lernen alle Modellgrößen aus `modeling.csv` und bewerten Zusatzalarme bei 6:1.

```powershell
python -X utf8 literaturversuch.py
python -X utf8 merkmalregeln.py
```

Quellen, Versuchsschritte und Ergebnisse stehen in `experimente/literaturversuch/Protokoll.md`.

## Abgabe

Das ZIP unter `abgabe/Abgabepaket_BI.zip` enthält die Einreichungsdateien, den reproduzierbaren Code, die Versuchsprotokolle und die einseitige `Management_Summary.pdf` für die Gruppe.

1. Gruppenname und Mitglieder in Bericht und Hilfsmittelverzeichnis ergänzen sowie die Titelfolie anpassen. Für die CSV erledigt `python -X utf8 modell.py --gruppe "MeinGruppenname" --nur-prognose` die Benennung.
2. Das Hilfsmittelverzeichnis fachlich prüfen und die tatsächlich genutzten weiteren Hilfsmittel ergänzen. Die Arbeitsschritt-Tabelle und der Beleganhang orientieren sich am bereitgestellten Leitfaden, Stand 08.12.2025. Die editierbaren Vorlagen heißen `abgabe/Hilfsmittelverzeichnis.md` und `abgabe/KI_Dialoge.md`. Eine gegebenenfalls in Moodle geforderte Eigenständigkeitserklärung von den Gruppenmitgliedern prüfen und unterzeichnen lassen.
3. Technischen Bericht, Präsentation, Hilfsmittelverzeichnis mit KI-Dialoganhang, Code und die passend benannte Challenge-Prognose nach den Moodle-Vorgaben bis einschließlich 25. November 2026 einreichen.

Die lokale Vorhersage mit Platzhalter heißt `challenge_prediction_GRUPPENNAME.csv`. Die Präsentation enthält den vierminütigen Pitch, die Vertiefung für zehn Minuten und Sprechernotizen.

Um den PDF-Bericht unter Windows mit den Gruppenangaben zu setzen, einmal ReportLab installieren und das Skript im Projektordner ausführen.

```powershell
python -m pip install reportlab
python -X utf8 bericht_erstellen.py --gruppe "MeinGruppenname" --mitglieder "Vorname Name, Vorname Name"
python -X utf8 uebersichten_erstellen.py
```

Das letzte Skript setzt die Management Summary aus `validierung.json` sowie Hilfsmittelverzeichnis und KI-Dialoganhang aus den Markdown-Vorlagen erneut als PDF.
