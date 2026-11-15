"""Build the technical report from the locally verified validation output."""

import argparse
import csv
import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# Gruppenangaben und gespeicherte Ergebnisse für den Abgabebericht laden.
parser = argparse.ArgumentParser(description="Technischen Bericht als PDF setzen.")
parser.add_argument("--gruppe", default="[Gruppenname]")
parser.add_argument("--mitglieder", default="[Mitglieder ergänzen]")
args = parser.parse_args()
project = Path(__file__).resolve().parent
folder = project / "abgabe"
data = json.loads((folder / "validierung.json").read_text(encoding="utf-8"))


def experiment_cost(experiment, version):
    """Liest den Kostenmittelwert einer Variante aus der gespeicherten Rangliste."""
    with (project / "experimente" / experiment / "rangliste.csv").open(
        encoding="utf-8", newline=""
    ) as result:
        for row in csv.DictReader(result):
            if row["version"] == version:
                return f"{float(row['kosten_mittel']):.1f}".replace(".", ",")
    raise ValueError(f"Variante {version!r} fehlt im Versuch {experiment!r}")

# Farben, Schriften und Abstände für die PDF-Seiten definieren.
ink = colors.HexColor("#142339")
blue = colors.HexColor("#1A5470")
neutral = colors.HexColor("#425466")
mist = colors.HexColor("#EDF4F7")
rule = colors.HexColor("#D2DEE5")
white = colors.white
for name, file in [("Segoe", "segoeui.ttf"), ("Segoe Bold", "segoeuib.ttf")]:
    pdfmetrics.registerFont(TTFont(name, f"C:/Windows/Fonts/{file}"))
pdfmetrics.registerFontFamily("Segoe", normal="Segoe", bold="Segoe Bold")

styles = {
    "small": ParagraphStyle("small", fontName="Segoe", fontSize=9, leading=13, textColor=neutral, spaceAfter=8),
    "hero": ParagraphStyle("hero", fontName="Segoe Bold", fontSize=32, leading=36, textColor=ink, spaceAfter=15),
    "h1": ParagraphStyle("h1", fontName="Segoe Bold", fontSize=25, leading=31, textColor=ink, spaceAfter=14),
    "h2": ParagraphStyle("h2", fontName="Segoe Bold", fontSize=13, leading=17, textColor=blue, spaceBefore=9, spaceAfter=5, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Segoe", fontSize=11, leading=16.1, alignment=TA_LEFT, textColor=ink, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="Segoe", fontSize=9.5, leading=13.5, textColor=ink),
    "head": ParagraphStyle("head", fontName="Segoe Bold", fontSize=9.5, leading=13.5, textColor=white),
    "kpi": ParagraphStyle("kpi", fontName="Segoe", fontSize=10, leading=15, textColor=ink),
}
story = []  # Geordnete Elemente, aus denen ReportLab anschließend das PDF setzt.


def p(text, style="body"):
    # Absatz mit der gewählten Formatierung zur Seite hinzufügen.
    story.append(Paragraph(text, styles[style]))


def table(header, rows, widths, padding=7):
    # Kopfzeile und Inhalt formatieren, Spaltenbreiten festlegen und Abstand setzen.
    content = [[Paragraph(str(x), styles["head"]) for x in header]]
    content += [[Paragraph(str(x), styles["cell"]) for x in row] for row in rows]
    grid = Table(content, colWidths=widths, hAlign="LEFT", repeatRows=1)
    grid.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ink),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [mist, white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("LINEBELOW", (0, -1), (-1, -1), .5, rule),
    ]))
    story.extend([grid, Spacer(1, 10)])


def section(title, ki_evidence):
    # Jede Hauptsektion beginnt auf einer neuen Seite.
    story.append(PageBreak())
    p(f"{title} <font size='9' color='#1A5470'>{ki_evidence}</font>", "h1")


# Titel, Gruppenangaben und Übersicht der wichtigsten Kennzahlen.
p("TECHNISCHER BERICHT  /  BUSINESS INTELLIGENCE  /  WISE 2026/27", "small")
p("KI-unterstützter Entwurf mit OpenAI Codex [KI-4]. Die Nachweise [KI-1] bis [KI-4] "
  "stehen im Anhang KI_Dialoge.pdf. Einsatz nach Arbeitsschritten im Hilfsmittelverzeichnis.", "small")
p("Predictive Maintenance", "hero")
p(f"Gruppenarbeit Classification · Technische Hochschule Mannheim<br/>"
  f"{escape(args.gruppe)} · {escape(args.mitglieder)}", "small")

# Die festen Berichtszahlen müssen zu den geladenen Validierungsergebnissen passen.
mean_cost = sum(run["cost"] for run in data["validation"]) / len(data["validation"])
assert abs(mean_cost - 237.6) < 1e-8 and data["challenge_alarms"] == 63
kpis = [["<font size='22'><b>237,6</b></font><br/>Kosten im CV-Mittel",
         "<font size='22'><b>−85,4 %</b></font><br/>gegenüber Freigabe aller",
         "<font size='22'><b>59,4</b></font><br/>erwartete Challenge-Kosten"]]
kpi_table = Table([[Paragraph(label, styles["kpi"]) for label in kpis[0]]],
                  colWidths=[163, 163, 163], hAlign="LEFT")
kpi_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), mist), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ("TOPPADDING", (0, 0), (-1, -1), 16), ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
]))
story.extend([kpi_table, Spacer(1, 17)])

# Datenbeschreibung und betriebliche Referenzkosten.
p("1  Explorative Datenanalyse", "h1")
p("KI-Unterstützung bei Datenanalyse und Codeentwurf [KI-1].", "small")
p("Jeder Produktionsvorgang erhält eine Entscheidung über Freigabe oder Inspektion. Die Aufgabenstellung bewertet einen Fehlalarm mit einer und einen übersehenen Ausfall mit sechs Kosteneinheiten. Optimiert wird die Summe <b>FP + 6 · FN</b>.")
p("<b>Stichprobe.</b> modeling.csv enthält 8.000 synthetische Produktionsvorgänge mit sechs Merkmalen. 271 Vorgänge enden mit Ausfall (3,39 %). Alle Eingabefelder sind belegt und alle Merkmalszeilen eindeutig. Die Ausfallquote unterscheidet sich nach Produktvariante. Die 18 Ausfälle in H begrenzen die Präzision von Untergruppenvergleichen.")
table(("Variante", "Vorgänge", "Ausfälle", "Ausfallquote"), [
    ("L", "4.808", "184", "3,83 %"),
    ("M", "2.397", "69", "2,88 %"),
    ("H", "795", "18", "2,26 %"),
], [112, 121, 121, 135])
p("<b>Prozessstruktur.</b> Die Randbereiche der mechanischen Leistung sind auffällig. In 76 Vorgängen mit Leistung unter 3,5 kW oder über 9,0 kW tritt jeweils ein Ausfall auf. Weitere Ausfälle häufen sich in einer Kombination aus kleinem Temperaturabstand und niedriger Drehzahl sowie bei hoher kumulierter Werkzeugbelastung. Diese Muster motivieren die Interaktionen auf Seite 2. Alle EDA-Zahlen stammen aus modeling.csv.")
p("<b>Geschäftliche Referenz.</b> Durchgehende Freigabe ergibt 271 übersehene Ausfälle und 1.626 Kosteneinheiten. Durchgehender Alarm ergibt 7.729 Fehlalarme und entsprechend 7.729 Einheiten. Die geringe Ausfallquote macht die Standard-Accuracy als Auswahlgröße ungeeignet.")
p("Quellen · Aufgabenstellung, S. 1 und 4 · Eigene Berechnungen aus modeling.csv", "small")

# Herleitung der Eingangsgrößen und der Grenzwerte.
section("2  Features", "[KI-1]")
p("Die sechs Originalmerkmale sind Type, Luft- und Prozesstemperatur, Drehzahl, Drehmoment und Werkzeugstandzeit. Physikalische Kombinationen machen die Randbereiche als zusammenhängende Regionen sichtbar. Die folgenden Grenzen sind die Fits auf den vollständigen Trainingsdaten. Im CV-Lauf werden sie jeweils neu geschätzt.")
table(("Merkmal", "Interpretation und Alarmgrenze", "Treffer¹"), [
    ("Leistung P = 2π · n · M / 60", "Leistungsfenster. Alarm bei P &lt; 3.500 W oder P &gt; 9.000 W.", "21 / 55"),
    ("Temperaturabstand ΔT = T<sub>Prozess</sub> − T<sub>Luft</sub>", "Kühlungsverdacht bei ΔT &lt; 8,6 K <b>und</b> Drehzahl &lt; 1.380 min⁻¹.", "90"),
    ("Belastungsindex B = Standzeit · M", "Grenze bei B &gt; 11.000, 12.000 oder 13.000 Nm·min für L, M oder H.", "79"),
    ("Verschleißsignal max(Standzeit − t₀, 0)", "Im Restkollektiv steigt das Risiko ab t₀. Die Schätzung nutzt zusätzlich die ordinale Produktvariante L=0, M=1, H=2.", "7"),
], [149, 274, 66])
p("¹ Anzahl Ausfälle im Vollfit. Die physikalischen Alarme überlappen und decken zusammen 229 Ausfälle ab. Der Verschleißalarm greift erst außerhalb dieser Vereinigung.", "small")

p("Präzision der Temperaturrechnung", "h2")
p("Der CSV-Import und die Subtraktion verwenden Float64. Bei niedriger Drehzahl gibt es 20 Zeilen mit angezeigter Differenz 8,6 K. Der tatsächlich berechnete Float64-Wert liegt in 8 Zeilen knapp unter 8,6. Alle 8 sind Ausfälle. In den 12 übrigen Zeilen liegt er knapp darüber und es gibt einen Ausfall. Der Abstand der Repräsentationen liegt in der Größenordnung 10⁻¹⁴ K. Das Muster ist ein numerischer Effekt des synthetischen Datensatzes. Für reale Sensoren sind Auflösung und Messunsicherheit maßgeblich.")

p("Verschleiß nach Abzug der physikalischen Alarme", "h2")
table(("Standzeit", "&lt; 200 min", "200–209 min", "210–224 min", "225–234 min", "≥ 235 min"),
      [("Ausfälle / Fälle", "6 / 7.202", "14 / 289", "12 / 224", "8 / 39", "2 / 17")],
      [97, 76, 80, 80, 80, 76])
p("Das höher liegende Restrisiko wird ab 200 min sichtbar. Im für zusätzliche Alarme relevanten Bereich ab 225 min ist die Stichprobe klein (56 Vorgänge, 10 Ausfälle). Das spricht für eine geglättete Risikokurve und gegen separat optimierte Grenzwerte je Variante.")

p("Herkunft der Merkmale", "h2")
p("Leistung, Kühlung und Werkzeugbelastung sind physikalisch motiviert. Die bekannte Struktur des synthetischen Datensatzes prägte die frühen Hypothesen. Die konkreten Grenzwerte entstehen aus dem Fit auf modeling.csv. Matzkas IEEE-Beitrag und Samets Habilitationsschrift lieferten später Anregungen für eigene Zusatzversuche. Verwendet wurden daraus die Modellideen. Zahlenwerte, Falltabellen, Ausfalltypen, Labels und Generatorregeln der Publikationen blieben außerhalb der Modelle.")

# Aufbau der Alarmstufen und Ablauf der Modellauswahl.
section("3  Modelle und Optimierung", "[KI-1, KI-2, KI-3]")
p("Stufe 1  Physikalische Regeln aus Trainingsdaten", "h2")
p("Das Modell sucht Schnittpunkte entlang der Leistung und des Belastungsindex sowie ein Rechteck aus Temperaturabstand und Drehzahl. Die Regeln wirken per ODER. Je Suchschritt zählen nur Vorgänge, die von anderen Regeln noch nicht erfasst sind. Der Suchlauf initialisiert symmetrisch und maximiert danach die Einsparung gegenüber Freigabe. Die finale Einstellung verlangt in den aktiven Trainingsbereichen volle Präzision.")
p("Die Belastungsgrenze hat die Form a + b · Rang(Type). Der gemeinsame Abstand b wird auf einem Raster von 0 bis 3.000 Nm·min in 100er-Schritten gesucht. So fließen die wenigen H-Beispiele gemeinsam mit den Gruppen L und M in die Schätzung ein. Innerhalb eines Intervalls mit identischer Trainingsentscheidung wird ein Wert mit grober technischer Auflösung bevorzugt. Über alle 25 Fits blieben die finalen Grenzwerte gleich.")

p("Stufe 2  Logistisches Verschleißmodell", "h2")
p("Für Fälle außerhalb der Regelvereinigung wird die früheste 10-Minuten-Gruppe mit mindestens 30 Fällen und mindestens 2 % Ausfällen als Beginn t₀ gelernt. Der Vollfit findet 200 min. Eine L2-regularisierte logistische Regression (C = 1) nutzt die standardisierte positive Differenz zur Standzeit t₀ sowie die standardisierte ordinale Variante. Der gemeinsame Varianteneffekt begrenzt die Zahl der Parameter.")
p("Für eine geschätzte Ausfallwahrscheinlichkeit p kostet die Freigabe im Erwartungswert 6p, ein Alarm 1−p. Der Alarm wird damit ab p ≥ 1/7 ausgelöst. Stufe 2 erfasst zusätzlich Fälle mit erhöhtem Verschleißrisiko. Stufe 1 behält jede zuvor ausgelöste Warnung bei.")

p("Baumbasierte Vergleichsmodelle", "h2")
p("Verglichen wurden Entscheidungsbaum, Random Forest, Extra Trees, Histogram Gradient Boosting und Mittelwerte von Scores. Die erweiterte Suche umfasst 14 Einzelkonfigurationen und zwei Ensembles. Die innere stratifizierte 3-fache CV bestimmt pro Modell die kostenminimierende Schwelle aus den inneren Out-of-Fold-Scores. Bei gleich guten Schwellen wird die Mitte des breitesten Intervalls verwendet. Für den verschachtelten Vergleich bestimmt dieselbe innere CV die Modellkonfiguration.")
p("Die gemeinsame äußere Aufteilung mit Seed 2006 erlaubt einen direkten Kostenvergleich mit dem Hybridmodell. Die ergänzend implementierten Rekonstruktionen von CatBoost und LightGBM verwenden eigenständig gesetzte Konfigurationen. Ihr Ensemble mittelt beide Scores gleichgewichtet.")

p("Zusatzversuche am Regelkern und am Hybrid", "h2")
p("Die folgenden Versuche prüfen geschätzte Schwellen, Merkmalsabstände und flexible Verschleißkurven sowie ergänzende Alarme. Die innere 3-fache CV wählt Schwellen und Modellparameter. Die Bewertung erfolgt auf denselben fünf stratifizierten 5-fach-Wiederholungen wie für das finale Hybridmodell. Die Versuche mit zusätzlichen Alarmen halten dessen Entscheidungen fest und bewerten nur die Ergänzungen.")

# Gegenüberstellung der gemessenen Fehler und Kosten.
section("4  Evaluation", "[KI-2, KI-4]")
table(("Modell · gemeinsamer Split Seed 2006", "Fehlalarme", "Übersehen", "Kosten"), [
    ("Hybrid aus Regeln und Verschleiß", "24", "36", "<b>240</b>"),
    ("Physikalische Regeln allein", "0", "42", "252"),
    ("Random Forest mit präzisionserhaltendem ΔT", "18", "42", "270"),
    ("Gewichteter Random Forest, erweiterte Suche²", "37", "39", "271"),
    ("Extra Trees, erweiterte Suche²", "42", "39", "276"),
    ("Verschachtelte Auswahl unter 16 Baumvarianten", "37", "40", "277"),
    ("Regeln mit ΔT auf eine Dezimalstelle gerundet", "0", "50", "300"),
    ("Freigabe aller Fälle", "0", "271", "1.626"),
], [280, 70, 75, 64])
p("² Beste Einzelkonfigurationen ihrer jeweiligen Familie aus der erweiterten Suche nach Sichtung der äußeren Ergebnisse. Die Auswahlstrategie in Zeile 6 liefert eine getrennte Schätzung des baumbasierten Auswahlprozesses.", "small")
p("<b>Wiederholbarkeit.</b> Fünf stratifizierte 5-fache CV-Wiederholungen ergeben 240, 237, 242, 235 und 234 Einheiten (Mittel 237,6). Jeder Vorgang wird je Wiederholung einmal außerhalb seiner Trainingsfalte prognostiziert. Der Kern allein kostet jeweils 252, separate Belastungsgrenzen 258–259. Die CV nutzt wiederholt dieselbe Kohorte. Architektur und Merkmalsideen wurden anhand dieser Kohorte entwickelt, daher bleibt die Schätzung explorativ.")
p("<b>Fehler im gemeinsamen Split.</b> 235 von 271 Ausfällen erkannt (86,7 % Recall), bei 24 Fehlalarmen (90,7 % Alarmpräzision). Die übersehenen Ausfälle verursachen 216 der 240 Einheiten. Der Recall je Variante beträgt 164/184 in L, 58/69 in M und 13/18 in H. Die H-Schätzung stützt sich auf wenige Fälle.")
p("<b>Challenge.</b> Der Vollfit erzeugt 2.000 Vorhersagen mit 63 Alarmen, davon 5 zusätzliche Verschleißalarme. Aus 25,2 Fehlalarmen und 35,4 übersehenen Fällen im CV-Mittel ergeben sich 2.000 / 8.000 · (25,2 + 6 · 35,4) = <b>59,4</b> erwartete Einheiten. Die 95-%-Prognosespanne aus 50.000 Dirichlet-Multinomial-Ziehungen liegt bei <b>25–104</b> Einheiten. Sie setzt vergleichbare, annähernd unabhängige Vorgänge voraus. Verteilungsdrift und Modellauswahl sind darin nicht abgebildet.")

section("4  Evaluation · Zusatzversuche", "[KI-2, KI-3]")
p("Die Tabelle zeigt Mittelwerte der äußeren Wiederholungen. Bei den Zusatzversuchen steht die jeweilige innere Auswahlstrategie im Vergleich. Die ursprünglichen Split-Ergebnisse von S. 4 werden davon getrennt betrachtet.")
table(("Variante", "Kosten Ø"), [
    ("Finales Hybridmodell mit fester Schwelle 1/7", "<b>237,6</b>"),
    ("Kurze UND-Regeln aus Quantilen, optionale Zusatzalarme", experiment_cost(
        "merkmalregeln", "Hybrid mit Merkmalsregel, innere Auswahl")),
    ("Geglättete Restrisiken nach Standzeit und Typ", experiment_cost(
        "zusatzalarm_verschleiss", "Hybrid mit ergänzender Risikoschätzung")),
    ("Flache Bäume mit Merkmalsgruppen und bedingten Betriebspunkten", experiment_cost(
        "literaturversuch", "Hybrid + innere Baumauswahl")),
    ("K-Nächste-Nachbarn als Zusatzalarm", experiment_cost(
        "zusatzalarm_nachbarschaft", "KNN innere Auswahl")),
    ("Hybrid mit innerlich gewählter Alarmschwelle", experiment_cost(
        "modelloptimierung", "Hybrid mit optimierter Schwelle")),
    ("Gradient Boosting / Logistik mit Grenzabständen", "261,4 / 263,8"),
    ("Verschleißkurve mit innerer Merkmals- und Parameterauswahl", experiment_cost(
        "verschleisskombinationen", "Verschleißkurve mit innerer Auswahl")),
], [407, 82], padding=5)

p("<b>Fehlerdiagnostik.</b> Auf Split 2006 liegen 23 der 36 übersehenen Ausfälle bei 200 bis 219 min Standzeit. Grenzbänder bis 100 W bei Leistung, zugleich 0,1 K und 25 min⁻¹ bei Kühlung oder bis 500 Nm·min bei Belastung decken keinen dieser 36 Ausfälle ab. Bei doppelter Breite erfassen sie zwei Ausfälle und 291 korrekt freigegebene Vorgänge. Die flexible Verschleißkurve mit innerer Auswahl erkennt über alle fünf Wiederholungen in dem Standzeitband einen zusätzlichen Ausfall und verliert über alle Standzeiten im Mittel 3,8 bisherige Treffer.")
p("<b>Sensitivität und Auswahl.</b> Die nachträglich identifizierte Glättung mit sechs Minuten erreicht 237,4. Die Differenz von 0,2 Einheiten stammt aus einer Aufteilung mit einem Zusatztreffer und fünf Zusatzfehlalarmen. Die innere Auswahl glättender Varianten erreicht 241,6. Bei den UND-Regeln wählt die innere CV in allen 25 Falten die Grundvariante ohne Zusatzregel.")
p("<b>Recall-Kosten.</b> Eine innere Zielschwelle für 90 % Recall erreicht außen 90,18 % bei 319,2 Kosten. Mit Ziel 100 % werden auf drei der fünf Wiederholungen alle Ausfälle erkannt. Die mittleren Kosten betragen 7.353,4.")
p("Methodische Quellen · S. Matzka (2020), <i>Explainable Artificial Intelligence for Predictive Maintenance Applications</i>, §§ III-B und IV, S. 71-73, DOI 10.1109/AI4I49448.2020.00023.<br/>A. Samet (2023), <i>Data Mining Driven Approaches for Predictive Maintenance, Inventive Design Modelling and Energy Storage Systems</i>, Habilitationsschrift, Université de Strasbourg, Kap. 4, HAL tel-04775592.", "small")

# Kennzahlen einordnen und mögliche betriebliche Maßnahmen beschreiben.
section("5  Schlussfolgerungen", "[KI-3, KI-4]")
p("Die Kombination aus Prozessmerkmalen und einer auf die Kostenfunktion abgestimmten zweiten Alarmstufe erreicht 85,4 % geringere geschätzte Kosten gegenüber durchgehender Freigabe. Im gemeinsamen Split erfasst die zweite Stufe sechs weitere Ausfälle gegen 24 zusätzliche Fehlalarme. Sie spart 12 Einheiten beim Verhältnis 6:1. Für dieselben Alarme liegt die Gewinnschwelle bei einem Verhältnis von 4:1. Bei veränderten Inspektions- oder Ausfallkosten wird die Alarmschwelle neu bestimmt.")

p("Überwachung und Steuerung", "h2")
table(("Signal", "Maßnahme", "Prüfgröße"), [
    ("Leistung und Kühlung", "Leistung aus Drehmoment und Drehzahl online berechnen und Temperaturabstand parallel beobachten. Bei Alarm Regelauslöser und Rohwerte protokollieren.", "Alarmrate je Mechanismus, Abstand zur Grenze, Sensorabweichung"),
    ("Überlast und Werkzeugstandzeit", "Belastung nach Produktvariante überwachen. Im hohen Standzeitbereich geplante Inspektionen oder frühere Werkzeugwechsel im Pilotbetrieb vergleichen.", "Werkzeugstandzeit, ungeplante Stillstände und Kosten pro 1.000 Vorgänge"),
    ("Betrieb am Grenzwert", "Reale Messauflösung und Streuung bestimmen. Für temperaturkritische Werte ein fachlich kalibriertes Toleranzband mit manueller Sichtung erproben.", "Wiederholbarkeit bei Referenzmessung, Kosten von Grenzfällen"),
], [118, 223, 148])

p("Pilot mit Vorwarnzeit", "h2")
p("Die Zeilen erlauben eine Klassifikation auf Vorgangsebene. Zeitstempel und Werkzeugzyklus-IDs sind in der Aufgabe nicht verfügbar. Zeitliche Ausfallmuster nach Samet setzen geordnete Ereignisse voraus. Die hier getesteten UND-Regeln verwenden daher gleichzeitig gemessene Werte innerhalb einer Zeile. Ein Pilot erfasst Zeitpunkte von Sensorwert, Alarm, Inspektion und Ausfall. Im Schattenbetrieb wird geprüft, ob vor dem Ausfall tatsächlich Zeit für einen Eingriff bleibt.")
p("Bei einer kontrollierten Einführung gehören die Interventionskosten einschließlich echter Treffer in die Rechnung. Sobald c<sub>TP</sub> als Kosten eines berechtigten Alarms berücksichtigt wird, ergibt sich aus den erwarteten Kosten die Schwelle c<sub>FP</sub> / (c<sub>FP</sub> + c<sub>FN</sub> − c<sub>TP</sub>), sofern c<sub>FN</sub> &gt; c<sub>TP</sub>. Die Freigabeentscheidung im Pilot sollte auf den beobachteten Kosten und der verfügbaren Vorwarnzeit beruhen.")
p("<b>Reproduzierbarkeit.</b> modell.py trainiert auf modeling.csv und erzeugt die Challenge-Datei in Originalreihenfolge. validierung.json enthält die fünf Wiederholungen und die Simulation. Die Ranglisten, Faltenprotokolle und Befehle für die Zusatzversuche stehen unter experimente/ und in der README.", "small")
p("<b>Grundlage.</b> Aufgabenstellung Gruppenarbeit Classification, TH Mannheim, WiSe 2026/27, S. 1-5. Eigene Berechnungen aus modeling.csv. Methodische Quellen auf S. 5.", "small")


def footer(canvas, doc):
    # Auf jeder Seite Berichtstitel und Seitenzahl setzen.
    canvas.setStrokeColor(rule)
    canvas.line(48, 42, 547, 42)
    canvas.setFont("Segoe", 8.5)
    canvas.setFillColor(neutral)
    canvas.drawString(48, 28, "Predictive Maintenance  /  Technischer Bericht")
    canvas.drawRightString(547, 28, str(doc.page))


# Seitenformat festlegen und das Abgabe-PDF aus den gesammelten Elementen schreiben.
doc = SimpleDocTemplate(
    str(folder / "Technischer_Bericht.pdf"), pagesize=A4, leftMargin=48, rightMargin=48,
    topMargin=44, bottomMargin=53, title="Predictive Maintenance – Technischer Bericht",
    author="Gruppenarbeit Classification · TH Mannheim",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
