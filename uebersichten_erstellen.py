"""Erzeugt Management Summary, Hilfsmittelverzeichnis und KI-Dialoganhang."""

import json
from html import escape
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


FOLDER = Path(__file__).parent / "abgabe"
INK = colors.HexColor("#142339")
BLUE = colors.HexColor("#1A5470")
MIST = colors.HexColor("#EDF4F7")
RULE = colors.HexColor("#D2DEE5")


def register_fonts():
    """Verwendet die auf Windows vorhandene Schrift des technischen Berichts."""
    for name, file in (("Segoe", "segoeui.ttf"), ("Segoe Bold", "segoeuib.ttf")):
        pdfmetrics.registerFont(TTFont(name, f"C:/Windows/Fonts/{file}"))
    pdfmetrics.registerFontFamily("Segoe", normal="Segoe", bold="Segoe Bold")


STYLES = {
    "label": ParagraphStyle("label", fontName="Segoe Bold", fontSize=9, leading=13, textColor=BLUE, spaceAfter=10),
    "title": ParagraphStyle("title", fontName="Segoe Bold", fontSize=30, leading=35, textColor=INK, spaceAfter=15),
    "heading": ParagraphStyle("heading", fontName="Segoe Bold", fontSize=13, leading=17, textColor=BLUE,
                              spaceBefore=10, spaceAfter=5, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Segoe", fontSize=11, leading=16, textColor=INK, spaceAfter=8),
    "note": ParagraphStyle("note", fontName="Segoe", fontSize=9.5, leading=14, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Segoe", fontSize=10.3, leading=14.5, leftIndent=11,
                             bulletFontName="Segoe", bulletIndent=0, textColor=INK, spaceAfter=7),
    "kpi": ParagraphStyle("kpi", fontName="Segoe", fontSize=9.5, leading=16, textColor=INK),
    "table_head": ParagraphStyle("table_head", fontName="Segoe Bold", fontSize=9.5, leading=13,
                                 textColor=colors.white),
    "table_cell": ParagraphStyle("table_cell", fontName="Segoe", fontSize=9.5, leading=13, textColor=INK),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def german(number):
    """Formatiert Ergebniswerte für ein deutschsprachiges Publikum."""
    return f"{number:,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")


def build_pdf(filename, title, flowables):
    def footer(canvas, document):
        canvas.setStrokeColor(RULE)
        canvas.line(48, 42, 547, 42)
        canvas.setFillColor(BLUE)
        canvas.setFont("Segoe", 8.5)
        canvas.drawString(48, 28, f"Predictive Maintenance  /  {title}")
        canvas.drawRightString(547, 28, str(document.page))

    document = SimpleDocTemplate(
        str(FOLDER / filename), pagesize=A4, leftMargin=48, rightMargin=48,
        topMargin=48, bottomMargin=54, title=f"Predictive Maintenance – {title}",
    )
    document.build(flowables, onFirstPage=footer, onLaterPages=footer)


def management_summary():
    """Bezieht die Zahlen aus derselben Validierung wie der technische Bericht."""
    result = json.loads((FOLDER / "validierung.json").read_text(encoding="utf-8"))
    cv = result["validation"]
    average = mean(run["cost"] for run in cv)
    saving = 100 * (1 - average / (6 * result["modeling_failures"]))
    recall = 100 * mean(run["tp"] / (run["tp"] + run["fn"]) for run in cv)
    estimate = result["challenge_forecast"]["expected_cost"]
    low, high = result["challenge_forecast"]["predictive_95"]

    cards = Table([[
        p(f"<font size='21'><b>{german(average)}</b></font><br/>Kosten im CV-Mittel", "kpi"),
        p(f"<font size='21'><b>−{german(saving)} %</b></font><br/>gegenüber Freigabe aller Fälle", "kpi"),
        p(f"<font size='21'><b>{german(recall)} %</b></font><br/>erkannte Ausfälle im CV-Mittel", "kpi"),
    ]], colWidths=[162, 162, 162], hAlign="LEFT")
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), MIST),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 15),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))

    rows = f"{result['challenge_rows']:,}".replace(",", ".")
    alarms = result["challenge_alarms"]
    flowables = [
        p("TEAM-BRIEFING  /  GRUPPENARBEIT CLASSIFICATION  /  WISE 2026/27", "label"),
        p("Management Summary", "title"),
        p("<b>Empfehlung für die Abgabe.</b> Wir reichen das Hybridmodell ein. Es prüft zuerst "
          "Leistung, Kühlung und Werkzeugbelastung gegen gelernte Grenzen. Für die übrigen Vorgänge "
          "schätzt eine logistische Regression das Verschleißrisiko."),
        Spacer(1, 7), cards, Spacer(1, 9),
        p("Warum dieses Modell vorne liegt", "heading"),
        p(f"Die Challenge wertet einen übersehenen Ausfall sechsmal so teuer wie einen Fehlalarm. "
          f"Das Hybridmodell kostet in fünf wiederholten Datenaufteilungen "
          f"{min(run['cost'] for run in cv)}–{max(run['cost'] for run in cv)} Einheiten. "
          "Eine neu abgestimmte Schwelle, flexiblere Verschleißkurven und Zusatzalarme "
          "aus ähnlichen Betriebspunkten oder der Literatur brachten keinen stabilen Kostenvorteil."),
        p("Was wir für die Challenge erwarten", "heading"),
        p(f"Das Modell markiert <b>{alarms} von {rows} Vorgängen</b> für eine Inspektion. "
          f"Die geschätzten Kosten betragen <b>{german(estimate)} Einheiten</b>. "
          f"Unter vergleichbaren, annähernd unabhängigen Vorgängen liegt der geschätzte "
          f"95-%-Prognosebereich bei {low}–{high} Einheiten."),
        p("Was wir im Pitch zeigen", "heading"),
        p("Der Auslöser des Alarms lässt sich erklären. Bei einem Regelalarm zeigen wir die ausgelöste Regel "
          "und den Abstand zur gelernten Grenze. Der zusätzliche Verschleißalarm erkannte im "
          "gemeinsamen Test sechs weitere Ausfälle bei 24 zusätzlichen Fehlalarmen. Das spart bei 6:1 "
          "zwölf Einheiten. Ein Betriebspilot müsste zusätzlich messen, ob genug Zeit für den Eingriff bleibt."),
        p("Vor der Abgabe", "heading"),
        p("Gruppenname und Mitglieder ergänzen, den Namen der Prognosedatei anpassen und den "
          "vierminütigen Pitch gemeinsam proben.", "note"),
        p("KI-unterstützter Entwurf mit OpenAI Codex [KI-4]. Nachweis im Dialoganhang.", "note"),
    ]
    build_pdf("Management_Summary.pdf", "Management Summary", flowables)


def markdown_table(lines):
    """Setzt die Arbeitsschritte im Tabellenformat des Leitfadens."""
    rows = [[value.strip() for value in line.strip("|").split("|")] for line in lines]
    grid = Table([
        [p(escape(cell), "table_head" if index == 0 else "table_cell") for cell in row]
        for index, row in enumerate([rows[0], *rows[2:]])
    ], colWidths=(120, 91, 275), hAlign="LEFT", repeatRows=1)
    grid.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), (MIST, colors.white)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), .5, RULE),
    ]))
    return grid


def documentation(filename, title):
    """Setzt Markdown mit Überschriften, Aufzählungen und einfachen Tabellen als PDF."""
    source = (FOLDER / f"{filename}.md").read_text(encoding="utf-8")
    flowables = [p("GRUPPENARBEIT CLASSIFICATION  /  DOKUMENTATION", "label")]
    for block in source.strip().split("\n\n"):
        lines = block.splitlines()
        if block.startswith("|"):
            flowables.append(markdown_table(lines))
            continue
        for line in lines:
            if line.startswith("# "):
                flowables.append(p(escape(line[2:]), "title"))
            elif line.startswith("## "):
                flowables.append(p(escape(line[3:]), "heading"))
            elif line.startswith("- "):
                flowables.append(Paragraph(escape(line[2:]), STYLES["bullet"], bulletText="•"))
            elif line.strip():
                flowables.append(p(escape(line), "note"))
    build_pdf(f"{filename}.pdf", title, flowables)


if __name__ == "__main__":
    register_fonts()
    management_summary()
    documentation("Hilfsmittelverzeichnis", "Hilfsmittelverzeichnis")
    documentation("KI_Dialoge", "KI-Dialoge")
