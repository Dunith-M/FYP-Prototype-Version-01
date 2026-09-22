from __future__ import annotations

import csv
import html
from collections import Counter
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTER_PATH = PROJECT_ROOT / "docs" / "data_mapping_register.csv"
OUTPUT_PATH = PROJECT_ROOT / "output" / "pdf" / "data_mapping_register_guide.pdf"

EXPECTED_COLUMNS = [
    "source_dataset", "source_column", "canonical_field", "canonical_type",
    "key_role", "transform", "required", "null_policy", "example", "status",
    "owner", "notes",
]
DATASET_NAMES = {
    "sales": "Sales / Transactions",
    "inventory": "Inventory snapshots",
    "products": "Product master",
    "calendar": "Calendar",
    "branches": "Branch profile",
}
DATASET_INTROS = {
    "sales": (
        "This is the historical sales fact dataset. Its mappings preserve when a sale happened, "
        "which branch recorded it, the original product identity, quantities, values, and source context."
    ),
    "inventory": (
        "This dataset records observed stock snapshots. Its mappings make stock evidence joinable by date, "
        "branch, and product while preserving the important difference between missing and zero stock."
    ),
    "products": (
        "This is the product master. Its mappings provide the controlled product and family attributes used to "
        "describe products consistently after a join has been validated."
    ),
    "calendar": (
        "This dataset supplies the time and event context around historical dates. Its mappings support calendar-based "
        "analysis while clearly flagging conventions that still need confirmation."
    ),
    "branches": (
        "This dataset holds branch identifiers plus manager-reported context. Its mappings retain useful qualitative "
        "evidence without presenting it as approved Singer operational policy or a planning constraint."
    ),
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=False)


def build_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "GuideTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25,
            leading=30, textColor=colors.HexColor("#12324A"), alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "GuideSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12,
            leading=16, textColor=colors.HexColor("#4B5563"), alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "kicker": ParagraphStyle(
            "Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9,
            leading=12, textColor=colors.HexColor("#5B3F8C"), alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16,
            leading=20, textColor=colors.HexColor("#2E74B5"), spaceBefore=18, spaceAfter=10, keepWithNext=0,
        ),
        "h2": ParagraphStyle(
            "H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13,
            leading=16, textColor=colors.HexColor("#2E74B5"), spaceBefore=14, spaceAfter=7, keepWithNext=0,
        ),
        "body": ParagraphStyle(
            "Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5,
            leading=13.2, textColor=colors.HexColor("#1F2937"), spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=8,
            leading=9.6, textColor=colors.HexColor("#1F2937"), spaceAfter=0,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.5,
            leading=10.2, textColor=colors.white, spaceAfter=0,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=11,
            leading=14, textColor=colors.HexColor("#12324A"), spaceAfter=0,
        ),
        "foot": ParagraphStyle(
            "Foot", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.5,
            leading=9, textColor=colors.HexColor("#6B7280"), spaceAfter=0,
        ),
    }


def p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 0.45 * inch, letter[1] - doc.rightMargin, 0.45 * inch)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(doc.leftMargin, 0.27 * inch, "Singer Product Mix DSS MVP | Historical / Research Prototype")
    canvas.drawRightString(letter[1] - doc.rightMargin, 0.27 * inch, f"Page {doc.page}")
    canvas.restoreState()


def status_color(status: str):
    return {
        "CONFIRMED": colors.HexColor("#16794C"),
        "REVIEW": colors.HexColor("#8A5A00"),
        "TBC": colors.HexColor("#9B1C1C"),
    }[status]


def mapping_reason(row: dict[str, str]) -> str:
    dataset = row["source_dataset"]
    source = row["source_column"]
    role = row["key_role"]
    if row["canonical_field"] == "transaction_timestamp":
        return "Turn the sales date into the standard time field, so every historical sale can be placed on a business date."
    if row["canonical_field"] == "source_product_code":
        return "Keep the product code exactly as recorded in historical sales, before any future validated product crosswalk."
    if row["canonical_field"] == "raw_total_units":
        return "Preserve the observed signed unit quantity as the raw demand signal instead of silently changing returns, corrections, or zero events."
    if row["canonical_field"] == "raw_total_value":
        return "Retain the recorded total value for diagnostics and reconciliation without relabelling it as profit or margin."
    if row["canonical_field"] == "inventory_qty":
        return "Store the observed stock quantity as evidence at a particular branch, product, and snapshot date."
    if row["canonical_field"] == "list_price":
        return "Record product-master price as selling/list-price evidence for later descriptive analysis only."
    if row["canonical_field"] == "calendar_date":
        return "Create the calendar primary key so sales and inventory dates can be joined to one consistent daily calendar."
    if row["canonical_field"] == "branch_code":
        return "Standardize the branch identifier so records belonging to the same showroom can be joined safely."
    if row["canonical_field"] == "product_code":
        return "Standardize the product-master or inventory identity for an auditable product-level join."
    if role in {"date_key", "primary_date_key"}:
        return "Create a standard date key, allowing time-based records to be aligned and checked consistently."
    if role == "reconciliation":
        return "Keep a payment component separate so it can be compared with the reported total and potential data issues can be found."
    if role == "diagnostic":
        return "Retain the source measure for later checks and explanation, without giving it an unsupported business meaning."
    if role == "feature_fallback":
        return "Keep the historical description as a fallback feature for audit and analysis, not as a forced master-data classification."
    if role in {"audit_context", "audit_attribute"}:
        return "Keep source context unchanged so analysis results can be traced back and business assumptions can be checked."
    if role == "context_attribute":
        return "Preserve the source context that may help explain variation across sales, places, or time."
    if role == "attribute":
        return "Attach a controlled descriptive attribute to the standard record so later outputs are understandable to people."
    if role == "date_attribute":
        return "Retain the source calendar attribute so date patterns can be explored and validated against the actual date."
    if role in {"event_indicator", "event_attribute"}:
        return "Retain the source event label or indicator so potential calendar effects can be examined after validation."
    if role == "stock_evidence":
        return "Keep observed stock evidence separate from assumed stock, making data gaps visible rather than hidden."
    if role == "branch_attribute":
        return f"Record the manager-provided {source.lower().replace('_', ' ')} as branch context, separate from measured historical sales."
    if role == "provenance":
        return "Preserve where or when the information was supplied so users can judge whether it is suitable for a given use."
    if role == "governance":
        return "Keep the approval signal so unapproved profile information is not mistaken for an operational rule."
    if role == "quality_flag":
        return "Keep the source completeness signal so analysts can judge the reliability of the surrounding branch profile."
    return f"Translate {source} into a consistent field name while preserving the source evidence for audit."


def mapping_benefit(row: dict[str, str]) -> str:
    field = row["canonical_field"]
    if field == "transaction_timestamp":
        benefit = "Supports daily, weekly, and monthly demand analysis plus a calendar join."
    elif field == "source_product_code":
        benefit = "Prevents accidental claims that sales codes always equal product-master codes."
    elif field == "raw_total_units":
        benefit = "Provides the traceable historical input for future demand modelling after quality checks."
    elif field == "inventory_qty":
        benefit = "Supports stock-versus-sales analysis while keeping missing rows as UNKNOWN, not zero."
    elif field == "list_price":
        benefit = "Allows descriptive price-band analysis; it must never be used as cost, margin, or profit."
    elif field == "calendar_date":
        benefit = "Makes a one-row-per-date calendar join possible and testable."
    elif field == "branch_code":
        benefit = "Links sales, inventory, and branch profile records to the same showroom."
    elif field == "product_code":
        benefit = "Links product descriptions and families to inventory or validated historical sales products."
    elif row["source_dataset"] == "branches":
        benefit = "Qualitative evidence only; it is not a target or constraint."
    elif row["source_dataset"] == "calendar":
        benefit = "Adds calendar context to later analysis after validation."
    elif row["source_dataset"] == "sales":
        benefit = "Keeps the historical sales record traceable to its source."
    elif row["source_dataset"] == "inventory":
        benefit = "Keeps the record ready for an auditable branch-product-date join."
    else:
        benefit = "Provides a consistent application field for later validated analysis."
    return f"{benefit} <br/><font color='#6B7280'><b>Source caution:</b> {esc(row['notes'])}</font>"


def table_style(header_fill="#12324A", font_size=8.5):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_fill)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])


def register_table(rows, styles):
    body = styles["small"]
    header = styles["table_head"]
    data = [[
        p("Source column", header), p("Application field", header), p("Mapping action", header),
        p("Why you map it and how it helps", header), p("Status", header),
    ]]
    for row in rows:
        status = row["status"]
        status_html = f"<font color='{status_color(status).hexval()}'><b>{status}</b></font>"
        action = (
            f"<b>Type:</b> {esc(row['canonical_type'])}<br/>"
            f"<b>Transform:</b> {esc(row['transform'])}<br/>"
            f"<b>Required:</b> {esc(row['required'])}; <b>Null:</b> {esc(row['null_policy'])}"
        )
        data.append([
            p(f"<b>{esc(row['source_column'])}</b><br/><font color='#6B7280'>Example: {esc(row['example'])}</font>", body),
            p(f"<b>{esc(row['canonical_field'])}</b><br/><font color='#6B7280'>{esc(row['key_role'])}</font>", body),
            p(action, body),
            p(f"<b>Reason:</b> {esc(mapping_reason(row))}<br/>{mapping_benefit(row)}", body),
            p(status_html, body),
        ])
    widths = [1.25 * inch, 1.45 * inch, 1.65 * inch, 4.35 * inch, 1.10 * inch]
    table = Table(data, colWidths=widths, hAlign="LEFT")
    table.setStyle(table_style())
    return table


def simple_table(headers, rows, widths, styles):
    data = [[p(esc(h), styles["table_head"]) for h in headers]]
    for row in rows:
        data.append([p(esc(value), styles["small"]) for value in row])
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(table_style())
    return table


def load_register():
    with REGISTER_PATH.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or list(rows[0]) != EXPECTED_COLUMNS:
        raise ValueError("Mapping register does not have the expected columns.")
    if len(rows) != 69:
        raise ValueError(f"Expected 69 mappings; found {len(rows)}.")
    pairs = [(row["source_dataset"], row["source_column"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Mapping register contains duplicate dataset/column rows.")
    return rows


def build_document(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH), pagesize=landscape(letter), leftMargin=0.6 * inch,
        rightMargin=0.6 * inch, topMargin=0.55 * inch, bottomMargin=0.65 * inch,
        title="Data Mapping Register Guide", author="Singer Product Mix DSS MVP",
    )
    story = []
    counts = Counter(row["source_dataset"] for row in rows)
    status_counts = Counter(row["status"] for row in rows)

    story.extend([
        Spacer(1, 0.42 * inch),
        p("BEGINNER-FRIENDLY DATA GUIDE", styles["kicker"]),
        p("Why Your 69 Data Mappings Matter", styles["title"]),
        p("Singer Product Mix DSS MVP - Galle District Case Study", styles["subtitle"]),
    ])
    callout = Table([[p(
        "<b>What you completed:</b> You created a translation guide between the five raw Excel datasets and the "
        "standard fields that the Django application will use. This is foundation work: it makes later analytics "
        "and forecasting reliable, explainable, and safe.", styles["callout"])]], colWidths=[9.8 * inch])
    callout.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8EEF5")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#8BA9C7")),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.extend([callout, Spacer(1, 0.14 * inch)])
    story.append(simple_table(
        ["Dataset", "Mappings completed", "What it contributes"],
        [
            ["Sales / Transactions", str(counts["sales"]), "Historical demand, branch, product identity, sales quantities and values."],
            ["Inventory snapshots", str(counts["inventory"]), "Observed stock by snapshot date, branch and product."],
            ["Product master", str(counts["products"]), "Controlled product descriptions, families and selling/list price evidence."],
            ["Calendar", str(counts["calendar"]), "Date, weekday, weekend and event context."],
            ["Branch profile", str(counts["branches"]), "Branch identity plus manager-reported qualitative context."],
        ], [2.3 * inch, 1.2 * inch, 6.3 * inch], styles,
    ))
    story.append(Spacer(1, 0.10 * inch))
    story.append(p("Historical / Research Prototype", styles["h2"]))
    story.append(p(
        "This guide documents historical data for research. It does not state current Singer policy or operational advice. "
        "Synthetic values, unapproved branch-profile inputs, costs, margins, targets and resource constraints are not created by this mapping work.",
        styles["body"],
    ))
    story.append(PageBreak())

    story.append(p("1. The simple reason you do data mapping", styles["h1"]))
    story.append(p(
        "Raw files are created for day-to-day reporting, not automatically for one data-science system. Each file may use a different name, format, or level of detail. "
        "Data mapping creates one shared language before you load, join, analyse, or model the data.", styles["body"]
    ))
    story.append(p("A mapping answers five practical questions", styles["h2"]))
    story.append(simple_table(
        ["Question", "Why it matters to you as a data scientist"],
        [
            ["What is this source column?", "Stops you from guessing what abbreviations such as BM_AD or PERIOD mean."],
            ["What standard field will the app use?", "Lets every future script, database table and dashboard use one stable name."],
            ["How should it be cleaned?", "Makes routine steps such as trimming codes and parsing dates consistent and reproducible."],
            ["Can it join another dataset?", "Identifies keys such as branch_code, product_code and date before combining data."],
            ["Is the meaning fully trusted?", "Separates CONFIRMED fields from REVIEW and TBC fields so uncertainty is visible."],
        ], [2.7 * inch, 7.1 * inch], styles,
    ))
    story.append(p("How it fits into your MVP", styles["h2"]))
    flow = Table([[p(
        "Raw Excel files  ->  Mapping register  ->  Consistent staging tables  ->  Data-quality checks  ->  Historical analytics and forecasting  ->  Demonstration-only planning outputs",
        styles["callout"])]], colWidths=[9.8 * inch])
    flow.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6F9")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([flow, Spacer(1, 0.12 * inch)])
    story.append(p("The practical usefulness", styles["h2"]))
    for line in [
        "Reliable joins: sales, inventory, product, calendar and branch records can be connected using documented keys.",
        "Reproducible work: you can explain exactly how a raw value became an application field.",
        "Safer modelling: you do not build forecasts from a column whose meaning or null behaviour was guessed.",
        "Easier debugging: if a dashboard number looks wrong, you can trace it back to the raw column and transform.",
        "Stronger FYP evidence: the register shows supervisors how you protected data meaning before using AI or ML.",
    ]:
        story.append(p(f"<b>-</b> {esc(line)}", styles["body"]))
    story.append(p("What the mapping register does not do", styles["h2"]))
    story.append(p(
        "It does not clean away negative or zero sales; turn a missing inventory row into zero stock; fuzzy-match product descriptions; "
        "assume SALES_PART equals PRODUCT_CODE; or invent cost, profit, margin, targets, stock constraints, or current Singer policy.",
        styles["body"],
    ))
    story.append(PageBreak())

    story.append(p("2. How to read a mapping row", styles["h1"]))
    story.append(p(
        "Every mapping row is a small contract. It says what will happen to one source column and why that choice is safe.",
        styles["body"],
    ))
    story.append(simple_table(
        ["Register field", "Plain-English meaning", "Example from your register"],
        [
            ["source_dataset", "Which logical dataset the column comes from.", "sales"],
            ["source_column", "The original Excel header. It is never renamed inside the raw workbook.", "SHOP_CODE"],
            ["canonical_field", "The stable field name your app will use later.", "branch_code"],
            ["canonical_type", "The expected data type used for validation.", "datetime"],
            ["key_role", "Whether the field is a join key, date key, attribute, audit field or evidence.", "branch_key"],
            ["transform", "The documented handling rule, such as trim+uppercase or parse date.", "trim+uppercase"],
            ["required / null_policy", "Whether the field must be present and how blanks are treated.", "yes / not_null"],
            ["example", "One representative real value used to make the mapping concrete.", "SBO01"],
            ["status", "Confidence level: CONFIRMED, REVIEW, or TBC.", "CONFIRMED"],
            ["owner", "The role responsible for business meaning or source stewardship.", "Sales data owner"],
            ["notes", "Important guardrails and unresolved meaning.", "Do not assume source and master product codes are identical."],
        ], [1.9 * inch, 4.55 * inch, 3.35 * inch], styles,
    ))
    story.append(p("Why the status matters", styles["h2"]))
    story.append(simple_table(
        ["Status", "Meaning", "What you should do"],
        [
            ["CONFIRMED", "The mapping is directly supported by the source schema or project definition.", "Use it as documented, while still running normal data-quality checks."],
            ["REVIEW", "The mapping is reasonable, but its business interpretation or use should be checked.", "Keep it, but do not use it for a strong decision or model feature until reviewed."],
            ["TBC", "The meaning cannot safely be determined from the available material.", "Retain the raw value for audit and ask the data owner before relying on it."],
        ], [1.4 * inch, 4.0 * inch, 4.4 * inch], styles,
    ))
    story.append(Spacer(1, 0.1 * inch))
    story.append(p(
        f"Your register contains {status_counts['CONFIRMED']} CONFIRMED mappings, {status_counts['REVIEW']} REVIEW mappings and {status_counts['TBC']} TBC mappings. "
        "This is a strength: it makes uncertainty visible rather than hiding it.", styles["body"]
    ))
    story.append(PageBreak())

    for index, dataset in enumerate(DATASET_NAMES):
        dataset_rows = [row for row in rows if row["source_dataset"] == dataset]
        chunk_size = 2 if dataset in {"calendar", "branches"} else 6
        chunks = [dataset_rows[start:start + chunk_size] for start in range(0, len(dataset_rows), chunk_size)]
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index:
                story.append(PageBreak())
                chunk_flow = [p(f"3.{index + 1}. {DATASET_NAMES[dataset]} (continued)", styles["h2"])]
                chunk_flow.append(p(
                    "Each row remains a documented source-to-application contract with a reason, benefit and caution.",
                    styles["body"],
                ))
            else:
                chunk_flow = [p(f"3.{index + 1}. {DATASET_NAMES[dataset]}: mapping-by-mapping guide", styles["h1"])]
                chunk_flow.append(p(DATASET_INTROS[dataset], styles["body"]))
                chunk_flow.append(p(
                    "Read the fourth column slowly: it explains both the reason for the mapping and the practical benefit you gain later.",
                    styles["body"],
                ))
            chunk_flow.append(register_table(chunk, styles))
            story.append(KeepTogether(chunk_flow))
        story.append(PageBreak())

    story.append(p("4. What needs attention before later data-science work", styles["h1"]))
    story.append(p(
        "REVIEW and TBC do not mean the data is useless. They mean you should use it carefully and ask a focused business question before making it drive a model or recommendation.",
        styles["body"],
    ))
    attention = [
        ["Sales", "PERIOD, AREA, SITE_DES", "Confirm the reporting-period definition, organisational hierarchy and the relationship between site description and shop code."],
        ["Sales", "BM_AD (TBC)", "Ask the sales data owner what the abbreviation means before using it as a feature or filter."],
        ["Calendar", "DAY_OF_WEEK_NUM, WEEK_OF_YEAR", "Confirm the numbering convention, for example ISO versus a different calendar convention."],
        ["Calendar", "Holiday and event fields", "Confirm holiday coverage, naming and event taxonomy before attributing changes in demand to events."],
        ["Calendar", "KNOWN_AT_DATE (TBC)", "Confirm whether it represents source availability, publication, knowledge date or another concept."],
        ["Branches", "Manager-reported profile attributes", "Treat them as qualitative evidence. Obtain review/approval before using them as quantitative features, constraints or targets."],
    ]
    story.append(simple_table(
        ["Dataset", "Fields", "Why attention is needed"], attention,
        [1.25 * inch, 2.45 * inch, 6.1 * inch], styles,
    ))
    story.append(p("Your next benefit", styles["h2"]))
    story.append(p(
        "When you start data loading, this document can guide your ingestion code and database schema. When you start analysis, it can guide your joins and quality checks. "
        "When you explain the FYP, it shows that your AI work is built on controlled, auditable data rather than assumptions.",
        styles["body"],
    ))
    story.append(p("Document basis", styles["h2"]))
    story.append(p(
        "Prepared from docs/data_mapping_register.csv (69 mappings) and the project historical/research assumptions. "
        f"Generated on {date.today().isoformat()}. Raw Excel files were not modified.", styles["foot"]
    ))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    build_document(load_register())
    print(OUTPUT_PATH)
