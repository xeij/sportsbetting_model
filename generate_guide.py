"""Generate a beginner's PDF guide for testing and benchmarking the sports betting ML model."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import Flowable
import os

# ── Colour palette ──────────────────────────────────────────────────────────
DARK_BG     = HexColor("#1a1a2e")
ACCENT      = HexColor("#4ade80")   # green
ACCENT2     = HexColor("#60a5fa")   # blue
LIGHT_GREY  = HexColor("#f0f4f8")
MID_GREY    = HexColor("#cbd5e1")
CODE_BG     = HexColor("#0f172a")
CODE_FG     = HexColor("#a3e635")
WARN_BG     = HexColor("#fef3c7")
WARN_BORDER = HexColor("#f59e0b")
INFO_BG     = HexColor("#e0f2fe")
INFO_BORDER = HexColor("#38bdf8")
TIP_BG      = HexColor("#f0fdf4")
TIP_BORDER  = HexColor("#4ade80")
SECTION_HDR = HexColor("#1e3a5f")

W, H = A4


# ── Custom flowables ─────────────────────────────────────────────────────────
class ColoredLine(Flowable):
    def __init__(self, color, width=None, thickness=2):
        Flowable.__init__(self)
        self.color = color
        self._width = width
        self.thickness = thickness
        self.width = width or (W - 4*cm)
        self.height = thickness + 4

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.thickness/2, self.width, self.thickness/2)


class CalloutBox(Flowable):
    """A coloured callout box (warning / info / tip)."""
    def __init__(self, paragraphs, bg_color, border_color, label="", width=None):
        Flowable.__init__(self)
        self.paragraphs = paragraphs if isinstance(paragraphs, list) else [paragraphs]
        self.bg_color = bg_color
        self.border_color = border_color
        self.label = label
        self._box_width = width or (W - 4*cm)
        self.height = 0   # will be set in wrap

    def wrap(self, availWidth, availHeight):
        pad = 10
        total_h = pad
        for p in self.paragraphs:
            w, h = p.wrapOn(self.canv, self._box_width - 2*pad - 4, availHeight)
            total_h += h + 4
        total_h += pad
        self.height = total_h
        self.width = self._box_width
        return self._box_width, total_h

    def draw(self):
        pad = 10
        c = self.canv
        # Background
        c.setFillColor(self.bg_color)
        c.setStrokeColor(self.border_color)
        c.setLineWidth(2)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=1)
        # Left accent bar
        c.setFillColor(self.border_color)
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        # Draw paragraphs from top down
        y = self.height - pad
        for p in self.paragraphs:
            w, h = p.wrapOn(c, self.width - 2*pad - 4, self.height)
            y -= h
            p.drawOn(c, pad + 6, y)
            y -= 4


# ── Styles ───────────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    styles = {
        "cover_title": s("cover_title",
            fontName="Helvetica-Bold", fontSize=32, textColor=white,
            alignment=TA_CENTER, spaceAfter=8, leading=38),
        "cover_sub": s("cover_sub",
            fontName="Helvetica", fontSize=16, textColor=ACCENT,
            alignment=TA_CENTER, spaceAfter=6, leading=20),
        "cover_tag": s("cover_tag",
            fontName="Helvetica-Oblique", fontSize=11, textColor=MID_GREY,
            alignment=TA_CENTER, spaceAfter=4, leading=16),
        "chapter": s("chapter",
            fontName="Helvetica-Bold", fontSize=18, textColor=white,
            backColor=SECTION_HDR, spaceBefore=18, spaceAfter=10,
            leftIndent=-6, rightIndent=-6, borderPad=8,
            leading=24),
        "section": s("section",
            fontName="Helvetica-Bold", fontSize=13, textColor=SECTION_HDR,
            spaceBefore=14, spaceAfter=6, leading=17),
        "subsection": s("subsection",
            fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT2,
            spaceBefore=10, spaceAfter=4, leading=14),
        "body": s("body",
            fontName="Helvetica", fontSize=10, textColor=HexColor("#1e293b"),
            spaceBefore=3, spaceAfter=4, leading=15, alignment=TA_JUSTIFY),
        "body_bold": s("body_bold",
            fontName="Helvetica-Bold", fontSize=10, textColor=HexColor("#1e293b"),
            spaceBefore=3, spaceAfter=4, leading=15),
        "code": s("code",
            fontName="Courier", fontSize=9, textColor=CODE_FG,
            backColor=CODE_BG, spaceBefore=6, spaceAfter=6, leading=13,
            leftIndent=10, rightIndent=10, borderPad=6,
            borderWidth=0, borderColor=CODE_BG),
        "code_comment": s("code_comment",
            fontName="Courier-Oblique", fontSize=9, textColor=HexColor("#94a3b8"),
            backColor=CODE_BG, leading=13, leftIndent=10, rightIndent=10,
            spaceBefore=0, spaceAfter=0),
        "callout_body": s("callout_body",
            fontName="Helvetica", fontSize=9.5, textColor=HexColor("#1e293b"),
            leading=14, spaceBefore=0, spaceAfter=0),
        "callout_label": s("callout_label",
            fontName="Helvetica-Bold", fontSize=9.5, textColor=HexColor("#1e293b"),
            leading=14, spaceBefore=0, spaceAfter=0),
        "table_header": s("table_header",
            fontName="Helvetica-Bold", fontSize=9, textColor=white,
            alignment=TA_CENTER, leading=12),
        "table_cell": s("table_cell",
            fontName="Helvetica", fontSize=9, textColor=HexColor("#1e293b"),
            alignment=TA_CENTER, leading=12),
        "table_cell_l": s("table_cell_l",
            fontName="Helvetica", fontSize=9, textColor=HexColor("#1e293b"),
            alignment=TA_LEFT, leading=12),
        "toc_item": s("toc_item",
            fontName="Helvetica", fontSize=10.5, textColor=HexColor("#1e293b"),
            spaceBefore=4, spaceAfter=2, leading=14),
        "toc_sub": s("toc_sub",
            fontName="Helvetica-Oblique", fontSize=9.5, textColor=HexColor("#475569"),
            spaceBefore=1, spaceAfter=1, leading=13, leftIndent=16),
        "footer": s("footer",
            fontName="Helvetica-Oblique", fontSize=8, textColor=HexColor("#94a3b8"),
            alignment=TA_CENTER),
        "number": s("number",
            fontName="Helvetica-Bold", fontSize=28, textColor=ACCENT,
            alignment=TA_CENTER, leading=32),
    }
    return styles


# ── Page templates ────────────────────────────────────────────────────────────
def cover_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARK_BG)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # decorative top stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, H - 8, W, 8, fill=1, stroke=0)
    # decorative bottom stripe
    canvas.setFillColor(ACCENT2)
    canvas.rect(0, 0, W, 6, fill=1, stroke=0)
    canvas.restoreState()


def normal_page(canvas, doc):
    canvas.saveState()
    # subtle header bar
    canvas.setFillColor(SECTION_HDR)
    canvas.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Oblique", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(2*cm, H - 0.8*cm, "Football Betting Model  |  Testing & Benchmarking Guide")
    canvas.drawRightString(W - 2*cm, H - 0.8*cm, f"Page {doc.page}")
    # footer
    canvas.setFillColor(LIGHT_GREY)
    canvas.rect(0, 0, W, 1*cm, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#64748b"))
    canvas.setFont("Helvetica-Oblique", 7.5)
    canvas.drawCentredString(W/2, 0.35*cm, "For personal / educational use only. Gamble responsibly.")
    canvas.restoreState()


# ── Helper builders ──────────────────────────────────────────────────────────
def code_block(lines, styles):
    """Return a list of Paragraphs styled as a code block."""
    blocks = []
    for line in lines:
        if line.startswith("#"):
            blocks.append(Paragraph(line.replace(" ", "&nbsp;"), styles["code_comment"]))
        else:
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe = safe.replace(" ", "&nbsp;")
            blocks.append(Paragraph(safe, styles["code"]))
    return blocks


def warn(text, styles, label="⚠ Warning"):
    return CalloutBox(
        [Paragraph(f"<b>{label}:</b> {text}", styles["callout_body"])],
        WARN_BG, WARN_BORDER)


def info(text, styles, label="ℹ Info"):
    return CalloutBox(
        [Paragraph(f"<b>{label}:</b> {text}", styles["callout_body"])],
        INFO_BG, INFO_BORDER)


def tip(text, styles, label="✅ Tip"):
    return CalloutBox(
        [Paragraph(f"<b>{label}:</b> {text}", styles["callout_body"])],
        TIP_BG, TIP_BORDER)


def bullet_list(items, styles, indent=0):
    li = []
    for item in items:
        li.append(ListItem(Paragraph(item, styles["body"]), leftIndent=20+indent, bulletIndent=8+indent))
    return ListFlowable(li, bulletType='bullet', bulletFontName='Helvetica',
                        bulletFontSize=10, bulletColor=ACCENT)


def numbered_list(items, styles, indent=0):
    li = []
    for i, item in enumerate(items, 1):
        li.append(ListItem(Paragraph(item, styles["body"]), leftIndent=28+indent, bulletIndent=8+indent))
    return ListFlowable(li, bulletType='1', bulletFontName='Helvetica-Bold',
                        bulletFontSize=10, bulletColor=ACCENT2)


def metric_table(data, col_widths, styles):
    """data = [header_row, ...data_rows]"""
    t = Table(data, colWidths=col_widths)
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECTION_HDR),
        ('TEXTCOLOR',  (0,0), (-1,0), white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LIGHT_GREY]),
        ('GRID',       (0,0), (-1,-1), 0.5, MID_GREY),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('ROUNDEDCORNERS', [4]),
    ])
    t.setStyle(ts)
    return t


# ── Content builders ──────────────────────────────────────────────────────────
def build_cover(story, styles):
    story.append(Spacer(1, 3.5*cm))
    story.append(Paragraph("Football Betting Model", styles["cover_title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Testing &amp; Benchmarking Guide", styles["cover_sub"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(ColoredLine(ACCENT, W - 6*cm, 3))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("A Step-by-Step Guide for Beginners", styles["cover_tag"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Covers: Model Evaluation · Backtesting · Interpreting Results", styles["cover_tag"]))
    story.append(Spacer(1, 4*cm))
    story.append(Paragraph("XGBoost · LightGBM · Ensemble Prediction", styles["cover_tag"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("English Football  |  Premier League to National League", styles["cover_tag"]))
    story.append(Spacer(1, 2*cm))
    story.append(ColoredLine(ACCENT2, W - 6*cm, 1))
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("Version 1.0  ·  February 2026", styles["cover_tag"]))
    story.append(PageBreak())


def build_toc(story, styles):
    story.append(Paragraph("Table of Contents", styles["chapter"]))
    story.append(Spacer(1, 0.3*cm))

    chapters = [
        ("1", "Introduction", [
            "What this guide covers",
            "How the model works — a plain-English overview",
            "Key concepts you need to know",
        ]),
        ("2", "Setting Up Your Environment", [
            "Prerequisites & dependencies",
            "Running the project for the first time",
            "Understanding the project folder structure",
        ]),
        ("3", "Step 1 — Downloading Data", [
            "What data is downloaded",
            "How to run the download step",
            "Verifying the downloaded data",
        ]),
        ("4", "Step 2 — Training the Models", [
            "What happens during training",
            "How to run the training step",
            "Understanding training output",
        ]),
        ("5", "Step 3 — Evaluating Model Quality", [
            "Accuracy, Log Loss, and Brier Score explained",
            "Reading the calibration curve",
            "Reading the feature importance chart",
            "Good vs bad numbers — what to look for",
        ]),
        ("6", "Step 4 — Running the Backtest", [
            "What backtesting is and why it matters",
            "How to run the backtest",
            "Reading the backtest report",
            "Understanding the ROI curve",
        ]),
        ("7", "Interpreting Your Results", [
            "Complete metrics reference table",
            "What the numbers mean in plain English",
            "Red flags to watch out for",
        ]),
        ("8", "Tweaking the Model (config.yaml)", [
            "Key settings you can change",
            "Conservative vs aggressive configurations",
        ]),
        ("9", "Common Problems & Fixes", []),
        ("10", "Responsible Use & Limitations", []),
    ]

    for num, title, subs in chapters:
        story.append(Paragraph(f"<b>{num}.</b>  {title}", styles["toc_item"]))
        for sub in subs:
            story.append(Paragraph(f"• {sub}", styles["toc_sub"]))
    story.append(PageBreak())


def build_intro(story, styles):
    story.append(Paragraph("1. Introduction", styles["chapter"]))

    story.append(Paragraph("What This Guide Covers", styles["section"]))
    story.append(Paragraph(
        "This guide walks you through <b>testing and benchmarking</b> the Football Betting Model "
        "from start to finish. By the end you will be able to:",
        styles["body"]))
    story.append(bullet_list([
        "Download 16 seasons of English football match data",
        "Train three machine-learning models (Logistic Regression, XGBoost, LightGBM)",
        "Evaluate how <i>well</i> those models actually predict match outcomes",
        "Run a historical simulation (backtest) to see how a betting strategy would have performed",
        "Read and understand every number in the results",
        "Safely tweak settings to see how they affect performance",
    ], styles))
    story.append(Spacer(1, 0.2*cm))
    story.append(warn(
        "Past performance does not guarantee future results. "
        "This project is for educational and research purposes. "
        "Never bet more than you can afford to lose.",
        styles))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("How the Model Works — Plain English", styles["section"]))
    story.append(Paragraph(
        "The model learns patterns from thousands of historical football matches. "
        "For each upcoming game it estimates three probabilities:",
        styles["body"]))
    data = [
        [Paragraph("Outcome", styles["table_header"]),
         Paragraph("What it means", styles["table_header"]),
         Paragraph("Example", styles["table_header"])],
        [Paragraph("Home Win", styles["table_cell"]),
         Paragraph("The home team wins", styles["table_cell_l"]),
         Paragraph("Arsenal 2-0 Chelsea", styles["table_cell_l"])],
        [Paragraph("Draw", styles["table_cell"]),
         Paragraph("The match ends in a draw", styles["table_cell_l"]),
         Paragraph("Arsenal 1-1 Chelsea", styles["table_cell_l"])],
        [Paragraph("Away Win", styles["table_cell"]),
         Paragraph("The away team wins", styles["table_cell_l"]),
         Paragraph("Arsenal 0-1 Chelsea", styles["table_cell_l"])],
    ]
    story.append(metric_table(data, [3*cm, 8*cm, 5.5*cm], styles))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "It then compares its probability estimates against the bookmaker's implied probabilities "
        "(extracted from the betting odds). When the model believes the true probability is "
        "meaningfully higher than what the bookmaker implies, that is called a <b>value bet</b>.",
        styles["body"]))

    story.append(Paragraph("Key Concepts", styles["section"]))
    terms = [
        ("Probability", "A number between 0 and 1 (or 0–100%) representing how likely something is.  0.6 = 60% chance."),
        ("Implied Probability", "The probability the bookmaker's odds suggest.  Odds of 2.00 imply a 50% chance."),
        ("Value Bet", "A bet where the model thinks the true probability is higher than the bookmaker's implied probability by at least 10% (configurable)."),
        ("Overround / Vig", "The bookmaker's built-in margin.  Odds across all outcomes sum to >100%, the excess is profit for the bookmaker."),
        ("Kelly Criterion", "A mathematical formula that calculates how much of your bankroll to bet based on your edge and the odds.  The model uses a very conservative 1/20th Kelly."),
        ("Backtest", "Re-running your strategy on historical data to see how it would have performed — like a flight simulator for betting strategies."),
        ("Chronological Split", "The model is trained on older matches and tested on newer ones to simulate real-world usage and prevent data leakage."),
        ("Calibration", "How closely the model's predicted probabilities match reality.  A well-calibrated model that says '70%' should be right ~70% of the time."),
        ("ROI (Return on Investment)", "Profit expressed as a percentage of total amount staked.  ROI of 10% means for every £100 staked, you gained £10 net."),
    ]
    for term, definition in terms:
        story.append(Paragraph(f"<b>{term}:</b>  {definition}", styles["body"]))
        story.append(Spacer(1, 0.1*cm))
    story.append(PageBreak())


def build_setup(story, styles):
    story.append(Paragraph("2. Setting Up Your Environment", styles["chapter"]))

    story.append(Paragraph("Prerequisites", styles["section"]))
    story.append(Paragraph(
        "Before running anything, make sure the following are installed on your computer:",
        styles["body"]))
    story.append(bullet_list([
        "<b>Python 3.9 or higher</b>  — download from python.org",
        "<b>pip</b>  — comes bundled with Python",
        "<b>Git</b> (optional)  — for cloning the repository",
        "<b>4 GB free disk space</b>  — for data and model files",
    ], styles))

    story.append(Paragraph("Installing Dependencies", styles["section"]))
    story.append(Paragraph(
        "Open a terminal, navigate to the project folder, and run:",
        styles["body"]))
    story.extend(code_block([
        "# Navigate to the project folder",
        "cd /path/to/sportsbetting_model",
        "",
        "# (Recommended) Create and activate a virtual environment",
        "python -m venv venv",
        "source venv/bin/activate          # Mac / Linux",
        "venv\\Scripts\\activate             # Windows",
        "",
        "# Install all required libraries",
        "pip install -r requirements.txt",
    ], styles))
    story.append(info(
        "The requirements.txt file lists all necessary libraries "
        "(pandas, xgboost, lightgbm, scikit-learn, etc.).  "
        "The install may take 2–5 minutes on a fresh environment.",
        styles))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Project Folder Structure", styles["section"]))
    story.append(Paragraph(
        "Here is what each important file and folder does:",
        styles["body"]))
    data = [
        [Paragraph("Path", styles["table_header"]),
         Paragraph("What it is", styles["table_header"])],
        [Paragraph("main.py", styles["table_cell"]),
         Paragraph("Command-line entry point — the main script you will run", styles["table_cell_l"])],
        [Paragraph("config.yaml", styles["table_cell"]),
         Paragraph("All tuneable settings (thresholds, model parameters, odds ranges)", styles["table_cell_l"])],
        [Paragraph("gradio_app.py", styles["table_cell"]),
         Paragraph("Optional graphical web interface (click-based, no terminal needed)", styles["table_cell_l"])],
        [Paragraph("src/data_loader.py", styles["table_cell"]),
         Paragraph("Downloads and merges historical match CSV files", styles["table_cell_l"])],
        [Paragraph("src/feature_engineering.py", styles["table_cell"]),
         Paragraph("Calculates team form, head-to-head stats, rest days, etc.", styles["table_cell_l"])],
        [Paragraph("src/modeling.py", styles["table_cell"]),
         Paragraph("Trains and evaluates Logistic Regression, XGBoost, LightGBM", styles["table_cell_l"])],
        [Paragraph("src/backtest.py", styles["table_cell"]),
         Paragraph("Simulates the betting strategy on historical test data", styles["table_cell_l"])],
        [Paragraph("src/prediction.py", styles["table_cell"]),
         Paragraph("Makes predictions and identifies value bets", styles["table_cell_l"])],
        [Paragraph("data/raw/", styles["table_cell"]),
         Paragraph("Downloaded raw CSV files (created automatically)", styles["table_cell_l"])],
        [Paragraph("data/processed/", styles["table_cell"]),
         Paragraph("Merged & feature-engineered data files", styles["table_cell_l"])],
        [Paragraph("data/visualizations/", styles["table_cell"]),
         Paragraph("Generated charts, reports and bet history CSVs", styles["table_cell_l"])],
        [Paragraph("models/", styles["table_cell"]),
         Paragraph("Saved trained model files (.joblib)", styles["table_cell_l"])],
    ]
    story.append(metric_table(data, [4.5*cm, 12*cm], styles))
    story.append(PageBreak())


def build_download(story, styles):
    story.append(Paragraph("3. Step 1 — Downloading Data", styles["chapter"]))

    story.append(Paragraph("What Data Is Downloaded", styles["section"]))
    story.append(Paragraph(
        "The model downloads <b>16 seasons</b> (2010/11 – 2025/26) of English football match data "
        "from football-data.co.uk across four divisions:",
        styles["body"]))
    story.append(bullet_list([
        "<b>E0</b> — Premier League",
        "<b>E2</b> — League One",
        "<b>E3</b> — League Two",
        "<b>EC</b> — National League (Non-League / Conference)",
    ], styles))
    story.append(Paragraph(
        "Each CSV file contains: match date, home team, away team, full-time score, "
        "half-time score, and betting odds from multiple bookmakers (Bet365, Pinnacle, etc.).",
        styles["body"]))

    story.append(Paragraph("How to Run the Download", styles["section"]))
    story.append(Paragraph(
        "In your terminal (with the virtual environment active) run:",
        styles["body"]))
    story.extend(code_block([
        "python main.py --download",
    ], styles))
    story.append(Paragraph(
        "You will see progress messages like:",
        styles["body"]))
    story.extend(code_block([
        "[INFO] Downloading E0 2024/25 ...",
        "[INFO] Downloading E2 2024/25 ...",
        "[INFO] Merging 64 files ...",
        "[INFO] Data download complete. 48,320 matches saved to data/processed/all_matches.csv",
    ], styles))

    story.append(Paragraph("Verifying the Download", styles["section"]))
    story.append(Paragraph(
        "After download, quickly verify everything is in order:",
        styles["body"]))
    story.extend(code_block([
        "# Check the file exists and has rows",
        "python -c \"",
        "import pandas as pd",
        "df = pd.read_csv('data/processed/all_matches.csv')",
        "print('Total matches:', len(df))",
        "print('Date range:', df['Date'].min(), 'to', df['Date'].max())",
        "print('Divisions:', df['Div'].unique())",
        "\"",
    ], styles))
    story.append(Paragraph("<b>Expected output (approximately):</b>", styles["body_bold"]))
    story.extend(code_block([
        "Total matches: 45000",
        "Date range: 2010-08-07 to 2026-02-22",
        "Divisions: ['E0' 'E2' 'E3' 'EC']",
    ], styles))
    story.append(tip(
        "If fewer than 40,000 matches are shown, some files may have failed to download "
        "(network issue). Run --download again — it skips files that already exist.",
        styles))
    story.append(PageBreak())


def build_training(story, styles):
    story.append(Paragraph("4. Step 2 — Training the Models", styles["chapter"]))

    story.append(Paragraph("What Happens During Training", styles["section"]))
    story.append(Paragraph(
        "The training step performs the following operations in order:",
        styles["body"]))
    story.append(numbered_list([
        "<b>Load data</b>  — reads the merged CSV from Step 1",
        "<b>Feature engineering</b>  — calculates team form (last 5 & 10 games), head-to-head records, rest days, goal differences, and implied probabilities from the bookmaker odds",
        "<b>Chronological split</b>  — the most recent 15% of matches become the test set; the rest are for training.  This is critical to prevent 'data leakage'",
        "<b>Train Baseline</b>  — a simple Logistic Regression model (the benchmark)",
        "<b>Train XGBoost</b>  — a powerful gradient-boosting model with 300 trees",
        "<b>Train LightGBM</b>  — a faster alternative gradient-boosting model",
        "<b>Evaluate each model</b>  — prints Accuracy, Log Loss, and Brier Score",
        "<b>Save models</b>  — stored in the models/ folder as .joblib files",
        "<b>Generate charts</b>  — calibration curves and feature importance plots",
    ], styles))

    story.append(Paragraph("How to Run Training", styles["section"]))
    story.extend(code_block([
        "python main.py --train",
    ], styles))
    story.append(Paragraph(
        "Training takes approximately <b>3–10 minutes</b> depending on your hardware. "
        "You will see live progress messages in the terminal.",
        styles["body"]))

    story.append(Paragraph("Understanding Training Output", styles["section"]))
    story.append(Paragraph(
        "The terminal will print metrics for each model.  Here is a sample:",
        styles["body"]))
    story.extend(code_block([
        "[INFO] Train set: 41072 matches (2010-08-07 to 2024-07-31)",
        "[INFO] Test set:  7248 matches (2024-08-01 to 2026-02-22)",
        "[INFO] Prepared 6891 samples with 42 features",
        "",
        "[INFO] Baseline (Logistic Regression)",
        "[INFO]   Accuracy: 0.4712   Log Loss: 1.0231   Brier: 0.2187",
        "",
        "[INFO] XGBoost",
        "[INFO]   Accuracy: 0.4835   Log Loss: 0.9881   Brier: 0.2101",
        "",
        "[INFO] LightGBM",
        "[INFO]   Accuracy: 0.4821   Log Loss: 0.9914   Brier: 0.2108",
    ], styles))
    story.append(info(
        "Don't worry about what these numbers mean yet — Section 5 explains each metric "
        "in detail with clear thresholds for 'good' and 'bad'.",
        styles))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Output Files After Training", styles["section"]))
    story.append(bullet_list([
        "<b>models/baseline_model.joblib</b>  — saved Logistic Regression model",
        "<b>models/xgboost_model.joblib</b>  — saved XGBoost model",
        "<b>models/lightgbm_model.joblib</b>  — saved LightGBM model",
        "<b>data/processed/all_matches_features.csv</b>  — data with engineered features",
        "<b>data/visualizations/calibration_xgboost.png</b>  — calibration chart",
        "<b>data/visualizations/calibration_lightgbm.png</b>  — calibration chart",
        "<b>data/visualizations/feature_importance_xgboost.png</b>  — feature chart",
        "<b>data/visualizations/feature_importance_lightgbm.png</b>  — feature chart",
    ], styles))
    story.append(PageBreak())


def build_evaluation(story, styles):
    story.append(Paragraph("5. Step 3 — Evaluating Model Quality", styles["chapter"]))

    story.append(Paragraph("The Three Core Metrics", styles["section"]))
    story.append(Paragraph(
        "The model reports three metrics after training.  Here is what each one means:",
        styles["body"]))

    # --- Accuracy ---
    story.append(Paragraph("Metric 1: Accuracy", styles["subsection"]))
    story.append(Paragraph(
        "<b>What it is:</b>  The percentage of matches where the model correctly predicted the outcome "
        "(Home Win, Draw, or Away Win).",
        styles["body"]))
    story.append(Paragraph(
        "<b>Formula:</b>  Correct predictions ÷ Total predictions",
        styles["body"]))
    story.append(Paragraph(
        "<b>Example:</b>  Accuracy of 0.48 means the model predicted the right outcome in 48 out of 100 matches.",
        styles["body"]))
    story.append(Spacer(1, 0.15*cm))
    data = [
        [Paragraph("Accuracy Range", styles["table_header"]),
         Paragraph("What it means", styles["table_header"]),
         Paragraph("Assessment", styles["table_header"])],
        [Paragraph("< 0.40", styles["table_cell"]),
         Paragraph("Worse than random guessing", styles["table_cell_l"]),
         Paragraph("Poor — something is wrong", styles["table_cell"])],
        [Paragraph("0.40 – 0.46", styles["table_cell"]),
         Paragraph("Below average for football prediction", styles["table_cell_l"]),
         Paragraph("Needs improvement", styles["table_cell"])],
        [Paragraph("0.47 – 0.53", styles["table_cell"]),
         Paragraph("Typical range for football ML models", styles["table_cell_l"]),
         Paragraph("Good ✓", styles["table_cell"])],
        [Paragraph("> 0.53", styles["table_cell"]),
         Paragraph("Exceptional — verify for data leakage", styles["table_cell_l"]),
         Paragraph("Suspicious — double-check", styles["table_cell"])],
    ]
    story.append(metric_table(data, [3.5*cm, 9*cm, 4*cm], styles))
    story.append(Spacer(1, 0.1*cm))
    story.append(info(
        "Football is inherently unpredictable — even the best models rarely exceed 55% accuracy. "
        "An accuracy above 60% almost certainly indicates a bug or data leakage.",
        styles))
    story.append(Spacer(1, 0.3*cm))

    # --- Log Loss ---
    story.append(Paragraph("Metric 2: Log Loss (Cross-Entropy Loss)", styles["subsection"]))
    story.append(Paragraph(
        "<b>What it is:</b>  Measures how confident and correct the model's probability estimates are.  "
        "Unlike accuracy, it cares about how sure the model was — not just whether it was right or wrong.",
        styles["body"]))
    story.append(Paragraph(
        "<b>Direction:</b>  <b>Lower is better.</b>  "
        "A perfect model would have Log Loss = 0.  A model that always predicts 33% for all three "
        "outcomes (pure guessing) has Log Loss ≈ 1.099.",
        styles["body"]))
    data = [
        [Paragraph("Log Loss Range", styles["table_header"]),
         Paragraph("Assessment", styles["table_header"])],
        [Paragraph("> 1.10", styles["table_cell"]),
         Paragraph("Worse than random — check your data", styles["table_cell"])],
        [Paragraph("1.00 – 1.10", styles["table_cell"]),
         Paragraph("About as good as random guessing", styles["table_cell"])],
        [Paragraph("0.95 – 1.00", styles["table_cell"]),
         Paragraph("Slightly better than random — acceptable", styles["table_cell"])],
        [Paragraph("0.85 – 0.95", styles["table_cell"]),
         Paragraph("Good — model has real predictive power", styles["table_cell"])],
        [Paragraph("< 0.85", styles["table_cell"]),
         Paragraph("Excellent — verify for data leakage", styles["table_cell"])],
    ]
    story.append(metric_table(data, [4.5*cm, 12*cm], styles))
    story.append(Spacer(1, 0.3*cm))

    # --- Brier Score ---
    story.append(Paragraph("Metric 3: Brier Score", styles["subsection"]))
    story.append(Paragraph(
        "<b>What it is:</b>  Measures the mean squared error between the model's predicted probabilities "
        "and the actual outcomes.  Think of it as a 'how wrong were the probabilities' score.",
        styles["body"]))
    story.append(Paragraph(
        "<b>Direction:</b>  <b>Lower is better.</b>  "
        "The worst possible score is 1.0; a random model scores ~0.222 for a 3-class problem.",
        styles["body"]))
    data = [
        [Paragraph("Brier Score", styles["table_header"]),
         Paragraph("Assessment", styles["table_header"])],
        [Paragraph("> 0.25", styles["table_cell"]),
         Paragraph("Worse than random", styles["table_cell"])],
        [Paragraph("0.21 – 0.25", styles["table_cell"]),
         Paragraph("About as good as random — needs work", styles["table_cell"])],
        [Paragraph("0.18 – 0.21", styles["table_cell"]),
         Paragraph("Good — probabilities are meaningful", styles["table_cell"])],
        [Paragraph("< 0.18", styles["table_cell"]),
         Paragraph("Excellent", styles["table_cell"])],
    ]
    story.append(metric_table(data, [4.5*cm, 12*cm], styles))
    story.append(Spacer(1, 0.3*cm))

    # Calibration
    story.append(Paragraph("Reading the Calibration Curve", styles["section"]))
    story.append(Paragraph(
        "The calibration curve is saved to <b>data/visualizations/calibration_xgboost.png</b>.  "
        "It shows three charts — one for each outcome (Home Win, Draw, Away Win).",
        styles["body"]))
    story.append(Paragraph(
        "Each chart has a <b>dashed diagonal line</b> representing perfect calibration.  "
        "The model's line (solid with dots) should follow this diagonal closely.",
        styles["body"]))
    story.append(bullet_list([
        "<b>Line above the diagonal:</b>  model is <i>under-confident</i> — it predicts 40% but actually wins 55% of the time",
        "<b>Line below the diagonal:</b>  model is <i>over-confident</i> — it predicts 70% but only wins 50% of the time",
        "<b>Line on the diagonal:</b>  <i>perfect calibration</i> — predicted probabilities match reality",
    ], styles))
    story.append(tip(
        "For value betting, calibration is more important than raw accuracy. "
        "A well-calibrated model's probability estimates can be directly compared to bookmaker odds.",
        styles))
    story.append(Spacer(1, 0.3*cm))

    # Feature Importance
    story.append(Paragraph("Reading the Feature Importance Chart", styles["section"]))
    story.append(Paragraph(
        "The feature importance chart shows which input variables had the most influence on predictions.  "
        "Open <b>data/visualizations/feature_importance_xgboost.png</b>.",
        styles["body"]))
    story.append(Paragraph("Common top features you should expect to see:", styles["body"]))
    story.append(bullet_list([
        "<b>home_form_5_points</b>  — home team's points in last 5 matches",
        "<b>away_form_5_points</b>  — away team's points in last 5 matches",
        "<b>true_prob_home_win / true_prob_away_win</b>  — implied probabilities from bookmaker odds (after removing vig)",
        "<b>h2h_home_win_rate</b>  — historical head-to-head home win rate",
        "<b>home_goal_diff / away_goal_diff</b>  — goal difference over recent form window",
    ], styles))
    story.append(warn(
        "If odds-based features (true_prob_*) dominate with very high importance, the model is "
        "largely following the bookmaker's probabilities.  This is normal — bookmakers are very accurate — "
        "but it limits the model's edge.",
        styles))
    story.append(PageBreak())


def build_backtest(story, styles):
    story.append(Paragraph("6. Step 4 — Running the Backtest", styles["chapter"]))

    story.append(Paragraph("What Backtesting Is", styles["section"]))
    story.append(Paragraph(
        "Backtesting simulates placing bets on the <b>test set</b> (the 15% of matches not used for training) "
        "using your model's predictions and the actual historical odds.  "
        "It tells you: 'If you had used this model during this period, what would have happened to your bankroll?'",
        styles["body"]))
    story.append(Paragraph(
        "The simulation applies these rules (set in config.yaml):",
        styles["body"]))
    story.append(bullet_list([
        "<b>Only bet when edge > 10%</b>  (model probability − implied probability > 0.10)",
        "<b>Only bet within odds range 1.8 – 6.0</b>  (avoids huge favourites and longshots)",
        "<b>Stake = 1/20th Kelly</b>  (very conservative — small stakes, low risk)",
        "<b>Maximum stake = 5% of starting bankroll per bet</b>  (hard cap on any single bet)",
        "<b>Starting bankroll = £1,000</b>",
    ], styles))

    story.append(Paragraph("How to Run the Backtest", styles["section"]))
    story.extend(code_block([
        "# Default: uses XGBoost model",
        "python main.py --backtest",
        "",
        "# To test a different model:",
        "python main.py --backtest --model lightgbm_model",
        "python main.py --backtest --model baseline_model",
    ], styles))
    story.append(info(
        "The backtest requires training to have been completed first. "
        "If you get an error saying 'engineered features not found', run --train first.",
        styles))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Reading the Backtest Report", styles["section"]))
    story.append(Paragraph(
        "The report is printed to the terminal and saved to "
        "<b>data/visualizations/backtest_report.txt</b>.  "
        "Here is an example report and what each line means:",
        styles["body"]))
    story.extend(code_block([
        "================================================================================",
        "BACKTEST REPORT",
        "================================================================================",
        "",
        "OVERALL METRICS:",
        "  Total Bets:        312",
        "  Total Staked:      $4,820.50",
        "  Total Profit:      $1,417.23",
        "  ROI:               29.40%",
        "  Win Rate:          48.40%",
        "  Average Odds:      2.84",
        "  Final Bankroll:    $2,417.23",
        "",
        "BET TYPE BREAKDOWN:",
        "  Home Win:",
        "    Bets: 148    Profit: $820.11    ROI: 31.2%    Win Rate: 52.0%",
        "  Draw:",
        "    Bets:  34    Profit: $112.44    ROI: 18.4%    Win Rate: 29.4%",
        "  Away Win:",
        "    Bets: 130    Profit: $484.68    ROI: 27.8%    Win Rate: 43.8%",
    ], styles))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("What Each Number Means", styles["section"]))
    data = [
        [Paragraph("Field", styles["table_header"]),
         Paragraph("Plain English Explanation", styles["table_header"]),
         Paragraph("Target", styles["table_header"])],
        [Paragraph("Total Bets", styles["table_cell"]),
         Paragraph("How many bets met the value threshold", styles["table_cell_l"]),
         Paragraph("> 100", styles["table_cell"])],
        [Paragraph("Total Staked", styles["table_cell"]),
         Paragraph("Total money wagered across all bets", styles["table_cell_l"]),
         Paragraph("—", styles["table_cell"])],
        [Paragraph("Total Profit", styles["table_cell"]),
         Paragraph("Net profit/loss after all bets settled", styles["table_cell_l"]),
         Paragraph("> 0", styles["table_cell"])],
        [Paragraph("ROI", styles["table_cell"]),
         Paragraph("Profit as % of total staked — the key metric", styles["table_cell_l"]),
         Paragraph("> 5%", styles["table_cell"])],
        [Paragraph("Win Rate", styles["table_cell"]),
         Paragraph("% of bets that won — misleading alone, see note", styles["table_cell_l"]),
         Paragraph("Depends on odds", styles["table_cell"])],
        [Paragraph("Average Odds", styles["table_cell"]),
         Paragraph("Mean decimal odds of all bets placed", styles["table_cell_l"]),
         Paragraph("1.8 – 4.0", styles["table_cell"])],
        [Paragraph("Final Bankroll", styles["table_cell"]),
         Paragraph("Starting £1,000 plus/minus net profit", styles["table_cell_l"]),
         Paragraph("> £1,000", styles["table_cell"])],
    ]
    story.append(metric_table(data, [3.5*cm, 9.5*cm, 3.5*cm], styles))
    story.append(Spacer(1, 0.2*cm))
    story.append(warn(
        "Win Rate alone is not a good metric for betting.  At average odds of 2.84 you only "
        "need a ~35% win rate to break even.  The model's 48% win rate at those odds implies "
        "genuine edge.",
        styles))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Understanding the ROI Curve", styles["section"]))
    story.append(Paragraph(
        "Open <b>data/visualizations/roi_curve.png</b>.  "
        "It shows two sub-charts:",
        styles["body"]))
    story.append(bullet_list([
        "<b>Top chart — Cumulative Profit:</b>  "
        "starts at £0 and shows profit/loss after each successive bet.  "
        "A healthy model trends upward overall with normal drawdown periods (going temporarily below £0).",
        "<b>Bottom chart — Bankroll over time:</b>  "
        "starts at £1,000.  A healthy curve trends upward.  "
        "Watch for prolonged flat or downward periods (drawdowns).",
    ], styles))
    story.append(tip(
        "A perfectly smooth upward line would be suspicious — real betting has variance. "
        "Expect zigzag patterns with an overall positive trend if the model has genuine edge.",
        styles))
    story.append(PageBreak())


def build_interpretation(story, styles):
    story.append(Paragraph("7. Interpreting Your Results", styles["chapter"]))

    story.append(Paragraph("Complete Metrics Quick-Reference", styles["section"]))
    data = [
        [Paragraph("Metric", styles["table_header"]),
         Paragraph("Good Range", styles["table_header"]),
         Paragraph("Red Flag", styles["table_header"]),
         Paragraph("Notes", styles["table_header"])],
        [Paragraph("Accuracy", styles["table_cell"]),
         Paragraph("0.47 – 0.53", styles["table_cell"]),
         Paragraph("> 0.58", styles["table_cell"]),
         Paragraph("Football is hard to predict", styles["table_cell_l"])],
        [Paragraph("Log Loss", styles["table_cell"]),
         Paragraph("0.85 – 0.98", styles["table_cell"]),
         Paragraph("< 0.70", styles["table_cell"]),
         Paragraph("Lower is better", styles["table_cell_l"])],
        [Paragraph("Brier Score", styles["table_cell"]),
         Paragraph("0.18 – 0.21", styles["table_cell"]),
         Paragraph("< 0.15", styles["table_cell"]),
         Paragraph("Lower is better", styles["table_cell_l"])],
        [Paragraph("Backtest ROI", styles["table_cell"]),
         Paragraph("5% – 30%", styles["table_cell"]),
         Paragraph("> 50%", styles["table_cell"]),
         Paragraph("Highly variable season to season", styles["table_cell_l"])],
        [Paragraph("Win Rate", styles["table_cell"]),
         Paragraph("35% – 55%", styles["table_cell"]),
         Paragraph("< 25% or > 65%", styles["table_cell"]),
         Paragraph("Depends heavily on avg odds", styles["table_cell_l"])],
        [Paragraph("Total Bets", styles["table_cell"]),
         Paragraph("> 100", styles["table_cell"]),
         Paragraph("< 50", styles["table_cell"]),
         Paragraph("Too few bets = unreliable stats", styles["table_cell_l"])],
        [Paragraph("Avg Odds", styles["table_cell"]),
         Paragraph("1.8 – 4.5", styles["table_cell"]),
         Paragraph("< 1.5 or > 7.0", styles["table_cell"]),
         Paragraph("Matches config min/max odds", styles["table_cell_l"])],
    ]
    story.append(metric_table(data, [3*cm, 3*cm, 3*cm, 7.5*cm], styles))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Red Flags — Things to Watch Out For", styles["section"]))
    red_flags = [
        ("Accuracy > 58% or ROI > 50%",
         "These are unrealistically high results for football betting.  "
         "It almost certainly means the model accidentally 'saw' future data during training "
         "(data leakage).  Check that the chronological split is working correctly."),
        ("Total bets < 50 in the backtest",
         "Too few bets makes all statistics unreliable (high variance).  "
         "Try lowering the value_threshold in config.yaml from 0.10 to 0.07."),
        ("Calibration curve far from diagonal",
         "The model's probabilities are systematically biased.  "
         "This affects value bet detection.  Try re-training with more data."),
        ("Final bankroll < £500 (lost > 50% starting capital)",
         "The strategy is losing money significantly.  "
         "Try raising the value_threshold (more selective) or using a different model."),
        ("All bets of the same type (e.g. all Home Win)",
         "The model may have a bias.  Check the feature importance chart."),
    ]
    for flag, explanation in red_flags:
        story.append(CalloutBox(
            [Paragraph(f"<b>🚩 {flag}</b>", styles["callout_label"]),
             Paragraph(explanation, styles["callout_body"])],
            WARN_BG, WARN_BORDER))
        story.append(Spacer(1, 0.15*cm))
    story.append(PageBreak())


def build_config(story, styles):
    story.append(Paragraph("8. Tweaking the Model (config.yaml)", styles["chapter"]))

    story.append(Paragraph(
        "The file <b>config.yaml</b> is your control panel.  "
        "You can change settings without editing any Python code.  "
        "Always make a backup before changing things: copy config.yaml to config_backup.yaml first.",
        styles["body"]))
    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("Key Settings", styles["section"]))
    data = [
        [Paragraph("Setting", styles["table_header"]),
         Paragraph("Default", styles["table_header"]),
         Paragraph("What it does", styles["table_header"]),
         Paragraph("Try lowering to...", styles["table_header"])],
        [Paragraph("backtest.value_threshold", styles["table_cell"]),
         Paragraph("0.10 (10%)", styles["table_cell"]),
         Paragraph("Minimum edge required to place a bet.  Higher = fewer but higher-quality bets.", styles["table_cell_l"]),
         Paragraph("0.07 for more bets", styles["table_cell"])],
        [Paragraph("backtest.kelly_fraction", styles["table_cell"]),
         Paragraph("0.05 (1/20 Kelly)", styles["table_cell"]),
         Paragraph("Controls stake size.  Lower = smaller stakes, lower risk, lower reward.", styles["table_cell_l"]),
         Paragraph("0.02 for smaller stakes", styles["table_cell"])],
        [Paragraph("backtest.min_odds", styles["table_cell"]),
         Paragraph("1.80", styles["table_cell"]),
         Paragraph("Skip bets below this odds — avoids heavy favourites.", styles["table_cell_l"]),
         Paragraph("1.5 to include more favourites", styles["table_cell"])],
        [Paragraph("backtest.max_odds", styles["table_cell"]),
         Paragraph("6.00", styles["table_cell"]),
         Paragraph("Skip bets above this odds — avoids longshots.", styles["table_cell_l"]),
         Paragraph("4.0 for more conservative", styles["table_cell"])],
        [Paragraph("models.test_size", styles["table_cell"]),
         Paragraph("0.15 (15%)", styles["table_cell"]),
         Paragraph("Proportion of data reserved for testing (not training).", styles["table_cell_l"]),
         Paragraph("0.20 for larger test set", styles["table_cell"])],
        [Paragraph("models.xgboost.n_estimators", styles["table_cell"]),
         Paragraph("300", styles["table_cell"]),
         Paragraph("Number of trees in XGBoost.  More = slower training, potentially better model.", styles["table_cell_l"]),
         Paragraph("500 for experimentation", styles["table_cell"])],
    ]
    story.append(metric_table(data, [4.5*cm, 2.5*cm, 6.5*cm, 3*cm], styles))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Conservative vs Aggressive Configurations", styles["section"]))
    story.extend(code_block([
        "# --- CONSERVATIVE (fewer bets, higher confidence, lower risk) ---",
        "backtest:",
        "  value_threshold: 0.15   # only bet when 15%+ edge",
        "  kelly_fraction:  0.02   # very small stakes",
        "  min_odds: 2.00          # skip short-odds favourites",
        "  max_odds: 5.00          # skip longshots",
        "",
        "# --- AGGRESSIVE (more bets, lower confidence, higher risk) ---",
        "backtest:",
        "  value_threshold: 0.07   # bet when 7%+ edge",
        "  kelly_fraction:  0.10   # larger stakes",
        "  min_odds: 1.60",
        "  max_odds: 8.00",
    ], styles))
    story.append(warn(
        "Aggressive settings can amplify both profits AND losses.  "
        "Always test changes in the backtest simulation before considering any real-money application.",
        styles))
    story.append(PageBreak())


def build_troubleshooting(story, styles):
    story.append(Paragraph("9. Common Problems &amp; Fixes", styles["chapter"]))

    problems = [
        (
            "ModuleNotFoundError: No module named 'xgboost'",
            "A required library is not installed.",
            [
                "Run: pip install -r requirements.txt",
                "Make sure your virtual environment is activated (source venv/bin/activate)",
            ]
        ),
        (
            "FileNotFoundError: data/processed/all_matches.csv not found",
            "Data hasn't been downloaded yet.",
            [
                "Run: python main.py --download",
                "Then re-run the failing step",
            ]
        ),
        (
            "Engineered features not found — run training first",
            "The backtest is trying to read feature data that hasn't been generated.",
            [
                "Run: python main.py --train  first",
                "Then run: python main.py --backtest",
            ]
        ),
        (
            "Total Bets = 0 in backtest report",
            "No bets met the value threshold — the model found no edge above 10%.",
            [
                "In config.yaml, lower value_threshold from 0.10 to 0.07",
                "Or check that bookmaker odds columns (B365H, B365D, B365A) exist in your data",
            ]
        ),
        (
            "Training takes > 30 minutes",
            "Your computer may be running the feature engineering step on very large data.",
            [
                "This is normal on older hardware — let it run",
                "You can reduce n_estimators in config.yaml from 300 to 100 for faster (but less accurate) training",
            ]
        ),
        (
            "UnicodeDecodeError when loading CSV",
            "Some older match CSV files use non-UTF-8 encoding.",
            [
                "Delete the data/raw/ folder and re-download: python main.py --download",
                "The data loader handles encoding automatically for most files",
            ]
        ),
        (
            "Calibration chart shows a flat horizontal line",
            "The model is outputting nearly identical probabilities for all matches — it has failed to learn.",
            [
                "This usually happens when features are all NaN.  Check your data has enough history (run --download first)",
                "Re-run with: python main.py --train",
            ]
        ),
    ]

    for error, cause, fixes in problems:
        story.append(Paragraph(f"<b>Problem:</b>  {error}", styles["body_bold"]))
        story.append(Paragraph(f"<b>Cause:</b>  {cause}", styles["body"]))
        story.append(Paragraph("<b>Fix:</b>", styles["body"]))
        story.append(numbered_list(fixes, styles, indent=4))
        story.append(ColoredLine(MID_GREY, thickness=0.5))
        story.append(Spacer(1, 0.15*cm))
    story.append(PageBreak())


def build_responsible(story, styles):
    story.append(Paragraph("10. Responsible Use &amp; Limitations", styles["chapter"]))

    story.append(Paragraph("Important Limitations", styles["section"]))
    story.append(bullet_list([
        "<b>Backtesting is not a guarantee.</b>  Historical performance can look very different from live performance due to market changes, line movements, and new bookmaker restrictions.",
        "<b>Bookmakers close winning accounts.</b>  Even if the model finds genuine edge, bookmakers may limit or close accounts of consistently profitable bettors.",
        "<b>The model uses delayed data.</b>  Team news, injuries, weather, and line-up changes are not captured in historical CSV data.",
        "<b>Odds data quality varies.</b>  Some historical odds may be missing or incorrect, which affects backtest reliability.",
        "<b>Small sample problem.</b>  Even 300 bets is a relatively small sample in statistics — genuine edge can look like luck over 300 bets, and luck can look like genuine edge.",
        "<b>Model drift.</b>  A model trained on 2010–2024 data may become less accurate as team dynamics, playing styles, and league structures change.",
    ], styles))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Responsible Use Guidelines", styles["section"]))
    story.append(CalloutBox(
        [Paragraph(
            "This model is a research and educational tool.  "
            "If you choose to use any insights for real-money betting, please follow these guidelines:",
            styles["callout_body"])],
        INFO_BG, INFO_BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(numbered_list([
        "<b>Never bet more than you can comfortably afford to lose.</b>  Treat any money used for betting as entertainment expense.",
        "<b>Set a fixed monthly budget</b>  and stick to it regardless of wins or losses.",
        "<b>Do not chase losses</b>  — a losing run is normal even with a profitable model.",
        "<b>Keep records</b>  — track every bet placed, stake, odds, and outcome.",
        "<b>Use demo / paper trading first</b>  — run 3–6 months of simulated bets before committing real money.",
        "<b>If gambling becomes a problem</b>, seek help at <b>BeGambleAware.org</b> or call <b>0808 8020 133</b> (UK, free, 24/7).",
    ], styles))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Quick Command Cheat Sheet", styles["section"]))
    story.extend(code_block([
        "# Step 1: Download data",
        "python main.py --download",
        "",
        "# Step 2: Train models",
        "python main.py --train",
        "",
        "# Step 3: Run backtest (XGBoost — default)",
        "python main.py --backtest",
        "",
        "# Step 3b: Run backtest with LightGBM",
        "python main.py --backtest --model lightgbm_model",
        "",
        "# Step 4: Run entire pipeline in one command",
        "python main.py --all",
        "",
        "# Step 5: Use the graphical interface instead",
        "python gradio_app.py",
        "# Then open http://localhost:7860 in your browser",
    ], styles))
    story.append(Spacer(1, 0.3*cm))
    story.append(tip(
        "Use python gradio_app.py to access a click-based web interface if you prefer "
        "not to use the terminal.  It offers the same functionality with buttons and progress bars.",
        styles))


# ── Main ─────────────────────────────────────────────────────────────────────
def build_pdf(output_path: str = "Football_Model_Testing_Guide.pdf"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=1.8*cm,
        bottomMargin=1.5*cm,
        title="Football Betting Model — Testing & Benchmarking Guide",
        author="Generated Guide",
        subject="ML Model Testing for Beginners",
    )

    styles = build_styles()
    story = []

    # Pages
    build_cover(story, styles)
    build_toc(story, styles)
    build_intro(story, styles)
    build_setup(story, styles)
    build_download(story, styles)
    build_training(story, styles)
    build_evaluation(story, styles)
    build_backtest(story, styles)
    build_interpretation(story, styles)
    build_config(story, styles)
    build_troubleshooting(story, styles)
    build_responsible(story, styles)

    # Build with alternating page templates
    def page_dispatcher(canvas, doc):
        if doc.page == 1:
            cover_bg(canvas, doc)
        else:
            normal_page(canvas, doc)

    doc.build(story, onFirstPage=page_dispatcher, onLaterPages=page_dispatcher)
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
