"""Reproducible vector figure using the existing bundled ReportLab runtime.

The current training/bundled runtimes do not contain Matplotlib. This script
uses ReportLab's LinePlot chart engine and Poppler renders the PDF to PNG.
All plotted estimates come from english_confidence_comparison.json.
"""
import hashlib
import json
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Circle, Group
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent
source = ROOT / "english_confidence_comparison.json"
report = json.loads(source.read_text(encoding="utf-8"))
V2 = colors.HexColor("#365f91")
V3 = colors.HexColor("#078578")
INK = colors.HexColor("#172334")
MUTED = colors.HexColor("#596779")
GRID = colors.HexColor("#dce2e8")
d = Drawing(1020, 700)
d.add(Rect(0, 0, 1020, 700, strokeColor=None, fillColor=colors.white))


def label(x, y, text, size=11, color=INK, font="Helvetica", anchor="start"):
    d.add(String(x, y, text, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


label(42, 658, "Parakeet v2 and v3: English confidence", 24, font="Helvetica-Bold")
label(42, 634, "100 fixed FLEURS test clips; temperatures fit on 50 separate development clips", 12, MUTED)
label(42, 588, "A  Word reliability", 15, font="Helvetica-Bold")
label(574, 588, "B  Confidence on the same correct words", 15, font="Helvetica-Bold")

# Standard plotting engine: each point is a nonempty confidence bin.
chart = LinePlot()
chart.x = 95
chart.y = 282
chart.width = 410
chart.height = 280
data = [[(0, 0), (1, 1)]]
styles = [(colors.HexColor("#9ba6b2"), None)]
for model, color in [("v2", V2), ("v3", V3)]:
    for key, raw in [("raw_test_word_confidence", True), ("calibrated_test_word_confidence", False)]:
        bins = report["models"][model][key]["reliability_bins"]
        data.append([(b["mean_confidence"], b["accuracy"]) for b in bins if b["count"]])
        styles.append((color, raw))
chart.data = data
chart.xValueAxis.valueMin = chart.yValueAxis.valueMin = 0
chart.xValueAxis.valueMax = chart.yValueAxis.valueMax = 1
chart.xValueAxis.valueSteps = chart.yValueAxis.valueSteps = [0, .2, .4, .6, .8, 1]
for axis in (chart.xValueAxis, chart.yValueAxis):
    axis.strokeColor = GRID
    axis.labels.fontName = "Helvetica"
    axis.labels.fontSize = 10
    axis.labels.fillColor = MUTED
    axis.gridStrokeColor = GRID
    axis.visibleGrid = True
for i, (color, raw) in enumerate(styles):
    chart.lines[i].strokeColor = color
    chart.lines[i].strokeWidth = 1.3 if i == 0 else 1.7
    if i == 0:
        chart.lines[i].strokeDashArray = [4, 4]
    else:
        chart.lines[i].symbol = makeMarker("FilledCircle")
        chart.lines[i].symbol.fillColor = color
        chart.lines[i].symbol.size = 5
        if raw:
            chart.lines[i].strokeDashArray = [4, 3]
d.add(chart)
label(300, 251, "Mean emitted-word confidence", 11, anchor="middle")
y_axis_label = Group()
y_axis_label.add(String(0, 0, "Observed word correctness", fontSize=11, fillColor=INK, textAnchor="middle"))
y_axis_label.transform = (0, 1, -1, 0, 45, 420)
d.add(y_axis_label)
legend = [("v2 raw", V2, True), ("v2 dev-fit T=2", V2, False), ("v3 raw", V3, True), ("v3 dev-fit T=1.5", V3, False)]
for i, (text, color, raw) in enumerate(legend):
    x = 84 + (i % 2) * 222
    y = 225 - (i // 2) * 20
    d.add(Line(x, y + 3, x + 25, y + 3, strokeColor=color, strokeWidth=1.8, strokeDashArray=[4, 3] if raw else None))
    label(x + 33, y, text, 10)
label(84, 177, "Ten bins; low-count bins are noisy. Dashed diagonal = ideal reliability.", 9, MUTED)

# Paired-word point estimates and recording-cluster percentile intervals.
shared = report["shared_correct_words"]
uncertainty = report["shared_correct_word_uncertainty"]
label(574, 557, f"{shared['shared_correct_word_count']:,} words correct in both systems", 11, MUTED)
label(574, 539, "v3 minus v2, percentage points; 95% paired-recording bootstrap CI", 10, MUTED)
x0, x1, lo, hi = 680, 965, -.45, .10


def xpos(value):
    return x0 + (value - lo) / (hi - lo) * (x1 - x0)


for tick in [-.4, -.3, -.2, -.1, 0, .1]:
    d.add(Line(xpos(tick), 385, xpos(tick), 496, strokeColor=GRID, strokeWidth=.7))
    label(xpos(tick), 368, f"{tick:.1f}", 10, MUTED, anchor="middle")
d.add(Line(xpos(0), 385, xpos(0), 496, strokeColor=MUTED, strokeWidth=1.1, strokeDashArray=[3, 3]))
for name, key, y in [("Full vocabulary", "mean_word_confidence", 470), ("Nonblank only", "mean_conditional_nonblank_word_confidence", 407)]:
    value = 100 * shared["delta_v3_minus_v2"][key]
    low, high = [100 * number for number in uncertainty["delta_v3_minus_v2_ci"][key]]
    label(574, y - 3, name, 10)
    d.add(Line(xpos(low), y, xpos(high), y, strokeColor=V3, strokeWidth=2.1))
    for endpoint in (low, high):
        d.add(Line(xpos(endpoint), y - 5, xpos(endpoint), y + 5, strokeColor=V3, strokeWidth=1.6))
    d.add(Circle(xpos(value), y, 4, fillColor=V3, strokeColor=None))
    label(xpos(value), y + 16, f"{value:+.3f} [{low:+.3f}, {high:+.3f}]", 9, anchor="middle")
label(825, 345, "Lower v3 confidence                         Higher v3 confidence", 9, MUTED, anchor="middle")
label(574, 316, "Full-vocabulary confidence includes blank competition.", 10)
label(574, 299, "Nonblank conditioning diagnoses timing uncertainty; it is not calibrated.", 9, MUTED)

# Minimal numeric table keeps the actual calibration comparison readable.
label(574, 267, "Held-out English measures", 12, font="Helvetica-Bold")
columns = [(574, "Measure"), (755, "v2"), (863, "v3")]
for x, name in columns:
    label(x, 244, name, 10, font="Helvetica-Bold")
d.add(Line(574, 237, 965, 237, strokeColor=GRID))
rows = [
    ("WER (%)", "english_asr_on_common_test_recordings", "wer", 100),
    ("Raw ECE (%)", "raw_test_word_confidence", "ece", 100),
    ("Dev-fit ECE (%)", "calibrated_test_word_confidence", "ece", 100),
    ("Raw Brier", "raw_test_word_confidence", "brier", 1),
    ("Dev-fit Brier", "calibrated_test_word_confidence", "brier", 1),
]
for i, (name, section, key, factor) in enumerate(rows):
    y = 219 - 19 * i
    label(574, y, name, 10)
    for x, model in [(755, "v2"), (863, "v3")]:
        value = factor * report["models"][model][section][key]
        label(x, y, f"{value:.3f}" if factor == 100 else f"{value:.4f}", 10)

d.add(Line(42, 108, 978, 108, strokeColor=GRID))
label(42, 85, "Raw correct-word scores are near 1 in both models; higher confidence alone does not establish better calibration.", 11)
label(42, 66, "Calibration evaluates emitted words; deletions remain separate ASR errors. Temperature changes confidence, not transcripts.", 10, MUTED)
label(42, 46, "Source: english_confidence_comparison.json | Small clean/read-speech pilot; confidence intervals resample recordings, not sentence clusters.", 9, MUTED)
renderPDF.drawToFile(d, str(ROOT / "english_confidence_figure.pdf"))
(ROOT / "english_confidence_figure_metadata.json").write_text(json.dumps({"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "renderer": "ReportLab LinePlot; no package install or GPU use", "test_recordings": report["overlap"]["test"]["common_recordings"], "validation_recordings": report["overlap"]["validation"]["common_recordings"], "shared_correct_word_count": shared["shared_correct_word_count"]}, indent=2), encoding="utf-8")
print(ROOT / "english_confidence_figure.pdf")
