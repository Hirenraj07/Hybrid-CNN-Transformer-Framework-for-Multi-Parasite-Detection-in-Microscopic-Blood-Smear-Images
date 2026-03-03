import os
import json
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# --------------------------------------------------
# Flask Setup
# --------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# Load Model (.keras format)
# --------------------------------------------------
model = tf.keras.models.load_model(
    "hybrid_efficientnet_vit_finetuned_99.keras",
    compile=False
)

# --------------------------------------------------
# Load Class Indices
# --------------------------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# --------------------------------------------------
# Globals for PDF
# --------------------------------------------------
last_prediction = None
last_confidence = None
last_risk = None
last_margin = None

# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------
last_conv_layer_name = "top_conv"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --------------------------------------------------
# Main Route
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global last_prediction, last_confidence, last_risk, last_margin

    prediction = None
    confidence = None
    result_image = None
    probabilities = None
    risk = None
    badge_color = None
    margin = None

    if request.method == "POST":

        file = request.files["image"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess
        img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224,224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_exp = np.expand_dims(img_array, axis=0)
        img_array_exp = tf.keras.applications.efficientnet.preprocess_input(img_array_exp)

        # Prediction
        preds = model.predict(img_array_exp)
        sorted_probs = np.sort(preds[0])
        top1 = float(sorted_probs[-1])
        top2 = float(sorted_probs[-2])

        margin = top1 - top2
        pred_class = np.argmax(preds[0])

        confidence = top1
        prediction = index_to_class[pred_class]
        probabilities = preds[0].tolist()

        # Risk Logic
        if confidence < 0.70 or margin < 0.15:
            risk = "Low Confidence – Review Recommended"
            badge_color = "#dc3545"
        elif confidence < 0.90:
            risk = "Moderate Confidence – Clinical Correlation Suggested"
            badge_color = "#ffc107"
        else:
            risk = "High Confidence – Strong Model Agreement"
            badge_color = "#28a745"

        # Store for PDF
        last_prediction = prediction
        last_confidence = confidence
        last_risk = risk
        last_margin = margin

        # Grad-CAM
        heatmap = make_gradcam_heatmap(
            img_array_exp,
            model,
            last_conv_layer_name,
            pred_index=pred_class
        )

        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + img_array
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, superimposed_img)

        result_image = "results/" + filename

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence*100,2) if confidence else None,
        margin=round(margin*100,2) if margin else None,
        result_image=result_image,
        probabilities=probabilities,
        class_labels=index_to_class,
        risk=risk,
        badge_color=badge_color
    )

# --------------------------------------------------
# PDF Route
# --------------------------------------------------
@app.route("/download_report", methods=["POST"])
def download_report():
    global last_prediction, last_confidence, last_risk, last_margin

    file_path = "diagnostic_report.pdf"
    now = datetime.now()
    report_id = f"PSA-{now.strftime('%Y%m%d%H%M%S')}"

    # ── Color palette ──────────────────────────────────────────
    DARK_TEAL   = colors.HexColor("#0d3d4a")
    MED_TEAL    = colors.HexColor("#0e5c6e")
    ACCENT_CYAN = colors.HexColor("#00b8a0")
    LIGHT_BG    = colors.HexColor("#f0fafa")
    ROW_ALT     = colors.HexColor("#e6f7f5")
    TEXT_DARK   = colors.HexColor("#0d2b33")
    TEXT_MUTED  = colors.HexColor("#4a7a85")
    WHITE       = colors.white

    # Risk colour
    if "High" in last_risk:
        RISK_COLOR = colors.HexColor("#1a7a4a")
        RISK_BG    = colors.HexColor("#e6f9ef")
    elif "Moderate" in last_risk:
        RISK_COLOR = colors.HexColor("#8a6200")
        RISK_BG    = colors.HexColor("#fff8e1")
    else:
        RISK_COLOR = colors.HexColor("#a00020")
        RISK_BG    = colors.HexColor("#fff0f2")

    # ── Custom styles ──────────────────────────────────────────
    styles = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    sTitle = S("sTitle",
        fontName="Helvetica-Bold", fontSize=22, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=4)
    sSubtitle = S("sSubtitle",
        fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#b2e8e0"),
        alignment=TA_CENTER, spaceAfter=2)
    sReportId = S("sReportId",
        fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#80d0c8"),
        alignment=TA_CENTER)
    sSectionHead = S("sSectionHead",
        fontName="Helvetica-Bold", fontSize=11, textColor=WHITE,
        alignment=TA_LEFT)
    sLabel = S("sLabel",
        fontName="Helvetica-Bold", fontSize=9, textColor=TEXT_MUTED)
    sValue = S("sValue",
        fontName="Helvetica-Bold", fontSize=14, textColor=TEXT_DARK)
    sNormal = S("sNormal",
        fontName="Helvetica", fontSize=9, textColor=TEXT_DARK,
        leading=14)
    sFooter = S("sFooter",
        fontName="Helvetica", fontSize=7.5, textColor=TEXT_MUTED,
        alignment=TA_CENTER)
    sDisc = S("sDisc",
        fontName="Helvetica-Oblique", fontSize=8, textColor=TEXT_MUTED,
        leading=12)

    # ── Page setup with custom header/footer canvas ────────────
    W, H = letter

    def on_page(canvas, doc):
        """Draw letterhead header and page footer on every page."""
        canvas.saveState()

        # Header banner
        canvas.setFillColor(DARK_TEAL)
        canvas.rect(0, H - 1.15*inch, W, 1.15*inch, fill=1, stroke=0)

        # Teal accent stripe
        canvas.setFillColor(ACCENT_CYAN)
        canvas.rect(0, H - 1.18*inch, W, 0.03*inch, fill=1, stroke=0)

        # Title text
        canvas.setFont("Helvetica-Bold", 18)
        canvas.setFillColor(WHITE)
        canvas.drawCentredString(W/2, H - 0.52*inch, "CLINICAL PARASITE DIAGNOSTIC REPORT")

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#80d0c8"))
        canvas.drawCentredString(W/2, H - 0.70*inch,
            "ParasiteScan AI  ·  Hybrid CNN–Transformer Diagnostic Engine  ·  v1.0")
        canvas.drawCentredString(W/2, H - 0.84*inch,
            f"Report ID: {report_id}   |   Generated: {now.strftime('%B %d, %Y  %H:%M:%S')}")

        # Footer line
        canvas.setStrokeColor(ACCENT_CYAN)
        canvas.setLineWidth(0.5)
        canvas.line(0.6*inch, 0.55*inch, W - 0.6*inch, 0.55*inch)

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawString(0.6*inch, 0.38*inch,
            "FOR RESEARCH USE ONLY – Not intended for clinical diagnosis without physician review.")
        canvas.drawRightString(W - 0.6*inch, 0.38*inch,
            f"Page {doc.page}")

        canvas.restoreState()

    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        topMargin=1.35*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.65*inch,
        rightMargin=0.65*inch,
    )

    elems = []
    CW = W - 1.3*inch   # content width

    # ── SECTION HELPER ─────────────────────────────────────────
    def section_header(title):
        tbl = Table(
            [[Paragraph(title, sSectionHead)]],
            colWidths=[CW]
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), MED_TEAL),
            ("TOPPADDING",   (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0), (-1,-1), 6),
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("ROUNDEDCORNERS", [4]),
        ]))
        return tbl

    # ── 1. DIAGNOSIS RESULT ─────────────────────────────────────
    elems.append(Spacer(1, 0.18*inch))
    elems.append(section_header("01 · DIAGNOSIS RESULT"))
    elems.append(Spacer(1, 8))

    conf_pct  = round(last_confidence * 100, 2)
    marg_pct  = round(last_margin     * 100, 2)

    diag_data = [
        [Paragraph("IDENTIFIED ORGANISM", sLabel),
         Paragraph("CONFIDENCE SCORE",    sLabel),
         Paragraph("DECISION MARGIN",     sLabel)],
        [Paragraph(last_prediction,        sValue),
         Paragraph(f"{conf_pct}%",         sValue),
         Paragraph(f"{marg_pct}%",         sValue)],
    ]
    diag_tbl = Table(diag_data, colWidths=[CW*0.44, CW*0.28, CW*0.28])
    diag_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1, 0), LIGHT_BG),
        ("BACKGROUND",    (0,1), (-1,-1), WHITE),
        ("BOX",           (0,0), (-1,-1), 0.8, ACCENT_CYAN),
        ("INNERGRID",     (0,0), (-1,-1), 0.4, colors.HexColor("#c0e8e4")),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS",[4]),
    ]))
    elems.append(diag_tbl)
    elems.append(Spacer(1, 10))

    # Risk badge row
    risk_data = [[Paragraph(f"RISK ASSESSMENT:  {last_risk}", S("rsk",
        fontName="Helvetica-Bold", fontSize=10,
        textColor=RISK_COLOR, alignment=TA_CENTER))]]
    risk_tbl = Table(risk_data, colWidths=[CW])
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), RISK_BG),
        ("BOX",           (0,0), (-1,-1), 1.2, RISK_COLOR),
        ("TOPPADDING",    (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ("ROUNDEDCORNERS",[5]),
    ]))
    elems.append(risk_tbl)

    # ── 2. CONFIDENCE METRICS ───────────────────────────────────
    elems.append(Spacer(1, 14))
    elems.append(section_header("02 · CONFIDENCE METRICS"))
    elems.append(Spacer(1, 8))

    metrics_data = [
        [Paragraph("Metric", S("mh", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
         Paragraph("Value",  S("mh", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
         Paragraph("Interpretation", S("mh", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE))],
        ["Top-1 Confidence",  f"{conf_pct}%",
         "Probability assigned to the predicted class"],
        ["Decision Margin",   f"{marg_pct}%",
         "Gap between top-1 and top-2 class probabilities"],
        ["Risk Level",        last_risk.split("–")[0].strip(),
         last_risk.split("–")[1].strip() if "–" in last_risk else last_risk],
        ["Model Agreement",
         "Strong" if "High" in last_risk else ("Moderate" if "Moderate" in last_risk else "Weak"),
         "Degree of model certainty across ensemble heads"],
    ]

    col_w = [CW*0.28, CW*0.20, CW*0.52]
    met_tbl = Table(metrics_data, colWidths=col_w)
    met_style = [
        ("BACKGROUND",    (0,0), (-1, 0), MED_TEAL),
        ("TEXTCOLOR",     (0,0), (-1, 0), WHITE),
        ("FONTNAME",      (0,0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_DARK),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, ROW_ALT]),
        ("BOX",           (0,0), (-1,-1), 0.8, ACCENT_CYAN),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0e8e4")),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]
    met_tbl.setStyle(TableStyle(met_style))
    elems.append(met_tbl)

    # ── 3. MODEL INFORMATION ────────────────────────────────────
    elems.append(Spacer(1, 14))
    elems.append(section_header("03 · MODEL & SYSTEM INFORMATION"))
    elems.append(Spacer(1, 8))

    model_data = [
        [Paragraph("Parameter", S("mh", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
         Paragraph("Details",   S("mh", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE))],
        ["Architecture",       "EfficientNet-B7 + Vision Transformer (ViT) Hybrid"],
        ["Explainability",     "Gradient-weighted Class Activation Mapping (Grad-CAM)"],
        ["Input Resolution",   "224 × 224 pixels (RGB)"],
        ["Preprocessing",      "EfficientNet standard normalisation"],
        ["Final Layer",        "top_conv (Grad-CAM target layer)"],
        ["Training Accuracy",  "99% (fine-tuned on parasite microscopy dataset)"],
        ["System Version",     "ParasiteScan AI v1.0 — Clinical Research Build"],
    ]

    mod_tbl = Table(model_data, colWidths=[CW*0.32, CW*0.68])
    mod_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1, 0), MED_TEAL),
        ("TEXTCOLOR",     (0,0), (-1, 0), WHITE),
        ("FONTNAME",      (0,0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), TEXT_DARK),
        ("FONTNAME",      (0,1),  (0,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,1),  (0,-1), TEXT_MUTED),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, ROW_ALT]),
        ("BOX",           (0,0), (-1,-1), 0.8, ACCENT_CYAN),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#c0e8e4")),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    elems.append(mod_tbl)

    # ── 4. CLINICAL NOTES ───────────────────────────────────────
    elems.append(Spacer(1, 14))
    elems.append(section_header("04 · CLINICAL NOTES & DISCLAIMER"))
    elems.append(Spacer(1, 8))

    notes_text = (
        "This report was generated automatically by the ParasiteScan AI diagnostic engine. "
        "The results represent the model's statistical prediction based on image features and "
        "should be interpreted in the context of clinical findings, patient history, and "
        "laboratory confirmation. A confidence score above 90% with a high decision margin "
        "indicates strong model agreement; however, all AI-assisted diagnoses must be reviewed "
        "and validated by a qualified healthcare professional before any clinical decisions are made. "
        "The Grad-CAM attention map highlights the image regions most influential in the model's "
        "decision and is provided for explainability purposes only."
    )
    elems.append(Paragraph(notes_text, sDisc))

    doc.build(elems, onFirstPage=on_page, onLaterPages=on_page)
    return send_file(file_path, as_attachment=True)

# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)