"""
========================================================================================
CLINICAL DECISION SUPPORT & WHO TRIAGE PROTOCOL ENGINE
Translates AI Predictions into Clinician-Grade Actionable Triage Dossiers
========================================================================================
Implements:
1. World Health Organization (WHO) Severe Malaria Diagnostic Scoring
2. Three-Tier Clinical Urgency Stratification (Red, Yellow, Green)
3. Actionable Treatment Recommendation Generator
4. Exportable / Printable Clean Medical Triage Report in HTML
========================================================================================
"""

import os
import datetime
import pandas as pd

try:
    from src.config import WHO_SEVERE_CRITERIA, FEATURE_DESCRIPTIONS
except ImportError:
    from config import WHO_SEVERE_CRITERIA, FEATURE_DESCRIPTIONS


def compute_who_danger_score(patient_data: dict) -> dict:
    """
    Evaluates patient symptoms against official WHO Severe Malaria clinical criteria.
    Returns composite score, danger flags detected, and severity level.
    """
    detected_flags = []
    total_score = 0
    max_possible_score = sum(item["severity_weight"] for item in WHO_SEVERE_CRITERIA.values())

    for symptom, info in WHO_SEVERE_CRITERIA.items():
        val = patient_data.get(symptom, 0)
        if int(val) == 1:
            total_score += info["severity_weight"]
            detected_flags.append({
                "symptom": symptom,
                "label": info["label"],
                "weight": info["severity_weight"]
            })

    # Triage Tier Determination
    if total_score >= 5 or patient_data.get("Convulsion", 0) == 1 or patient_data.get("cocacola_urine", 0) == 1:
        tier = "RED_EMERGENCY"
        tier_label = "CRITICAL / EMERGENCY TRIAGE (TIER 1)"
        color = "#d9534f"
        action = "Immediate hospital admission. Initiate parenteral Artesunate (IV/IM) stat; check blood glucose and hemoglobin."
    elif total_score >= 2:
        tier = "YELLOW_URGENT"
        tier_label = "URGENT CLINICAL OBSERVATION (TIER 2)"
        color = "#f0ad4e"
        action = "High clinical vigilance required. Re-evaluate vital signs every 2-4 hours, monitor hydration and blood glucose."
    else:
        tier = "GREEN_ROUTINE"
        tier_label = "ROUTINE OUTPATIENT CARE (TIER 3)"
        color = "#5cb85c"
        action = "Uncomplicated presentation. Prescribe oral Artemisinin-based Combination Therapy (ACT) if smear/RDT positive."

    return {
        "score": total_score,
        "max_score": max_possible_score,
        "score_percentage": round((total_score / max_possible_score) * 100, 1),
        "detected_flags": detected_flags,
        "tier": tier,
        "tier_label": tier_label,
        "tier_color": color,
        "recommended_action": action
    }


def generate_clinical_html_report(
    patient_data: dict,
    model_prediction: dict,
    conformal_result: dict = None,
    counterfactuals: list = None,
    doctor_notes: str = "Triage evaluation completed via ML Decision Support System."
) -> str:
    """
    Generates a print-ready, professional Medical Triage & Diagnostic Dossier in HTML.
    Can be printed directly or downloaded by the clinician.
    """
    who_res = compute_who_danger_score(patient_data)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_age = patient_data.get("age", "N/A")
    pred_label = "SEVERE MALARIA POSITIVE" if model_prediction.get("prediction", 0) == 1 else "MALARIA NEGATIVE / UNCOMPLICATED"
    prob_severe = model_prediction.get("probability", 0.0)

    # Compile symptoms table
    symptoms_rows = ""
    for k, v in patient_data.items():
        if k == "age":
            continue
        status_badge = '<span style="color:#d9534f;font-weight:bold;">PRESENT</span>' if int(v) == 1 else '<span style="color:#6c757d;">ABSENT</span>'
        desc = FEATURE_DESCRIPTIONS.get(k, k)
        symptoms_rows += f"<tr><td>{desc}</td><td style='text-align:center;'>{status_badge}</td></tr>"

    # Compile WHO flags
    who_flags_html = ""
    if who_res["detected_flags"]:
        for flag in who_res["detected_flags"]:
            who_flags_html += f"<li><strong>{flag['label']}</strong> (Severity Weight: +{flag['weight']})</li>"
    else:
        who_flags_html = "<li>No critical WHO severe red-flag markers detected.</li>"

    # Counterfactual interventions section
    cf_html = ""
    if counterfactuals:
        cf_html += "<h3>Recommended Actionable Reversal Targets (Counterfactual Analysis)</h3><ul>"
        for cf in counterfactuals[:2]:
            changes = [f"Treat/Resolve <strong>{k}</strong>" for k in cf.get("changes_required", {}).keys()]
            if changes:
                cf_html += f"<li>To shift status to low risk: {', '.join(changes)}</li>"
        cf_html += "</ul>"

    # Conformal uncertainty
    conf_text = ""
    if conformal_result:
        conf_text = f"<p><strong>Conformal Prediction Set ({conformal_result.get('confidence_level', '95%')} Confidence):</strong> {conformal_result.get('clinical_status')} - <em>{conformal_result.get('clinical_guidance')}</em></p>"

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Malaria Diagnostic Dossier</title>
        <style>
            body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; color: #2c3e50; line-height: 1.5; }}
            .header {{ border-bottom: 3px solid #2c3e50; padding-bottom: 15px; margin-bottom: 25px; }}
            .hospital-title {{ font-size: 24px; font-weight: bold; color: #1a365d; }}
            .sub-title {{ font-size: 13px; color: #718096; }}
            .badge {{ display: inline-block; padding: 6px 14px; font-size: 14px; font-weight: bold; border-radius: 4px; color: white; }}
            .badge-red {{ background-color: #d9534f; }}
            .badge-green {{ background-color: #5cb85c; }}
            .badge-yellow {{ background-color: #f0ad4e; }}
            .grid {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
            .card {{ border: 1px solid #e2e8f0; border-radius: 6px; padding: 15px; width: 48%; background-color: #f8fafc; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; font-size: 13px; }}
            th {{ background-color: #e2e8f0; text-align: left; }}
            .footer {{ margin-top: 40px; border-top: 1px solid #cbd5e1; padding-top: 15px; font-size: 12px; color: #64748b; }}
            .signature-box {{ margin-top: 30px; display: flex; justify-content: space-between; }}
            .sig-line {{ width: 250px; border-top: 1px solid #333; text-align: center; padding-top: 5px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="hospital-title">EXPLAINABLE CLINICAL MALARIA DECISION SUPPORT SYSTEM</div>
            <div class="sub-title">Advanced Diagnostic Triage Report | Generated: {timestamp}</div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Patient Telemetry</h3>
                <p><strong>Patient Age:</strong> {patient_age} years</p>
                <p><strong>Assessment Date:</strong> {timestamp[:10]}</p>
                <p><strong>Primary AI Prediction:</strong> <span class="badge { 'badge-red' if model_prediction.get('prediction', 0) == 1 else 'badge-green' }">{pred_label}</span></p>
                <p><strong>Calculated Severe Risk:</strong> {(prob_severe * 100):.1f}%</p>
            </div>
            <div class="card" style="border-left: 5px solid {who_res['tier_color']};">
                <h3>WHO Clinical Danger Assessment</h3>
                <p><strong>Triage Stratification:</strong> <span style="color:{who_res['tier_color']};font-weight:bold;">{who_res['tier_label']}</span></p>
                <p><strong>WHO Score:</strong> {who_res['score']} / {who_res['max_score']} ({who_res['score_percentage']}%)</p>
                <p><strong>Clinical Protocol:</strong> {who_res['recommended_action']}</p>
            </div>
        </div>

        {conf_text}

        <h3>Critical WHO Danger Markers</h3>
        <ul>{who_flags_html}</ul>

        {cf_html}

        <h3>Recorded Clinical Symptom Profile</h3>
        <table>
            <thead>
                <tr>
                    <th>Clinical Predictor</th>
                    <th style="text-align:center;width:120px;">Recorded Finding</th>
                </tr>
            </thead>
            <tbody>
                {symptoms_rows}
            </tbody>
        </table>

        <div style="margin-top:20px;padding:12px;background:#eef2ff;border-radius:6px;font-size:13px;">
            <strong>Physician Notes & Clinical Directives:</strong><br>
            {doctor_notes}
        </div>

        <div class="signature-box">
            <div class="sig-line">Attending Medical Officer Signature</div>
            <div class="sig-line">Medical Laboratory Scientist Verification</div>
        </div>

        <div class="footer">
            Note: This decision support report was synthesized using Stacking Meta-Ensemble Machine Learning and Inductive Conformal Prediction. It is intended to assist registered medical clinicians and does not replace comprehensive laboratory differential diagnosis.
        </div>
    </body>
    </html>
    """
    return html_template
