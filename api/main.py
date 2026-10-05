"""
========================================================================================
CLINICAL DECISION SUPPORT REST API (FastAPI)
Production Edge & Hospital EHR Integration Endpoint
========================================================================================
Endpoints:
- POST /v1/triage/predict: Evaluates symptoms, WHO danger score, conformal uncertainty
- POST /v1/triage/counterfactual: DiCE actionable minimal clinical interventions
- POST /v1/triage/multimodal: Fused Thin Blood Smear Cytology + Syndromic Symptoms
- POST /v1/triage/dossier: Generates printable HTML clinical report
- GET  /health: Health check and model registry status
========================================================================================
"""

import os
import sys
import io
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Ensure BASE_DIR is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.clinical_engine import compute_who_danger_score, generate_clinical_html_report
from src.counterfactuals import ClinicalCounterfactualExplainer
from src.conformal_prediction import ConformalMalariaPredictor
from src.multimodal_fusion import MultimodalMalariaClassifier

app = FastAPI(
    title="Explainable Malaria Clinical Decision Support API",
    description="Multi-Modal Vision-Tabular Fusion, Conformal Uncertainty, and DiCE Counterfactual Triage Service",
    version="2.0.0"
)

# Global State / Cache
MODELS = {}
SCALER = None
CONFORMAL_PRED = None
MULTIMODAL_ENGINE = None


@app.on_event("startup")
def load_artifacts():
    global MODELS, SCALER, CONFORMAL_PRED, MULTIMODAL_ENGINE
    
    # Load stacking ensemble
    stack_path = os.path.join(MODELS_DIR, "stacking_meta_ensemble.pkl")
    if os.path.exists(stack_path):
        MODELS["Stacking"] = joblib.load(stack_path)
    
    rf_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    if os.path.exists(rf_path):
        MODELS["RandomForest"] = joblib.load(rf_path)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)

    conf_path = os.path.join(MODELS_DIR, "conformal_predictor.pkl")
    if os.path.exists(conf_path):
        CONFORMAL_PRED = joblib.load(conf_path)

    MULTIMODAL_ENGINE = MultimodalMalariaClassifier()
    print("FastAPI Clinical CDSS Models loaded successfully.")


class PatientSymptomPayload(BaseModel):
    age: int = Field(35, ge=1, le=110, description="Patient age in years")
    fever: int = Field(1, ge=0, le=1)
    cold: int = Field(0, ge=0, le=1)
    rigor: int = Field(1, ge=0, le=1)
    fatigue: int = Field(1, ge=0, le=1)
    headace: int = Field(1, ge=0, le=1)
    bitter_tongue: int = Field(0, ge=0, le=1)
    vomitting: int = Field(0, ge=0, le=1)
    diarrhea: int = Field(0, ge=0, le=1)
    Convulsion: int = Field(0, ge=0, le=1)
    Anemia: int = Field(0, ge=0, le=1)
    jundice: int = Field(0, ge=0, le=1)
    cocacola_urine: int = Field(0, ge=0, le=1)
    hypoglycemia: int = Field(0, ge=0, le=1)
    prostraction: int = Field(0, ge=0, le=1)
    hyperpyrexia: int = Field(0, ge=0, le=1)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_available": list(MODELS.keys()),
        "conformal_predictor_active": CONFORMAL_PRED is not None,
        "multimodal_engine_active": MULTIMODAL_ENGINE is not None
    }


@app.post("/v1/triage/predict")
def predict_patient_triage(payload: PatientSymptomPayload):
    patient_dict = payload.dict()
    df_patient = pd.DataFrame([patient_dict])[FEATURE_COLUMNS]

    model = MODELS.get("Stacking", MODELS.get("RandomForest"))
    if model is None:
        raise HTTPException(status_code=500, detail="Inference models not initialized.")

    prob_severe = float(model.predict_proba(df_patient)[0, 1])
    is_severe = bool(prob_severe >= 0.40)

    # WHO Danger Evaluation
    who_res = compute_who_danger_score(patient_dict)

    # Conformal Prediction Safety Bound
    conformal_info = None
    if CONFORMAL_PRED is not None:
        conformal_info = CONFORMAL_PRED.predict_conformal_set(df_patient, alpha=0.05)[0]

    return {
        "patient_age": payload.age,
        "predicted_class": "Severe Malaria" if is_severe else "Uncomplicated / No Malaria",
        "severe_probability": round(prob_severe, 4),
        "who_danger_assessment": {
            "score": who_res["score"],
            "max_score": who_res["max_score"],
            "triage_tier": who_res["tier_label"],
            "recommended_action": who_res["recommended_action"]
        },
        "conformal_safety_guarantee": conformal_info
    }


@app.post("/v1/triage/counterfactual")
def get_counterfactual_interventions(payload: PatientSymptomPayload):
    patient_dict = payload.dict()
    df_patient = pd.DataFrame([patient_dict])[FEATURE_COLUMNS]

    model = MODELS.get("Stacking", MODELS.get("RandomForest"))
    if model is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    explainer = ClinicalCounterfactualExplainer(model, df_patient)
    cf_df = explainer.generate_counterfactuals(df_patient.iloc[0], total_CFs=2, desired_class=0)
    deltas = explainer.compute_intervention_deltas(df_patient.iloc[0], cf_df)

    return {
        "original_prediction": "Severe" if model.predict(df_patient)[0] == 1 else "Non-Severe",
        "actionable_pathways": deltas,
        "counterfactual_records": cf_df.to_dict(orient="records")
    }


@app.post("/v1/triage/multimodal")
async def predict_multimodal_triage(
    file: UploadFile = File(...),
    age: int = Form(35),
    fever: int = Form(1),
    cold: int = Form(0),
    rigor: int = Form(1),
    fatigue: int = Form(1),
    headace: int = Form(1),
    bitter_tongue: int = Form(0),
    vomitting: int = Form(0),
    diarrhea: int = Form(0),
    Convulsion: int = Form(0),
    Anemia: int = Form(0),
    jundice: int = Form(0),
    cocacola_urine: int = Form(0),
    hypoglycemia: int = Form(0),
    prostraction: int = Form(0),
    hyperpyrexia: int = Form(0)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    patient_dict = {
        "age": age, "fever": fever, "cold": cold, "rigor": rigor, "fatigue": fatigue,
        "headace": headace, "bitter_tongue": bitter_tongue, "vomitting": vomitting,
        "diarrhea": diarrhea, "Convulsion": Convulsion, "Anemia": Anemia, "jundice": jundice,
        "cocacola_urine": cocacola_urine, "hypoglycemia": hypoglycemia,
        "prostraction": prostraction, "hyperpyrexia": hyperpyrexia
    }
    df_patient = pd.DataFrame([patient_dict])[FEATURE_COLUMNS]

    model = MODELS.get("Stacking", MODELS.get("RandomForest"))
    base_tab_prob = float(model.predict_proba(df_patient)[0, 1]) if model else 0.5

    result = MULTIMODAL_ENGINE.predict_multimodal(image, df_patient.iloc[0], base_tabular_prob=base_tab_prob)
    return result


@app.post("/v1/triage/dossier", response_class=HTMLResponse)
def get_patient_dossier_html(payload: PatientSymptomPayload, doctor_notes: str = "Automated Clinical Review"):
    patient_dict = payload.dict()
    df_patient = pd.DataFrame([patient_dict])[FEATURE_COLUMNS]
    model = MODELS.get("Stacking", MODELS.get("RandomForest"))
    prob = float(model.predict_proba(df_patient)[0, 1]) if model else 0.5
    
    html = generate_clinical_html_report(
        patient_data=patient_dict,
        model_prediction={"prediction": int(prob >= 0.40), "probability": prob},
        doctor_notes=doctor_notes
    )
    return HTMLResponse(content=html, status_code=200)
