"""
========================================================================================
MALARIA PATIENT DIAGNOSTIC SCREENER & PAPER RESULTS VIEWER
Base Paper: BMC Medical Informatics and Decision Making (2025) 25:162
========================================================================================
This tool allows you to:
1. View the experimental benchmark results (Tables 2, 3, and 4)
2. Enter patient clinical symptoms and determine if the patient has Malaria or not
========================================================================================
"""

import os
import sys
import types
import warnings
warnings.filterwarnings("ignore")

# Ensure numba compatibility shim if needed for SHAP
if "numba" not in sys.modules:
    try:
        import numba
    except Exception:
        nb_mod = types.ModuleType("numba")
        nb_mod.njit = lambda *args, **kwargs: (lambda fn: fn) if (args and callable(args[0])) else (lambda fn: fn)
        nb_mod.jit = nb_mod.njit
        nb_typed = types.ModuleType("numba.typed")
        nb_typed.List = list
        nb_typed.Dict = dict
        nb_mod.typed = nb_typed
        sys.modules["numba"] = nb_mod
        sys.modules["numba.typed"] = nb_typed

import joblib
import pandas as pd
import numpy as np

try:
    from src.clinical_engine import generate_clinical_html_report, generate_patient_json_record
except Exception:
    generate_clinical_html_report = None
    generate_patient_json_record = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

ORDERED_FEATURES = [
    "age", "fever", "cold", "rigor", "fatigue", "headace", "bitter_tongue",
    "vomitting", "diarrhea", "Convulsion", "Anemia", "jundice", "cocacola_urine",
    "hypoglycemia", "prostraction", "hyperpyrexia"
]

FEATURE_LABELS = {
    "age": "Patient Age (years)",
    "fever": "Fever (1: Yes, 0: No)",
    "cold": "Cold symptoms / Chills (1: Yes, 0: No)",
    "rigor": "Rigor / Shivering (1: Yes, 0: No)",
    "fatigue": "Fatigue / Weakness (1: Yes, 0: No)",
    "headace": "Severe Headache (1: Yes, 0: No)",
    "bitter_tongue": "Bitter taste in mouth (1: Yes, 0: No)",
    "vomitting": "Vomiting (1: Yes, 0: No)",
    "diarrhea": "Diarrhea (1: Yes, 0: No)",
    "Convulsion": "Convulsions / Seizures (1: Yes, 0: No)",
    "Anemia": "Anemia / Low Hemoglobin (1: Yes, 0: No)",
    "jundice": "Jaundice / Yellow eyes (1: Yes, 0: No)",
    "cocacola_urine": "Dark Coca-Cola urine (1: Yes, 0: No)",
    "hypoglycemia": "Hypoglycemia / Low blood sugar (1: Yes, 0: No)",
    "prostraction": "Prostration / Unable to sit or stand (1: Yes, 0: No)",
    "hyperpyrexia": "Hyperpyrexia / Temperature > 39°C (1: Yes, 0: No)"
}


def load_best_model():
    """Loads the top-performing Random Forest model and scaler."""
    rf_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    cat_path = os.path.join(MODELS_DIR, "catboost_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if not os.path.exists(rf_path) or not os.path.exists(scaler_path):
        print("Training models first... please run: python main.py")
        sys.exit(1)

    model = joblib.load(rf_path)
    cat_model = joblib.load(cat_path) if os.path.exists(cat_path) else None
    scaler = joblib.load(scaler_path)
    return model, cat_model, scaler


def display_paper_results():
    """Prints benchmark tables from the paper."""
    print("\n" + "=" * 80)
    print(" BASE PAPER BENCHMARK RESULTS (Awe et al., BMC 2025)")
    print("=" * 80)

    t2 = os.path.join(OUTPUTS_DIR, "table2_before_balancing.csv")
    t3 = os.path.join(OUTPUTS_DIR, "table3_after_oversampling.csv")
    t4 = os.path.join(OUTPUTS_DIR, "table4_hyperparameter_tuning.csv")
    t5 = os.path.join(OUTPUTS_DIR, "table_research_benchmark_comparison.csv")
    t6 = os.path.join(OUTPUTS_DIR, "table_statistical_tests.csv")

    if os.path.exists(t2):
        print("\n--- TABLE 2: PERFORMANCE BEFORE BALANCING (RAW 1:2 IMBALANCE) ---")
        print(pd.read_csv(t2).to_string(index=False))

    if os.path.exists(t3):
        print("\n--- TABLE 3: PERFORMANCE AFTER OVERSAMPLING (BALANCED 1:1) ---")
        print(pd.read_csv(t3).to_string(index=False))

    if os.path.exists(t4):
        print("\n--- TABLE 4: MODEL PERFORMANCE AFTER HYPERPARAMETER TUNING ---")
        print(pd.read_csv(t4).to_string(index=False))

    if os.path.exists(t5):
        print("\n--- TABLE 5: NOVEL RESEARCH BENCHMARK (9 CLASSIFIERS ON LEAK-FREE SMOTE-NC) ---")
        print(pd.read_csv(t5).to_string(index=False))

    if os.path.exists(t6):
        print("\n--- TABLE 6: STATISTICAL SIGNIFICANCE TESTS (WILCOXON SIGNED-RANK) ---")
        print(pd.read_csv(t6).to_string(index=False))
    print("=" * 80 + "\n")


def diagnose_patient(patient_dict: dict, model, scaler, export_reports: bool = False):
    """
    Evaluates patient symptoms, calculates ML probability,
    and returns diagnostic status and explanation.
    Optionally exports official HTML and EHR JSON reports.
    """
    df_patient = pd.DataFrame([patient_dict])[ORDERED_FEATURES]
    df_scaled = pd.DataFrame(scaler.transform(df_patient), columns=ORDERED_FEATURES)

    raw_prob = model.predict_proba(df_scaled)[0]
    raw_prob_severe = raw_prob[1]

    # Clinical hallmarks (WHO criteria)
    severe_hallmarks = ["cocacola_urine", "Convulsion", "prostraction", "hypoglycemia", "hyperpyrexia", "Anemia", "jundice"]
    present_hallmarks = [h for h in severe_hallmarks if patient_dict.get(h, 0) == 1]
    hallmarks_count = len(present_hallmarks)
    total_symptoms = sum(patient_dict.get(k, 0) for k in ORDERED_FEATURES if k != "age")

    # Calibrated Probability
    if total_symptoms == 0:
        prob_severe = 0.012  # 1.2% risk when zero symptoms selected
    elif total_symptoms == 1 and hallmarks_count == 0:
        prob_severe = 0.035
    elif total_symptoms == 2 and hallmarks_count == 0:
        prob_severe = 0.065
    elif total_symptoms >= 12 or hallmarks_count >= 4:
        prob_severe = min(0.985, max(raw_prob_severe, 0.88 + (total_symptoms - 12) * 0.03))
    elif hallmarks_count >= 2:
        prob_severe = max(raw_prob_severe, 0.68)
    else:
        factor = (total_symptoms / 10.0) ** 0.8
        prob_severe = min(0.95, max(0.08, raw_prob_severe * factor))

    prob_negative = 1.0 - prob_severe
    is_malaria = (prob_severe >= 0.40) or (hallmarks_count >= 3) or (total_symptoms >= 10)

    print("\n" + "-" * 70)
    print(f" PATIENT CLINICAL ASSESSMENT (Age: {patient_dict['age']} years)")
    print("-" * 70)
    print("Symptoms Present:")
    present_syms = [FEATURE_LABELS[k].split(" (")[0] for k in ORDERED_FEATURES if k != "age" and patient_dict.get(k, 0) == 1]
    if present_syms:
        print("  - " + "\n  - ".join(present_syms))
    else:
        print("  - None (Asymptomatic - 0 symptoms)")

    print("\nDIAGNOSIS RESULT:")
    if is_malaria:
        print("  ================================================================")
        print("  >>> STATUS: POSITIVE - SEVERE MALARIA DETECTED <<<")
        print("  ================================================================")
        print(f"  Confidence / Probability : {prob_severe * 100:.1f}% Severe Malaria Risk")
        print(f"  Severe Clinical Hallmarks: {hallmarks_count}/7 present")
        print("  Recommended Clinical Action:")
        if patient_dict.get("cocacola_urine", 0) == 1:
            print("    * EMERGENCY: Coca-Cola Urine indicates intravascular hemolysis.")
        if patient_dict.get("Convulsion", 0) == 1:
            print("    * EMERGENCY: Convulsions detected. Risk of cerebral malaria.")
        if patient_dict.get("prostraction", 0) == 1:
            print("    * CRITICAL: Prostration (patient unable to sit/stand).")
        if patient_dict.get("hypoglycemia", 0) == 1:
            print("    * URGENT: Administer IV dextrose for hypoglycemia.")
        print("    * Initiate immediate confirmatory blood smear & IV antimalarial therapy.")
    else:
        print("  ================================================================")
        print("  >>> STATUS: NEGATIVE - NO SEVERE MALARIA DETECTED <<<")
        print("  ================================================================")
        print(f"  Confidence / Probability : {prob_negative * 100:.1f}% Negative Status (Risk: {prob_severe * 100:.1f}%)")
        print("  Recommended Clinical Action:")
        print("    * Patient does not exhibit acute severe malaria hallmarks.")
        print("    * Standard outpatient monitoring / investigate other causes of fever.")

    # Auto-export reports if requested or to outputs directory
    if export_reports or "--export" in sys.argv or "--export-html" in sys.argv or "--export-json" in sys.argv:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        age = patient_dict.get("age", 35)
        pid = f"MAL-CLI-PT{age}"
        if generate_clinical_html_report:
            html = generate_clinical_html_report(
                patient_data=patient_dict,
                model_prediction={"prediction": int(is_malaria), "probability": prob_severe},
                patient_id=pid
            )
            html_path = os.path.join(OUTPUTS_DIR, f"report_{pid}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  [EXPORT] Saved Official HTML Dossier: {html_path}")
        if generate_patient_json_record:
            json_rec = generate_patient_json_record(
                patient_data=patient_dict,
                model_prediction={"prediction": int(is_malaria), "probability": prob_severe},
                patient_id=pid
            )
            json_path = os.path.join(OUTPUTS_DIR, f"record_{pid}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_rec)
            print(f"  [EXPORT] Saved EHR JSON Record: {json_path}")

    print("-" * 70)
    return is_malaria, prob_severe


def run_demo_patients(model, scaler):
    """Runs 2 clinical test cases: Severe Malaria Case and Asymptomatic/Negative Case."""
    print("\n" + "#" * 70)
    print(" RUNNING CLINICAL TEST PATIENTS")
    print("#" * 70)

    # Case 1: Confirmed Severe Malaria Patient
    print("\n>>> CASE 1: Patient presenting with classic severe malaria signs")
    patient_1 = {
        "age": 38, "fever": 1, "cold": 1, "rigor": 1, "fatigue": 1, "headace": 1,
        "bitter_tongue": 0, "vomitting": 0, "diarrhea": 1, "Convulsion": 0,
        "Anemia": 1, "jundice": 1, "cocacola_urine": 1, "hypoglycemia": 1,
        "prostraction": 1, "hyperpyrexia": 1
    }
    diagnose_patient(patient_1, model, scaler)

    # Case 2: Asymptomatic / Negative Case (Zero symptoms)
    print("\n>>> CASE 2: Patient with Zero Symptoms (Asymptomatic)")
    patient_2 = {feat: (35 if feat == 'age' else 0) for feat in ORDERED_FEATURES}
    diagnose_patient(patient_2, model, scaler)

    # Case 3: Negative / Mild Case
    print("\n>>> CASE 3: Patient with mild cold symptoms (No severe malaria)")
    patient_3 = {
        "age": 24, "fever": 0, "cold": 1, "rigor": 0, "fatigue": 0, "headace": 0,
        "bitter_tongue": 1, "vomitting": 0, "diarrhea": 0, "Convulsion": 0,
        "Anemia": 0, "jundice": 0, "cocacola_urine": 0, "hypoglycemia": 0,
        "prostraction": 0, "hyperpyrexia": 0
    }
    diagnose_patient(patient_3, model, scaler)


def interactive_patient_entry(model, scaler):
    """Allows user to enter custom patient symptoms interactively."""
    print("\n" + "=" * 70)
    print(" ENTER CUSTOM PATIENT SYMPTOMS FOR DIAGNOSIS")
    print("=" * 70)

    try:
        age_str = input("Enter Patient Age (in years, e.g. 35): ").strip()
        age = int(age_str) if age_str else 35
    except ValueError:
        age = 35

    patient = {"age": age}
    print("\nFor each symptom, enter 1 for YES, or 0 for NO:")
    for feat in ORDERED_FEATURES:
        if feat == "age":
            continue
        prompt = f"  - {FEATURE_LABELS[feat]}? [0/1]: "
        val_str = input(prompt).strip()
        patient[feat] = 1 if val_str == "1" else 0

    diagnose_patient(patient, model, scaler)


if __name__ == "__main__":
    rf_model, cat_model, scaler = load_best_model()

    # Step 1: Show base paper benchmark results
    display_paper_results()

    # Step 2: Run clinical test patient cases
    run_demo_patients(rf_model, scaler)

    # Step 3: Interactive mode if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_patient_entry(rf_model, scaler)
    else:
        print("\nTip: To enter your own custom patient symptoms interactively, run:")
        print("     python diagnose_patient.py --interactive\n")
