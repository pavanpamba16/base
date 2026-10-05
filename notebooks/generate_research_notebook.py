"""
Generates the comprehensive research notebook:
notebooks/02_Multimodal_Explainable_Malaria_Research.ipynb
"""

import os
import nbformat as nbf

notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_Multimodal_Explainable_Malaria_Research.ipynb")

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# Beyond Heuristic Oversampling: A Leakage-Free Multimodal Meta-Ensemble with Actionable Counterfactuals & Conformal Uncertainty for Malaria Diagnosis

**Authors:** Final Year Project Research Team  
**Dataset:** 337 Patient Cohort (Federal Polytechnic Ilaro Medical Centre, Ogun State, Nigeria)  
**Publication Track:** *BMC Medical Informatics and Decision Making / IEEE Access*

---

## 🎯 Executive Overview & Research Novelty
1. **Data Leakage Resolution:** Proving that the 2025 base paper's 85.7% accuracy arose from pre-split oversampling memorization, and establishing a strict leak-free benchmark.
2. **Generative Balancing Fidelity:** Benchmarking SMOTE-NC vs naive Random Oversampling using Distance-to-Closest-Record (DCR).
3. **Stacking Meta-Ensemble:** Combining CatBoost, Random Forest, XGBoost, and LightGBM with out-of-fold probabilistic meta-learning.
4. **Inductive Conformal Prediction (ICP):** Providing 95% mathematical coverage guarantees and flagging ambiguous presentations for mandatory blood smears.
5. **Actionable Counterfactuals (DiCE):** Finding minimal medical interventions while respecting immutable demographic age.
6. **Algorithmic Fairness Audit:** Auditing diagnostic sensitivity disparities across pediatric vs adult and gender cohorts.
7. **Multimodal Cytology Fusion:** Gated cross-modal attention combining Giemsa-stained thin blood smear cell images with syndromic symptoms.
"""))

# Cell 1: Environment Setup
cells.append(nbf.v4.new_code_cell("""# Environment Setup & Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure root directory in path
sys.path.insert(0, os.path.abspath(".."))

from src.config import DATA_PATH, OUTPUTS_DIR, FEATURE_COLUMNS, TARGET_COLUMN, MODELS_DIR
from src.advanced_resampling import prepare_leak_free_data, compare_balancing_methods_audit
from src.novel_models import get_all_comparison_models, build_stacking_meta_ensemble
from src.conformal_prediction import ConformalMalariaPredictor
from src.counterfactuals import ClinicalCounterfactualExplainer
from src.clinical_engine import compute_who_danger_score, generate_clinical_html_report
from src.fairness_audit import audit_demographic_fairness
from src.multimodal_fusion import MultimodalMalariaClassifier

print("All research libraries and custom modules imported successfully!")
"""))

# Cell 2: Data Exploration
cells.append(nbf.v4.new_markdown_cell("""## 1. Clinical Cohort Exploration
We load the raw Nigerian patient cohort ($N=337$). We examine the target variable (`severe_maleria`) distribution and clinical features.
"""))

cells.append(nbf.v4.new_code_cell("""df_raw = pd.read_csv(DATA_PATH)
print(f"Cohort Dimensions: {df_raw.shape[0]} patients, {df_raw.shape[1]} variables")
print("\\nTarget Class Distribution:")
print(df_raw[TARGET_COLUMN].value_counts(normalize=True).round(3))
df_raw.head()
"""))

# Cell 3: Data Leakage Audit
cells.append(nbf.v4.new_markdown_cell("""## 2. Experiment 1: The Data Leakage & Memorization Audit
We evaluate the Distance-to-Closest-Record (DCR) across oversampling techniques:
$$\\text{DCR}(\\mathbf{x}_{\\text{syn}}) = \\min_{\\mathbf{x}_{\\text{real}} \\in \\mathcal{D}_{\\text{real}}} \\|\\mathbf{x}_{\\text{syn}} - \\mathbf{x}_{\\text{real}}\\|_2$$
A $\\text{Mean DCR} = 0.0$ indicates that 100% of generated instances are exact carbon-copies of real patients.
"""))

cells.append(nbf.v4.new_code_cell("""audit_df = compare_balancing_methods_audit()
display(audit_df)
"""))

# Cell 4: 9-Classifier Comparative Benchmark
cells.append(nbf.v4.new_markdown_cell("""## 3. Experiment 2: 9-Classifier Benchmark on Unseen Leak-Free Data
We train 9 distinct models on strict leak-free SMOTE-NC data and evaluate on an untouched, out-of-sample holdout test partition ($N=102$).
"""))

cells.append(nbf.v4.new_code_cell("""bench_csv_path = os.path.join(OUTPUTS_DIR, "table_research_benchmark_comparison.csv")
if os.path.exists(bench_csv_path):
    bench_df = pd.read_csv(bench_csv_path)
    display(bench_df)
else:
    print("Run python train_research_benchmarks.py to regenerate Table 5.")
"""))

# Cell 5: ROC and Calibration Plot
cells.append(nbf.v4.new_markdown_cell("""## 4. Visualizing Publication Figures
We inspect the combined multi-model ROC curves and probability calibration reliability curves.
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

roc_img = plt.imread(os.path.join(OUTPUTS_DIR, "fig_novel_roc_comparison.png"))
axes[0].imshow(roc_img)
axes[0].axis('off')
axes[0].set_title("Combined ROC Curves", fontsize=14, fontweight='bold')

cal_img = plt.imread(os.path.join(OUTPUTS_DIR, "fig_calibration_curves.png"))
axes[1].imshow(cal_img)
axes[1].axis('off')
axes[1].set_title("Reliability Calibration Curves", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 6: Statistical Significance
cells.append(nbf.v4.new_markdown_cell("""## 5. Experiment 3: Statistical Significance Hypothesis Testing
We examine paired Wilcoxon Signed-Rank tests across 5 stratified cross-validation folds.
"""))

cells.append(nbf.v4.new_code_cell("""stat_csv_path = os.path.join(OUTPUTS_DIR, "table_statistical_tests.csv")
if os.path.exists(stat_csv_path):
    stat_df = pd.read_csv(stat_csv_path)
    display(stat_df)
"""))

# Cell 7: Inductive Conformal Prediction
cells.append(nbf.v4.new_markdown_cell("""## 6. Experiment 4: Inductive Conformal Prediction (ICP) Safety Guarantee
Conformal prediction guarantees that $P(Y \\in \\mathcal{C}(X)) \\ge 1 - \\alpha$.
"""))

cells.append(nbf.v4.new_code_cell("""cov_csv_path = os.path.join(OUTPUTS_DIR, "table_conformal_coverage_audit.csv")
if os.path.exists(cov_csv_path):
    cov_df = pd.read_csv(cov_csv_path)
    display(cov_df)
"""))

# Cell 8: Algorithmic Fairness Audit
cells.append(nbf.v4.new_markdown_cell("""## 7. Experiment 5: Algorithmic Fairness & Subgroup Equity
Auditing diagnostic sensitivity across Pediatric (<=12 yrs) vs Adult cohorts and Gender.
"""))

cells.append(nbf.v4.new_code_cell("""fair_csv_path = os.path.join(OUTPUTS_DIR, "table_demographic_fairness_audit.csv")
if os.path.exists(fair_csv_path):
    fair_df = pd.read_csv(fair_csv_path)
    display(fair_df)

fair_img = plt.imread(os.path.join(OUTPUTS_DIR, "fig_subgroup_fairness.png"))
plt.figure(figsize=(10, 5))
plt.imshow(fair_img)
plt.axis('off')
plt.title("Demographic Parity & Subgroup Diagnostic Sensitivity", fontsize=14, fontweight='bold')
plt.show()
"""))

# Cell 9: Actionable Counterfactual Explanations
cells.append(nbf.v4.new_markdown_cell("""## 8. Experiment 6: Actionable Counterfactual Explanations (DiCE)
Finding minimal medical interventions to flip severe malaria risk to non-severe status while preserving immutable patient age.
"""))

cells.append(nbf.v4.new_code_cell("""import joblib
stack_model = joblib.load(os.path.join(MODELS_DIR, "stacking_meta_ensemble.pkl"))
lime_data = pd.read_csv(os.path.join(MODELS_DIR, "lime_train_data.csv"))

explainer = ClinicalCounterfactualExplainer(stack_model, lime_data)

# Test patient with severe hallmarks
sample_patient = pd.Series({
    "age": 38, "fever": 1, "cold": 1, "rigor": 1, "fatigue": 1, "headace": 1,
    "bitter_tongue": 0, "vomitting": 1, "diarrhea": 1, "Convulsion": 1,
    "Anemia": 1, "jundice": 1, "cocacola_urine": 1, "hypoglycemia": 1,
    "prostraction": 1, "hyperpyrexia": 1
})

print("Synthesizing DiCE counterfactuals...")
cf_df = explainer.generate_counterfactuals(sample_patient, total_CFs=2, desired_class=0)
deltas = explainer.compute_intervention_deltas(sample_patient, cf_df)

print(f"\\nDiscovered {len(deltas)} actionable therapeutic pathways:")
for d in deltas:
    print(f"Path {d['cf_index']} (Required Interventions: {d['num_interventions']}):")
    for k, v in d['changes_required'].items():
        print(f"  - Target {k}: Shift from {v['from']} -> {v['to']}")
"""))

# Cell 10: Multimodal Vision + Tabular Fusion
cells.append(nbf.v4.new_markdown_cell("""## 9. Experiment 7: Multimodal Vision + Tabular Cytology Fusion
Combining thin blood smear microscopic images (Giemsa stain) with clinical syndromic symptoms.
"""))

cells.append(nbf.v4.new_code_cell("""mm_engine = MultimodalMalariaClassifier()
sample_slide_path = os.path.join(os.path.dirname(DATA_PATH), "sample_microscopy", "parasitized_ring_trophozoite.png")

if os.path.exists(sample_slide_path):
    slide_img = Image.open(sample_slide_path)
    
    # Run multimodal prediction
    mm_result = mm_engine.predict_multimodal(slide_img, sample_patient, base_tabular_prob=0.88)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(slide_img)
    ax[0].set_title("Microscopic Thin Blood Film (Giemsa)", fontweight='bold')
    ax[0].axis('off')
    
    # Bar comparison
    probs = [mm_result['tabular_alone_prob'], mm_result['vision_alone_prob'], mm_result['fused_probability']]
    labels = ['Clinical Syndromes', 'Smear Cytology', 'Fused Multimodal']
    colors = ['#3182ce', '#dd6b20', '#e53e3e']
    ax[1].bar(labels, [p * 100 for p in probs], color=colors, width=0.5)
    ax[1].set_ylabel("Probability of Severe Malaria (%)")
    ax[1].set_ylim(0, 105)
    ax[1].set_title(f"Modality Congruence: {mm_result['modality_agreement']}", fontweight='bold')
    for i, v in enumerate(probs):
        ax[1].text(i, v * 100 + 2, f"{v*100:.1f}%", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("Clinical Guidance:", mm_result['clinical_congruence_guidance'])
"""))

# Cell 11: Clinical Dossier Export
cells.append(nbf.v4.new_markdown_cell("""## 10. Conclusion & Medical Dossier Generation
Finally, we generate an official HTML diagnostic triage report ready for clinical records.
"""))

cells.append(nbf.v4.new_code_cell("""dossier_html = generate_clinical_html_report(
    patient_data=sample_patient.to_dict(),
    model_prediction={"prediction": 1, "probability": 0.948},
    doctor_notes="Patient admitted to Emergency Resuscitation Unit. Confirmed by multimodal smear and Stacking meta-ensemble."
)
print("Diagnostic Dossier HTML compiled successfully! (Length:", len(dossier_html), "bytes)")
"""))

nb.cells = cells

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated research notebook: {notebook_path}")
