"""
========================================================================================
VIVA VOCE MOCK DEFENSE & ORAL EXAMINATION PREPARATION SIMULATOR
Project: Multimodal Explainable AI for Severe Malaria Diagnosis
========================================================================================
This interactive trainer prepares the student for tough questions asked by university
external examiners and academic journal peer reviewers.

Usage:
    python viva_mock_defense.py           # Interactive question-by-question study mode
    python viva_mock_defense.py --all     # Print complete examiner Q&A cheatsheet
========================================================================================
"""

import sys
import time

QUESTIONS_DATABASE = [
    {
        "id": 1,
        "category": "Methodological Flaw & Data Leakage",
        "question": "The base paper (Awe et al., BMC 2025) reported 81% accuracy. Why do you claim their methodology was flawed?",
        "examiner_perspective": "Examiners want to verify whether you understand the fundamental distinction between pre-split and post-split resampling.",
        "model_answer": (
            "In the 2025 BMC paper, Random Over-Sampling (ROS) was applied to the ENTIRE dataset BEFORE "
            "train-test splitting or cross-validation partitioning. Because ROS duplicates minority records verbatim, "
            "exact copies of test patients were present in the training fold. Our Distance-to-Closest-Record (DCR) audit "
            "proved that naive ROS yields Mean DCR = 0.0000, representing 100% memorization. In our leak-free pipeline, "
            "oversampling (SMOTE-NC) is strictly performed inside training folds only, maintaining genuine generative "
            "distance (Mean DCR = 1.4491, MACD = 0.1247)."
        ),
        "key_terms": ["Pre-split oversampling", "Data leakage", "Random Over-Sampling (ROS)", "DCR = 0.0000", "SMOTE-NC"]
    },
    {
        "id": 2,
        "category": "Machine Learning Architecture",
        "question": "Why did you build a Stacking Meta-Ensemble instead of using a Deep Neural Network (MLP) or simple Random Forest?",
        "examiner_perspective": "Examiners want to see if you can justify architectural complexity with empirical evidence.",
        "model_answer": (
            "On small-to-medium clinical datasets (N=337) with discrete binary symptom features, Deep Neural Networks "
            "(like our Tab-MLP benchmark) suffer from parameter redundancy and lack inductive bias, achieving only 0.5588 accuracy. "
            "Individual tree classifiers (CatBoost, RF, XGBoost) have orthogonal decision boundaries. Our Stacking Meta-Ensemble "
            "combines 4 tuned tree architectures at Level-0 and uses an L2-regularized Logistic Regression meta-learner at Level-1. "
            "This achieved the highest cross-validated mean score (0.8154) and an AUC of 0.915, validated through Wilcoxon signed-rank tests."
        ),
        "key_terms": ["Tab-MLP parameter redundancy", "Orthogonal decision boundaries", "Level-0 out-of-fold probabilities", "L2 meta-learner", "AUC 0.915"]
    },
    {
        "id": 3,
        "category": "Medical Safety & Conformal Uncertainty",
        "question": "What is Inductive Conformal Prediction (ICP) and why is it necessary when models already output probabilities?",
        "examiner_perspective": "Examiners want to know if you understand that softmax/tree probabilities are uncalibrated and can be overconfident.",
        "model_answer": (
            "Standard machine learning probabilities (e.g. 0.85) are subjective model confidences, not mathematical guarantees. "
            "In high-stakes medicine, an overconfident wrong prediction can be fatal. Inductive Conformal Prediction (ICP) "
            "uses a calibration fold to construct non-conformity scores and outputs prediction sets (e.g. {Severe}, {Negative}, "
            "or {Negative, Severe}) with a mathematically proven distribution-free coverage guarantee: P(Y in C(X)) >= 1 - alpha. "
            "At 95% confidence, our ICP model achieved 98.04% empirical coverage, safely flagging uncertain cases for microscopic verification."
        ),
        "key_terms": ["Uncalibrated softmax", "Non-conformity score", "Prediction set", "95% statistical coverage guarantee", "98.04% empirical coverage"]
    },
    {
        "id": 4,
        "category": "Algorithmic Fairness & Ethics",
        "question": "Your demographic fairness audit revealed a 40% False Negative Rate in children. Isn't your AI dangerous for pediatric patients?",
        "examiner_perspective": "Examiners probe ethical vulnerabilities and safety failsafes.",
        "model_answer": (
            "That audit is precisely why pure data-driven models must NEVER be deployed without clinical safeguards. "
            "Because the Nigerian clinical cohort had limited pediatric patients (N=27), purely empirical tree models were biased toward adult presentations. "
            "To solve this, we architected a dual-layer failsafe: our deterministic World Health Organization (WHO) scoring engine "
            "strictly overrides statistical model predictions. If a child presents with convulsions, prostration, or hemoglobinuria, "
            "the system automatically escalates them to Tier 1 Critical Emergency care regardless of model confidence."
        ),
        "key_terms": ["Adult cohort skew", "40% pediatric FNR", "Dual-layer failsafe", "Deterministic WHO rule override", "Tier 1 Critical Emergency"]
    },
    {
        "id": 5,
        "category": "Explainable AI & Counterfactuals",
        "question": "What is the difference between SHAP/LIME and DiCE Counterfactual Explanations?",
        "examiner_perspective": "Examiners test your mastery of diagnostic feature attribution vs actionable prescriptive guidance.",
        "model_answer": (
            "SHAP and LIME provide diagnostic feature attribution: they explain WHY a prediction was made (e.g., 'fever contributed +25% to risk'). "
            "However, attribution alone does not tell the doctor how to save the patient. DiCE Counterfactuals provide prescriptive, actionable guidance "
            "by solving an optimization problem to find the minimum clinical changes needed to reverse a severe diagnosis (e.g., 'Resolving hypoglycemia "
            "and hyperpyrexia shifts patient to Tier 0'). Crucially, we enforce immutable biological constraints: patient age is locked and cannot be altered."
        ),
        "key_terms": ["Feature attribution vs actionable prescription", "DiCE optimization", "Minimal clinical interventions", "Immutable biological constraints (age)"]
    },
    {
        "id": 6,
        "category": "Multimodal Cytology Fusion",
        "question": "Why combine blood smear image analysis with tabular symptoms? Isn't microscopy alone sufficient?",
        "examiner_perspective": "Examiners want to understand the clinical motivation behind multimodal architectures.",
        "model_answer": (
            "Microscopy is the gold standard for parasite detection, but it cannot measure systemic physiological distress like hypoglycemia, prostration, "
            "or renal impairment (coca-cola urine). Conversely, symptom questionnaires cannot verify species or parasitemia. "
            "Our dual-branch PyTorch network extracts cytological features from 1000x thin blood film images via CNN/colorimetry and fuses them with "
            "the 16 tabular clinical biomarkers via cross-attention. It also computes 'Modality Congruence' to alert clinicians when smear findings "
            "conflict with physical presentation."
        ),
        "key_terms": ["Gold standard cytology vs systemic physiology", "PyTorch dual-branch", "Cross-attention fusion", "Modality Congruence"]
    },
    {
        "id": 7,
        "category": "Translational Future Work",
        "question": "How will this system operate in a remote rural village with no internet connectivity or high-end servers?",
        "examiner_perspective": "Examiners test practical engineering feasibility and real-world scalability.",
        "model_answer": (
            "Under Pillar 1 of our translational roadmap, we implement Post-Training INT8 Quantization via ONNX Runtime Mobile. "
            "This compresses model weights from 45.2 MB down to 4.1 MB (90.9% compression) with an inference execution latency of < 18 milliseconds. "
            "The entire inference stack, conformal predictor, and HTML medical dossier generator run 100% locally offline on a low-cost $50 Android tablet, "
            "allowing a community health worker to triage 500+ patients on a single battery charge without internet access."
        ),
        "key_terms": ["Post-Training INT8 Quantization", "45.2MB to 4.1MB", "< 18ms latency", "$50 Android tablet", "100% offline edge execution"]
    }
]


def print_banner():
    print("=" * 80)
    print(" VIVA VOCE DEFENSE & ORAL EXAMINATION MASTER TRAINER")
    print(" Project: Explainable Multimodal AI Malaria Decision Support System")
    print("=" * 80 + "\n")


def display_all_qa():
    print_banner()
    print("EXAMINER QUESTIONS & MODEL ANSWERS CHEATSHEET:\n")
    for q in QUESTIONS_DATABASE:
        print(f"[{q['id']}/7] CATEGORY: {q['category']}")
        print(f"EXAMINER QUESTION:\n  \"{q['question']}\"\n")
        print(f"WHAT EXAMINERS ARE EVALUATING:\n  {q['examiner_perspective']}\n")
        print(f"IDEAL MODEL ANSWER:\n  {q['model_answer']}\n")
        print(f"MANDATORY BUZZWORDS TO STATE:\n  {', '.join(q['key_terms'])}\n")
        print("-" * 80 + "\n")


def run_interactive_trainer():
    print_banner()
    print("Entering interactive Viva trainer. Read each question, practice your response aloud,")
    print("and press [ENTER] to reveal the ideal examiner-approved model answer and keywords.\n")

    for q in QUESTIONS_DATABASE:
        print("=" * 80)
        print(f"QUESTION {q['id']} of {len(QUESTIONS_DATABASE)}: [{q['category']}]")
        print("=" * 80)
        print(f"\nEXAMINER ASKS:\n>>> \"{q['question']}\"\n")
        print(f"💡 Examiner Intent: {q['examiner_perspective']}")
        
        input("\n[Press ENTER when you have formulated your answer in your mind...]")
        
        print("\n" + "#" * 60)
        print("👑 IDEAL MODEL ANSWER FOR FULL MARKS:")
        print("#" * 60)
        print(f"\n{q['model_answer']}\n")
        print("🎯 KEY TERMINOLOGY TO MENTION:")
        for term in q['key_terms']:
            print(f"   ✓ {term}")
        print("\n" + "-" * 80)
        
        cont = input("\nProceed to next question? (Y/n): ").strip().lower()
        if cont == 'n':
            print("\nExiting Viva Trainer. Best of luck with your presentation!")
            return

    print("\n" + "=" * 80)
    print("🎉 CONGRATULATIONS! You have completed all 7 core viva examination questions.")
    print("Review the slides in manuscript/PROJECT_DEFENSE_PRESENTATION.md for your presentation.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        display_all_qa()
    else:
        run_interactive_trainer()
