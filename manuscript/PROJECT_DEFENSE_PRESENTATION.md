# Final Year Project Defense & Oral Viva Presentation

**Project Title:** Beyond Heuristic Oversampling: A Leakage-Free Multimodal Meta-Ensemble with Actionable Counterfactual Explanations and Conformal Uncertainty for Interpretable Malaria Diagnosis  
**Student Name:** [Your Name / Candidate Name]  
**Degree / Program:** Bachelor of Technology / Science in Computer Science & Engineering  
**Department:** Department of Computer Science & Engineering, [University Name]  
**Supervisor / Guide:** [Advisor Name], [Faculty Title]  
**Academic Year:** 2025 – 2026  

---

## 📋 Defense Slide Outline & Verbatim Speaker Script

```text
SLIDE 1: Title & Introduction
SLIDE 2: Clinical Context & The Malaria Diagnostic Dilemma
SLIDE 3: Critical Audit of Prior Literature: The Data Leakage Flaw
SLIDE 4: Synthetic Generative Balancing & Privacy Audit (DCR)
SLIDE 5: Unified Multimodal & Meta-Ensemble System Architecture
SLIDE 6: Two-Stage Stacking Meta-Ensemble Design
SLIDE 7: 9-Classifier Benchmark on Unseen Clinical Data
SLIDE 8: Statistical Significance Hypothesis Testing (Wilcoxon)
SLIDE 9: Mathematical Safety via Inductive Conformal Prediction (ICP)
SLIDE 10: Prescriptive Explainability: Actionable Counterfactuals (DiCE)
SLIDE 11: Algorithmic Fairness & Subgroup Disparity Audit (Pediatric Cohort)
SLIDE 12: Multimodal Vision + Tabular Cytology Fusion
SLIDE 13: Clinical Web Workstation & Production REST API
SLIDE 14: Rigorous Engineering & Test Suite Validation
SLIDE 15: Publications, Societal Impact & Conclusion
```

---

### Slide 1: Title & Introduction
- **Visuals:** Project Title, University Crest, Candidate Details, Target Publication Banner (*BMC Medical Informatics / IEEE Access*).
- **Key Talking Points:**
  - Good morning, respected Chairman, external examiners, and members of the evaluation committee.
  - Today I present our research project which addresses critical flaws in published clinical machine learning models for malaria triage and advances them into a verified multimodal clinical decision support framework.

> **Speaker Script:**
> *"Malaria remains one of the world's most catastrophic infectious killers, claiming over 600,000 lives annually, primarily in sub-Saharan Africa. While recent publications have proposed AI for syndromic malaria screening, our investigation revealed a fundamental methodological vulnerability: pre-split data leakage, which produces artificially optimistic metrics. In this project, we not only resolve this leakage, but introduce a comprehensive clinical framework featuring Stacking Meta-Ensembles, Inductive Conformal Prediction, Actionable Counterfactuals, and Multimodal Vision-Tabular Fusion."*

---

### Slide 2: Clinical Context & The Diagnostic Bottleneck
- **Visuals:** Comparison of Optical Microscopy vs Rapid Diagnostic Tests (RDTs) vs Clinical Syndromes.
- **Key Talking Points:**
  - Gold standard microscopy is hindered by equipment shortages, power intermittency, and technician fatigue in rural centers.
  - RDTs suffer from false negatives during low-parasitemia and cannot quantify organ failure or severity.
  - Syndromic triage using machine learning provides instant pre-laboratory risk stratification.

> **Speaker Script:**
> *"In peripheral primary health clinics, doctors and nurses must decide within minutes whether a febrile patient requires routine oral treatment or immediate parenteral admission for life-threatening severe malaria. Optical microscopy is frequently unavailable after hours, while RDTs fail to measure clinical organ compromise. A trusted, transparent, and mathematically safe AI decision support system is desperately needed."*

---

### Slide 3: Critical Audit of the Base Paper: The Data Leakage Discovery
- **Visuals:** Diagram showing Pre-Split Oversampling vs Strict Post-Split Partitioning.
- **Key Talking Points:**
  - *Base Paper:* Awe et al., *BMC Medical Informatics and Decision Making* (2025).
  - *Methodological Flaw:* The minority class ($N_1 = 116$) was duplicated across the whole cohort *before* the 70/30 train-test split.
  - *Consequence:* 105 synthetic duplicates were distributed across both training and test partitions. Models were merely looking up memorized twins, falsely reporting 85.7% accuracy.

> **Speaker Script:**
> *"When we audited the raw source code of the 2025 BMC reference paper, we discovered a classic machine learning pitfall: pre-split oversampling. By resampling the minority class to 221 samples across the entire dataset before creating the test split, identical duplicate patient records existed in both train and test partitions. The reported 85.7% accuracy was inflated by memorization. Our first major research contribution is isolating this vulnerability and establishing a mathematically strict, leak-free benchmark."*

---

### Slide 4: Synthetic Generative Balancing & Privacy Audit (DCR)
- **Visuals:** Distance-to-Closest-Record (DCR) Comparison Table (ROS vs SMOTE-NC).
- **Key Talking Points:**
  - We introduced the Euclidean **Distance-to-Closest-Record (DCR)** metric to measure memorization risk.
  - **Random Over-Sampling (ROS):** $\text{Mean DCR} = 0.0000$ (100% exact duplication).
  - **SMOTE-NC (Proposed):** $\text{Mean DCR} = 1.4491$, $\text{MACD} = 0.1247$ (generates novel, realistic clinical profiles while respecting discrete 0/1 symptom states).

> **Speaker Script:**
> *"To prove memorization empirically, we audited the Distance-to-Closest-Record. As shown on the slide, Random Over-Sampling yields a Mean DCR of exactly zero—proving complete duplication. To solve this, we implemented SMOTE-NC, which interpolates in hybrid Euclidean-Hamming space. It achieves a healthy DCR of 1.45, generating genuinely novel synthetic patient presentations without memorization."*

---

### Slide 5: System Architecture Overview
- **Visuals:** Complete pipeline diagram from patient inputs to Conformal Safety, WHO Scoring, Counterfactuals, and Multimodal Fusion.
- **Key Talking Points:**
  - Modular, end-to-end clinical workflow.
  - Dual-branch: Microscopic Blood Smear Vision + Clinical Questionnaire Tabular.
  - Safety layers: WHO Clinical Danger Engine and Conformal Prediction Sets.

---

### Slide 6: Two-Stage Stacking Meta-Ensemble Design
- **Visuals:** Level-0 Base Learners (CatBoost, RF, XGBoost, LightGBM) feeding into Level-1 Logistic Meta-Learner.
- **Key Talking Points:**
  - Level-0 models are trained across distinct algorithmic paradigms (bagging, oblivious trees, hessian boosting, leaf-wise boosting).
  - 5-Fold out-of-fold probabilistic margins are fed into an $L_2$-regularized meta-classifier.

> **Speaker Script:**
> *"Rather than relying on a single tree model, our architecture uses a Two-Stage Stacking Meta-Ensemble. By training four diverse base learners—Random Forest, CatBoost, XGBoost, and LightGBM—and feeding their 5-fold cross-validated probability outputs into an L2-regularized meta-learner, we combine their complementary inductive biases and improve boundary stability."*

---

### Slide 7: 9-Classifier Comparative Benchmark on Unseen Data
- **Visuals:** Table 5 (Full 9-Classifier Benchmark: Accuracy, ROC-AUC, Balanced Acc, Precision, Recall, F1, MCC).
- **Key Talking Points:**
  - Evaluated on 102 uncompromised, unseen test patients.
  - True real-world performance on pure symptom questionnaires is revealed (~55–60% accuracy on raw questionnaire alone).
  - Highlights why symptom questionnaires alone have clinical uncertainty, proving the vital need for conformal intervals and microscopic fusion.

---

### Slide 8: Statistical Hypothesis Testing (Wilcoxon Signed-Rank)
- **Visuals:** Table 6 (Comparison against Random Forest, CatBoost, XGBoost; Wilcoxon W, p-values).
- **Key Talking Points:**
  - Conducted paired Wilcoxon Signed-Rank tests across stratified cross-validation folds.
  - The Stacking Meta-Ensemble achieved the highest cross-validated mean score ($0.8154$), confirming superior generalization stability over single tree models.

---

### Slide 9: Mathematical Safety via Inductive Conformal Prediction
- **Visuals:** Prediction set taxonomy: $\{0\}$, $\{1\}$, $\{0, 1\}$. Conformal coverage audit table.
- **Key Talking Points:**
  - High-stakes medicine cannot tolerate overconfident false negatives.
  - Conformal prediction guarantees $P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$.
  - At $95\%$ confidence, our model achieved **$98.04\%$ empirical coverage**.
  - Ambiguous cases ($\{0, 1\}$) are mathematically flagged for mandatory urgent microscopic examination.

> **Speaker Script:**
> *"In clinical medicine, an overconfident wrong prediction can be lethal. We implemented Inductive Conformal Prediction to provide mathematical safety guarantees. At a 95% nominal confidence level, our system demonstrated 98% empirical coverage. If the system is uncertain, it does not guess; it outputs a prediction set containing both classes, which triggers an automated directive for mandatory laboratory blood film examination."*

---

### Slide 10: Prescriptive Explainability: Actionable Counterfactuals (DiCE)
- **Visuals:** Counterfactual Intervention Pathways diagram showing target clinical reversals.
- **Key Talking Points:**
  - Standard SHAP and LIME only tell a doctor *why* an alert happened.
  - DiCE provides **actionable clinical prescriptions**: *What minimal interventions reverse high risk?*
  - Enforces domain constraints: Patient age is **strictly immutable**. Mutable targets are clinical signs (`hypoglycemia`, `hyperpyrexia`, `convulsion`).

> **Speaker Script:**
> *"Traditional XAI methods like SHAP are descriptive—they tell you why a patient was classified as severe, but offer no treatment strategy. We implemented DiCE to provide prescriptive counterfactuals. By locking immutable biological features like patient age, the system determines the exact minimal clinical interventions—such as resolving hypoglycemia via IV dextrose or controlling convulsions with anticonvulsants—that will successfully transition the patient to low-risk status."*

---

### Slide 11: Algorithmic Fairness & Subgroup Equity Audit
- **Visuals:** Subgroup Performance Chart (Pediatric vs Adult, Male vs Female).
- **Key Talking Points:**
  - Audited across Pediatric ($\le 12$ yrs), Young Adult, Older Adult, and Gender cohorts.
  - Gender Equal Opportunity Difference is $0.1422$ (satisfies $< 0.15$ parity tolerance).
  - *Critical Discovery:* Pure ML models exhibit a 40% False Negative Rate in young children due to low cohort representation. Our **WHO Clinical Danger Rule Engine** acts as an essential clinical failsafe.

---

### Slide 12: Multimodal Cytology Fusion (Vision + Tabular)
- **Visuals:** Microscope thin blood smear slide image with signet-ring trophozoite alongside cross-modal attention diagram.
- **Key Talking Points:**
  - PyTorch `CytologyVisionBranch` processes $224 \times 224$ Giemsa-stained thin blood film cell images.
  - Detects parasite chromatin dots and ring cytoplasm, estimating Cytological Parasitemia Index.
  - Cross-attention fusion produces a unified diagnosis and evaluates **Modality Congruence** (flagging discrepancies between slide and symptoms).

---

### Slide 13: Clinical Web Workstation & Production REST API
- **Visuals:** Screenshots of the 8-tab Streamlit workstation and FastAPI Swagger documentation.
- **Key Talking Points:**
  - Complete clinician interface with risk gauges, counterfactual simulator, and downloadable PDF/HTML Medical Dossiers.
  - Production FastAPI backend (`api/main.py`) ready for hospital EHR integration.

---

### Slide 14: Rigorous Engineering & Test Suite Validation
- **Visuals:** Terminal output showing `Ran 8 tests in 5.739s - OK`.
- **Key Talking Points:**
  - 100% automated test pass rate across unit and integration tests.
  - Tests verify leak-free resampling, DCR non-memorization, conformal coverage bounds, counterfactual age immutability, and multimodal fusion.

---

### Slide 15: Publications, Societal Impact & Conclusion
- **Visuals:** Manuscript preview ([manuscript/PAPER_DRAFT.md](manuscript/PAPER_DRAFT.md)), Target Journals, Project Summary.
- **Key Talking Points:**
  - Complete paper draft ready for submission to *BMC Medical Informatics and Decision Making* or *IEEE Access*.
  - Solves a life-critical healthcare problem in low-resource endemic regions.
  - Thank you to the evaluation committee. Open for questions.

> **Closing Script:**
> *"In conclusion, this project transforms a simple questionnaire reproduction into a scientifically sound, clinically actionable, and multimodal decision support ecosystem. We identified and corrected publication data leakage, introduced Stacking Meta-Ensembles with Conformal Uncertainty, audited demographic fairness, and built an open-access clinical tool. Thank you, and I look forward to your questions."*
