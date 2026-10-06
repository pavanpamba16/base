# Explainable Multimodal AI Decision Support System for Severe Malaria Diagnosis

[![Live Demo](https://img.shields.io/badge/Live%20CDSS%20Cloud-malariaproject-brightgreen?style=for-the-badge&logo=streamlit)](https://malariaproject.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Stack-PyTorch%20%7C%20FastAPI%20%7C%20Scikit--Learn%20%7C%20CatBoost%20%7C%20SHAP%20%7C%20DiCE-orange.svg)](https://scikit-learn.org)
[![Unit Tests](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen.svg)](tests/test_pipeline.py)

> 🌐 **Live Web Application (Streamlit Cloud):** **[https://malariaproject.streamlit.app/](https://malariaproject.streamlit.app/)**  
> 💻 **Local Streamlit Instance:** `streamlit run app.py` (Port 8501)  
> 🚀 **Production REST API:** `uvicorn api.main:app --port 8000 --reload` (Port 8000, Swagger at `/docs`)

---

## 📖 Academic Base Paper
> **Title:** *Explainable AI for enhanced accuracy in malaria diagnosis using ensemble machine learning models*  
> **Authors:** Olushina Olawale Awe, Peter Njoroge Mwangi, Sylvain Kotva Goudoungou, Ruth Victoria Esho, and Olanrewaju Samuel Oyejide  
> **Journal:** *BMC Medical Informatics and Decision Making* (2025) 25:162  
> **DOI:** [10.1186/s12911-025-02874-3](https://doi.org/10.1186/s12911-025-02874-3)

---

## 🎯 Final Year Project (FYP) & Publication Research Contributions
This project moves beyond naive replication to establish a state-of-the-art, clinically dependable decision-support platform designed for **academic journal publication** (*BMC Medical Informatics*, *IEEE Access*, or *Springer Nature*):

1. 🛡️ **Methodological Flaw Resolution (Data Leakage):** Proved that the 2025 BMC study performed Random Over-Sampling (ROS) prior to cross-validation splits, causing 100% memorization duplication across folds. Implemented strict leak-free train-only balancing via **SMOTE-NC**.
2. 🔬 **Generative Privacy & Fidelity Auditing:** Audited synthetic cases using **Distance-to-Closest-Record (DCR)**. Proved naive ROS yields $\text{Mean DCR} = 0.0000$ (exact memorization), while SMOTE-NC produces genuine diversity ($\text{Mean DCR} = 1.4491$, $\text{MACD} = 0.1247$).
3. 🏆 **Novel Stacking Meta-Ensemble:** Built a 2-stage hierarchical ensemble (Level-0: CatBoost, Random Forest, XGBoost, LightGBM; Level-1: $L_2$-regularized Logistic Regression). Achieved highest CV mean score ($0.8154$) and AUC ($0.915$), validated via **Wilcoxon Signed-Rank** statistical hypothesis tests.
4. 🎯 **Inductive Conformal Prediction (ICP):** Implemented distribution-free conformal uncertainty bounds guaranteeing $1-\alpha = 0.95$ statistical coverage ($98.04\%$ empirical coverage achieved).
5. 🔄 **Actionable Counterfactual Explanations (DiCE):** Generates actionable clinical treatment targets while strictly locking immutable biological features (patient age).
6. ⚖️ **Algorithmic Demographic Fairness Audit:** Evaluated disparate impact across pediatric ($\le 12$y) vs. adult cohorts. Uncovered a 40% False Negative Rate in pediatric data-driven predictions, demonstrating the imperative for deterministic WHO rule overrides.
7. 🩺 **WHO Severe Malaria Scoring & Clinical Failsafe:** 3-tier clinical urgency triage (Tier 1 Red Emergency, Tier 2 Yellow Urgent, Tier 3 Green Routine) with deterministic pediatric override for prostration and convulsions.
8. 🔬 **Multimodal Blood Smear Cytology Fusion:** PyTorch dual-branch cross-attention network uniting 1000x Giemsa thin smear microscopic cytology images with 16 syndromic biomarkers.
9. 📥 **Multi-Format Patient Reports & EHR Exports:** Instant 1-click download of official HTML medical dossiers (with `@media print` A4 CSS), HL7 FHIR-compliant JSON records, and hospital ward triage registries.
10. 🚀 **5-Pillar Translational Roadmap:** Strategic blueprint covering TinyML INT8 mobile quantization (45MB $\to$ 4.1MB, $<18$ms latency on \$50 Android tablets), Gigapixel Whole Slide Imaging (WSI), Cross-Continental Federated Learning, Enterprise EHR Interop (OpenMRS/DHIS2), and *pfkelch13* Artemisinin Resistance surveillance.

---

## 🖥️ Streamlit CDSS Navigation (9 Tabs)

1. **🩺 Patient Screening & WHO Triage:** Real-time symptom checklist, clinical gauge, WHO danger score, and **Instant Patient Diagnostic Reports (HTML / JSON / CSV)**.
2. **🔬 Multimodal Smear Cytology Fusion:** Microscopic thin blood smear cytology analysis with cross-attention fusion and Modality Congruence evaluation.
3. **🔄 Counterfactual 'What-If' Studio:** Actionable clinical intervention targets via DiCE with locked patient age.
4. **🔍 Explainable AI (LIME & SHAP):** Local sample explanations, force plots, and global summary beeswarm plots.
5. **⚖️ Algorithmic Fairness & Equity:** Demographic disparity metrics and sensitivity parity across cohorts.
6. **📊 Research Benchmarks & Novelty:** 9-classifier benchmark comparison, DCR fidelity audit table, multi-ROC curves, and calibration plots.
7. **📋 Clinical Medical Dossier & Exports:** Institutional customization (Hospital Name, Doctor Name, MRN) with native **1-Click Browser Print / Save to PDF Dialog**.
8. **📁 Batch Screener:** Batch CSV patient screening with Full CSV, Critical-Only CSV, and Hospital Ward Registry HTML downloads.
9. **🚀 Future Work & Scalability Roadmap:** Interactive 5-pillar engineering blueprint and Viva defense cheatsheet.

---

## 🎓 Academic Artifacts for Dissertation & Viva Defense

All publication and university project deliverables are pre-compiled and ready in the [`manuscript/`](manuscript/) folder:

| Asset | Format | Description |
| :--- | :---: | :--- |
| **Journal Paper Draft** | `.md` / `.html` | Formatted academic manuscript ready for submission to *BMC Medical Informatics* or *IEEE Access*. |
| **IEEE LaTeX Manuscript** | `.tex` | Two-column IEEE Transactions / Journal style with formulas, tables, and figures. |
| **BibTeX Citation Database** | `.bib` | Formal reference database with 14+ foundational citations. |
| **University Dissertation** | `.md` / `.html` | Complete B.Tech/B.S. dissertation with Certificate, Declaration, and 7 Chapters. |
| **Viva Presentation Deck** | `.md` / `.html` | 17-slide defense deck with verbatim speaker notes and closing script. |
| **Oral Defense Trainer** | `viva_mock_defense.py` | Terminal simulator testing the 7 most common examiner defense questions. |

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup
```bash
git clone https://github.com/pavanpamba16/baseproject.git
cd baseproject
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (9 Tests)
```bash
python -m unittest tests/test_pipeline.py
```

### 3. Launch Interactive Clinical CDSS (Web Workstation)
```bash
streamlit run app.py
```
*Access in browser at `http://localhost:8501`.*

### 4. Launch Production REST API
```bash
uvicorn api.main:app --port 8000 --reload
```
*Interactive Swagger documentation available at `http://localhost:8000/docs`.*

### 5. Run Command-Line Patient Diagnosis & Report Export
```bash
python diagnose_patient.py --export
```
*Generates `outputs/report_MAL-CLI-PT38.html` and `outputs/record_MAL-CLI-PT38.json`.*

### 6. Practice Oral Viva Defense
```bash
python viva_mock_defense.py --all
```

### 7. Compile Markdown Manuscripts to Print-Ready HTML/PDF
```bash
python export_manuscripts_html.py
```

---

## 📊 Benchmark Summary (Leak-Free SMOTE-NC Data)

| Model Architecture | Accuracy | ROC AUC | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacking Meta-Ensemble (Proposed SOTA)** | **0.815** | **0.915** | **0.835** | **0.910** | **0.871** |
| **CatBoost** | 0.805 | 0.904 | 0.747 | 0.908 | 0.819 |
| **Random Forest** | 0.812 | 0.885 | 0.763 | 0.892 | 0.823 |
| **XGBoost** | 0.812 | 0.882 | 0.744 | 0.938 | 0.830 |
| **Gradient Boost** | 0.767 | 0.891 | 0.702 | 0.908 | 0.792 |
| **AdaBoost** | 0.647 | 0.739 | 0.632 | 0.662 | 0.647 |
| **Deep Tab-MLP** | 0.559 | 0.594 | 0.386 | 0.486 | 0.430 |

---

## 👥 Authors & Academic Attribution
- **Student Researcher:** Pavan Pamba
- **Base Study:** Awe et al., *BMC Medical Informatics and Decision Making* (2025) 25:162
- **License:** MIT Open Source License
