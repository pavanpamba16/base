# Explainable AI for Enhanced Accuracy in Malaria Diagnosis Using Ensemble Machine Learning Models

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost%20%7C%20CatBoost%20%7C%20SHAP%20%7C%20LIME-orange.svg)
![Status](https://img.shields.io/badge/Status-Fully%20Reproduced-brightgreen.svg)

---

## 📖 Reference Base Paper
> **Title:** *Explainable AI for enhanced accuracy in malaria diagnosis using ensemble machine learning models*  
> **Authors:** Olushina Olawale Awe, Peter Njoroge Mwangi, Samuel Kotva Goudoungou, Ruth Victoria Esho, and Olanrewaju Samuel Oyejide  
> **Journal:** *BMC Medical Informatics and Decision Making* (2025) 25:162  
> **DOI:** [10.1186/s12911-025-02874-3](https://doi.org/10.1186/s12911-025-02874-3)

---

## 🎯 Executive Summary & Objectives
Malaria remains a major public health challenge globally, particularly across sub-Saharan Africa. Traditional clinical diagnostic protocols often rely on optical microscopy or rapid diagnostic tests (RDTs), which suffer from sensitivity constraints during low-parasitemia infections.

This repository provides an **end-to-end, reproduction and clinical decision support system** based on the 2025 BMC paper. It integrates:
1. **Five Ensemble Classifiers:** Random Forest, AdaBoost, Gradient Boosting, XGBoost, and CatBoost.
2. **Spearman Rank Correlation Feature Selection:** Identifying and retaining 16 predictive features while dropping non-informative ones (`sex`).
3. **Class Imbalance Mitigation:** Oversampling the minority severe malaria cases on training data.
4. **Hyperparameter Tuning:** 5-fold cross-validated `RandomizedSearchCV` across all ensemble models.
5. **Transparent Explainable AI (XAI):**
   - **LIME:** Local Interpretable Model-agnostic Explanations for individual patient triage.
   - **SHAP:** Game-theoretic Shapley values showing global and patient-level symptom distributions.
   - **Permutation Feature Importance (PFI):** Empirical feature ranking across the dataset.
6. **Interactive Clinical Web Application (Streamlit):** Real-time patient triage, diagnostic probability gauges, local XAI visualizers, and batch screening.

---

## 📁 Repository Structure
```text
BASE/
├── data/
│   └── Malaria-Data.csv                 # Clinical dataset (337 patients, Federal Polytechnic Ilaro, Nigeria)
├── notebooks/
│   ├── Explainable_AI_Malaria_Diagnosis.ipynb  # Interactive Jupyter reproduction notebook
│   ├── Original_Author_Code.ipynb       # Author's raw original notebook
│   └── generate_notebook.py             # Script to regenerate notebook
├── outputs/                             # 23 generated publication figures and benchmark tables
│   ├── fig2_spearman_correlation.png    # Figure 2: Spearman correlation matrix
│   ├── fig3_class_distribution_before.png # Figure 3: Target classes before balancing
│   ├── fig4_class_distribution_after.png  # Figure 4: Label distribution after oversampling
│   ├── cm_*_before_balancing.png        # Figures 5–9: Confusion matrices before balancing
│   ├── cm_*_after_oversampling.png      # Figures 10–14: Confusion matrices after oversampling
│   ├── fig15_roc_curve.png              # Figure 15: Combined ROC curves
│   ├── lime_random_forest.png           # Figure 16: LIME Random Forest explanation
│   ├── lime_catboost.png                # Figure 17: LIME CatBoost explanation
│   ├── fig18_shap_summary.png           # Figure 18: SHAP summary beeswarm plot
│   ├── fig19_shap_overall.png           # Figure 19: Mean absolute SHAP values bar plot
│   ├── fig20_pfi_rf.png                 # Figure 20: PFI for Random Forest
│   ├── fig21_pfi_catboost.png           # Figure 21: PFI for CatBoost
│   ├── table2_before_balancing.csv      # Table 2: Benchmark results before balancing
│   ├── table3_after_oversampling.csv    # Table 3: Benchmark results after oversampling
│   └── table4_hyperparameter_tuning.csv # Table 4: Benchmark results after tuning
├── saved_models/                        # Serialized model weights and preprocessors
│   ├── random_forest_model.pkl          # Best Random Forest model
│   ├── catboost_model.pkl               # Best CatBoost model
│   ├── xgboost_model.pkl                # Best XGBoost model
│   ├── gradient_boost_model.pkl         # Best Gradient Boost model
│   ├── adaboost_model.pkl               # Best AdaBoost model
│   ├── scaler.pkl                       # StandardScaler transformer
│   ├── features.pkl                     # Ordered feature list
│   └── lime_train_data.csv              # Background data for LIME explainer
├── src/                                 # Modular Python source code
│   ├── config.py                        # Constants, paths, hyperparameter grids
│   ├── data_preprocessing.py            # Data loading, Spearman correlation, splitting, oversampling
│   ├── models.py                        # Model definitions, RandomizedSearchCV tuning, persistence
│   ├── evaluation.py                    # Metric calculations, confusion matrix & ROC plotting
│   └── explainability.py                # LIME, SHAP, and PFI implementation
├── app.py                               # Interactive Streamlit Clinical Decision Support System
├── train_pipeline.py                    # Master execution pipeline
├── requirements.txt                     # Python package dependencies
└── README.md                            # Comprehensive documentation
```

---

## 🔬 Dataset Overview
- **Origin:** Federal Polytechnic Ilaro Medical Centre, Ilaro, Ogun State, Nigeria.
- **Sample Size:** 337 patients (180 females, 157 males; ages 3 to 77).
- **Target Variable:** `severe_maleria` (1 = Severe Malaria, 0 = No Malaria).
  - *Class 0 (Non-severe / Negative):* 221 (65.6%)
  - *Class 1 (Severe Malaria):* 116 (34.4%)
- **Feature Selection:** Spearman rank correlation eliminated `sex` ($r_s = 0.00$), retaining **16 key clinical predictors**:
  1. `age`
  2. `fever`
  3. `cold`
  4. `rigor`
  5. `fatigue`
  6. `headace`
  7. `bitter_tongue`
  8. `vomitting`
  9. `diarrhea`
  10. `Convulsion`
  11. `Anemia`
  12. `jundice`
  13. `cocacola_urine` (Hemoglobinuria)
  14. `hypoglycemia`
  15. `prostraction`
  16. `hyperpyrexia` (>39°C)

---

## 📊 Experimental Results Summary

### 1. Before Balancing (Table 2 Baseline)
On imbalanced data, all models suffer from severe false negative rates and low true negative identification:
| Model | Accuracy | ROC AUC | MCC | B. Acc | Cohen's K | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.627 | 0.600 | 0.100 | 0.539 | 0.088 | 0.471 | 0.216 | 0.296 |
| **AdaBoost** | 0.647 | 0.566 | 0.112 | 0.525 | 0.062 | 0.600 | 0.081 | 0.143 |
| **Gradient Boost** | 0.618 | 0.581 | 0.100 | 0.543 | 0.094 | 0.455 | 0.270 | 0.339 |
| **CatBoost** | 0.608 | 0.602 | 0.058 | 0.523 | 0.052 | 0.421 | 0.216 | 0.286 |
| **XGBoost** | 0.588 | 0.605 | 0.088 | 0.543 | 0.088 | 0.424 | 0.378 | 0.400 |

### 2. After Oversampling (Table 3)
Balancing the minority class in the training partition resolves class bias and drastically improves discrimination:
| Model | Accuracy | ROC AUC | MCC | B. Acc | Cohen's K | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.857** | **0.918** | **0.714** | **0.857** | **0.714** | **0.859** | **0.846** | **0.853** |
| **CatBoost** | 0.842 | 0.921 | 0.689 | 0.843 | 0.685 | 0.806 | 0.892 | 0.847 |
| **XGBoost** | 0.812 | 0.867 | 0.638 | 0.814 | 0.626 | 0.756 | 0.908 | 0.825 |
| **Gradient Boost** | 0.774 | 0.883 | 0.568 | 0.777 | 0.551 | 0.716 | 0.892 | 0.795 |
| **AdaBoost** | 0.617 | 0.749 | 0.241 | 0.619 | 0.236 | 0.590 | 0.708 | 0.643 |

### 3. After Hyperparameter Tuning (Table 4)
Hyperparameter optimization using 5-fold cross-validation further regularizes models and enhances diagnostic reliability:
| Model | Accuracy | ROC AUC | MCC | Balanced Acc | Cohen's K | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.8571** | **0.9176** | **0.7141** | **0.8569** | **0.7140** | **0.8594** | **0.8462** | **0.8527** |
| **CatBoost** | 0.8271 | 0.9201 | 0.6584 | 0.8282 | 0.6548 | 0.7917 | 0.8769 | 0.8321 |
| **XGBoost** | 0.8120 | 0.8663 | 0.6377 | 0.8141 | 0.6255 | 0.7564 | 0.9077 | 0.8252 |
| **Gradient Boost** | 0.7744 | 0.9088 | 0.5633 | 0.7767 | 0.5508 | 0.7215 | 0.8769 | 0.7917 |
| **AdaBoost** | 0.6617 | 0.7682 | 0.3280 | 0.6630 | 0.3250 | 0.6351 | 0.7231 | 0.6763 |

---

## 🔍 Explainable AI (XAI) Clinical Insights
1. **Age:** Consistently ranks as the most influential feature across both SHAP and PFI, aligning with clinical evidence that age dictates physiological vulnerability and anti-disease immunity against *P. falciparum*.
2. **Coca-Cola Urine (Hemoglobinuria):** Identified by LIME and PFI as a hallmark indicator of massive intravascular hemolysis and severe blackwater fever.
3. **Prostration & Hyperpyrexia (>39°C):** Directly correlated with systemic inflammatory cascades and high parasite density.
4. **Headache & Gastrointestinal Distress (Diarrhea/Vomiting):** Common early prodromal symptoms that assist in distinguishing symptomatic illness.

---

## 🚀 How to Run the Project

### 1. Installation
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End ML Pipeline
To re-run data preprocessing, model training, hyperparameter tuning, metric evaluation, and XAI plot generation:
```bash
python train_pipeline.py
```
*All trained models will be saved in `saved_models/` and all publication-grade plots/tables in `outputs/`.*

### 3. Launch the Interactive Clinical Web Application
Launch the Streamlit clinical decision support system:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the Jupyter Notebook
Open the interactive Jupyter Notebook to step through code and figures cell by cell:
```bash
jupyter notebook notebooks/Explainable_AI_Malaria_Diagnosis.ipynb
```
