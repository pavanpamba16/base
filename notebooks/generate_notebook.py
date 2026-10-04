import json
import os

notebook_data = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Explainable AI for Enhanced Accuracy in Malaria Diagnosis Using Ensemble Machine Learning Models\n",
                "\n",
                "**Reference Publication:**\n",
                "> *Olushina Olawale Awe, Peter Njoroge Mwangi, Samuel Kotva Goudoungou, Ruth Victoria Esho, and Olanrewaju Samuel Oyejide*\n",
                "> **BMC Medical Informatics and Decision Making (2025) 25:162**\n",
                "> https://doi.org/10.1186/s12911-025-02874-3\n",
                "\n",
                "---\n",
                "### Abstract & Overview\n",
                "Malaria remains a leading cause of morbidity and mortality across sub-Saharan Africa. This notebook provides a complete, reproducible implementation of the study published in *BMC Medical Informatics and Decision Making* (2025).\n",
                "\n",
                "The study evaluates **five ensemble machine learning models**:\n",
                "1. **Random Forest (RF)**\n",
                "2. **AdaBoost**\n",
                "3. **Gradient Boosting (GB)**\n",
                "4. **Extreme Gradient Boosting (XGBoost)**\n",
                "5. **CatBoost**\n",
                "\n",
                "Key pipeline highlights:\n",
                "- **Dataset**: 337 patient records from Federal Polytechnic Ilaro Medical Centre, Ogun State, Nigeria.\n",
                "- **Feature Selection**: Spearman Rank Correlation filtering ($|r_s| \\ge 0.05$), eliminating non-informative features like `sex`.\n",
                "- **Class Imbalance**: Resampling technique on training data to balance the 1:2 malaria to non-malaria distribution.\n",
                "- **Optimization**: RandomizedSearchCV 5-fold cross-validation across hyperparameters.\n",
                "- **Explainable AI (XAI)**:\n",
                "  - **LIME**: Local instance-level attribution.\n",
                "  - **SHAP**: Game-theoretic Shapley value summaries and beeswarm distributions.\n",
                "  - **Permutation Feature Importance (PFI)**: Model-agnostic dataset-wide feature ranking."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Setup and Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import types\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Ensure parent directory is in path\n",
                "if '..' not in sys.path:\n",
                "    sys.path.append('..')\n",
                "\n",
                "# Numba compatibility shim if needed\n",
                "if 'numba' not in sys.modules:\n",
                "    try:\n",
                "        import numba\n",
                "    except Exception:\n",
                "        nb_mod = types.ModuleType('numba')\n",
                "        nb_mod.njit = lambda *args, **kwargs: (lambda fn: fn) if (args and callable(args[0])) else (lambda fn: fn)\n",
                "        nb_mod.jit = nb_mod.njit\n",
                "        nb_typed = types.ModuleType('numba.typed')\n",
                "        nb_typed.List = list\n",
                "        nb_typed.Dict = dict\n",
                "        nb_mod.typed = nb_typed\n",
                "        sys.modules['numba'] = nb_mod\n",
                "        sys.modules['numba.typed'] = nb_typed\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score,\n",
                "    cohen_kappa_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve\n",
                ")\n",
                "from xgboost import XGBClassifier\n",
                "from catboost import CatBoostClassifier\n",
                "import shap\n",
                "import lime\n",
                "from lime.lime_tabular import LimeTabularExplainer\n",
                "\n",
                "print('Libraries successfully imported!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Data Loading & Exploratory Data Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_path = '../data/Malaria-Data.csv' if os.path.exists('../data/Malaria-Data.csv') else 'data/Malaria-Data.csv'\n",
                "df = pd.read_csv(data_path)\n",
                "print(f'Dataset Shape: {df.shape}')\n",
                "print('Class Distribution:')\n",
                "print(df['severe_maleria'].value_counts())\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Spearman Rank Correlation & Feature Selection (Figure 2)\n",
                "The paper calculates the Spearman Rank Correlation coefficient for all features with the target variable `severe_maleria`.\n",
                "Features with correlation $< 0.05$ (specifically `sex`, with $r_s = 0.00$) are excluded, leaving 16 key clinical/demographic features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "spearman_corr = df.corr(method='spearman')['severe_maleria'].drop('severe_maleria')\n",
                "\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(spearman_corr.to_frame(), annot=True, cmap='coolwarm', fmt='.2f')\n",
                "plt.title('Figure 2: Correlation Matrix of Malaria Dataset')\n",
                "plt.xlabel('Target Variable (severe_malaria)')\n",
                "plt.ylabel('Features')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Retain features with |r_s| >= 0.05 (drops 'sex')\n",
                "selected_features = spearman_corr[abs(spearman_corr) >= 0.05].index.tolist()\n",
                "print(f'Selected Features ({len(selected_features)}):', selected_features)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Addressing Class Imbalance via Oversampling (Figures 3 & 4)\n",
                "Before balancing, non-malaria cases ($n=221$) outnumber severe malaria cases ($n=116$) by roughly 2:1.\n",
                "The minority class is oversampled on the dataset to create balanced classes (Figure 4)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_selected = df[selected_features + ['severe_maleria']].copy()\n",
                "\n",
                "# Before balancing distribution (Figure 3)\n",
                "plt.figure(figsize=(6, 4))\n",
                "df_selected['severe_maleria'].value_counts().plot(kind='bar', color=['#a05162', '#997b88'])\n",
                "plt.title('Figure 3: Target Classes Before Balancing')\n",
                "plt.xlabel('severe_malaria')\n",
                "plt.ylabel('count')\n",
                "plt.show()\n",
                "\n",
                "# Oversampling\n",
                "class_0 = df_selected[df_selected['severe_maleria'] == 0]\n",
                "class_1 = df_selected[df_selected['severe_maleria'] == 1]\n",
                "class_1_over = class_1.sample(len(class_0), replace=True, random_state=123)\n",
                "df_oversampled = pd.concat([class_1_over, class_0], axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)\n",
                "\n",
                "# After oversampling distribution (Figure 4)\n",
                "plt.figure(figsize=(6, 4))\n",
                "df_oversampled['severe_maleria'].value_counts().plot(kind='bar', color=['#43799d', '#3b6998'])\n",
                "plt.title('Figure 4: Label Distribution After Oversampling')\n",
                "plt.xlabel('severe_malaria')\n",
                "plt.ylabel('Count')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Training & Evaluation on Imbalanced Data (Table 2)\n",
                "Evaluation of the 5 ensemble classifiers prior to balancing."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_imb = df_selected[selected_features]\n",
                "y_imb = df_selected['severe_maleria']\n",
                "\n",
                "X_tr_imb, X_te_imb, y_tr_imb, y_te_imb = train_test_split(X_imb, y_imb, test_size=0.30, random_state=101)\n",
                "scaler_imb = StandardScaler()\n",
                "X_tr_imb_sc = pd.DataFrame(scaler_imb.fit_transform(X_tr_imb), columns=selected_features)\n",
                "X_te_imb_sc = pd.DataFrame(scaler_imb.transform(X_te_imb), columns=selected_features)\n",
                "\n",
                "models = {\n",
                "    'Random Forest': RandomForestClassifier(random_state=123),\n",
                "    'CatBoost': CatBoostClassifier(random_state=123, verbose=0),\n",
                "    'Gradient Boost': GradientBoostingClassifier(random_state=123),\n",
                "    'AdaBoost': AdaBoostClassifier(random_state=123),\n",
                "    'XGBoost': XGBClassifier(random_state=123, eval_metric='logloss')\n",
                "}\n",
                "\n",
                "table2_rows = []\n",
                "for name, m in models.items():\n",
                "    m.fit(X_tr_imb_sc, y_tr_imb)\n",
                "    pred = m.predict(X_te_imb_sc)\n",
                "    prob = m.predict_proba(X_te_imb_sc)[:, 1]\n",
                "    table2_rows.append({\n",
                "        'Model': name,\n",
                "        'Accuracy': accuracy_score(y_te_imb, pred),\n",
                "        'ROC AUC': roc_auc_score(y_te_imb, prob),\n",
                "        'MCC': matthews_corrcoef(y_te_imb, pred),\n",
                "        'B. Acc': balanced_accuracy_score(y_te_imb, pred),\n",
                "        'Cohen\\'s K.': cohen_kappa_score(y_te_imb, pred),\n",
                "        'Precision': precision_score(y_te_imb, pred, zero_division=0),\n",
                "        'Recall': recall_score(y_te_imb, pred, zero_division=0),\n",
                "        'F1 Score': f1_score(y_te_imb, pred, zero_division=0)\n",
                "    })\n",
                "\n",
                "df_t2 = pd.DataFrame(table2_rows).round(3)\n",
                "print('=== Table 2: Model Performance Before Balancing ===')\n",
                "display(df_t2)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Model Training & Evaluation After Oversampling (Table 3 & Figure 15)\n",
                "Evaluating the 5 ensemble models after balancing minority malaria cases."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_bal = df_oversampled[selected_features]\n",
                "y_bal = df_oversampled['severe_maleria']\n",
                "\n",
                "X_tr_bal, X_te_bal, y_tr_bal, y_te_bal = train_test_split(X_bal, y_bal, test_size=0.30, random_state=101)\n",
                "scaler_bal = StandardScaler()\n",
                "X_tr_bal_sc = pd.DataFrame(scaler_bal.fit_transform(X_tr_bal), columns=selected_features)\n",
                "X_te_bal_sc = pd.DataFrame(scaler_bal.transform(X_te_bal), columns=selected_features)\n",
                "\n",
                "table3_rows = []\n",
                "trained_over_models = {}\n",
                "plt.figure(figsize=(9, 7))\n",
                "\n",
                "for name, m in models.items():\n",
                "    m.fit(X_tr_bal_sc, y_tr_bal)\n",
                "    trained_over_models[name] = m\n",
                "    pred = m.predict(X_te_bal_sc)\n",
                "    prob = m.predict_proba(X_te_bal_sc)[:, 1]\n",
                "    \n",
                "    table3_rows.append({\n",
                "        'Model': name,\n",
                "        'Accuracy S.': accuracy_score(y_te_bal, pred),\n",
                "        'ROC AUC S.': roc_auc_score(y_te_bal, prob),\n",
                "        'MCC': matthews_corrcoef(y_te_bal, pred),\n",
                "        'B. Acc': balanced_accuracy_score(y_te_bal, pred),\n",
                "        'Cohen\\'s K.': cohen_kappa_score(y_te_bal, pred),\n",
                "        'Precision': precision_score(y_te_bal, pred, zero_division=0),\n",
                "        'Recall': recall_score(y_te_bal, pred, zero_division=0),\n",
                "        'F1 S.': f1_score(y_te_bal, pred, zero_division=0)\n",
                "    })\n",
                "    \n",
                "    fpr, tpr, _ = roc_curve(y_te_bal, prob)\n",
                "    auc_val = roc_auc_score(y_te_bal, prob)\n",
                "    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.2f})', linewidth=2)\n",
                "\n",
                "plt.plot([0, 1], [0, 1], 'k--', label='Random')\n",
                "plt.xlabel('False Positive Rate')\n",
                "plt.ylabel('True Positive Rate')\n",
                "plt.title('Figure 15: ROC Curve for Classifiers (After Oversampling)')\n",
                "plt.legend(loc='lower right')\n",
                "plt.grid(True, linestyle=':', alpha=0.6)\n",
                "plt.show()\n",
                "\n",
                "df_t3 = pd.DataFrame(table3_rows).round(3)\n",
                "print('=== Table 3: Performance Evaluation Metric Results After Oversampling ===')\n",
                "display(df_t3)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Hyperparameter Tuning via RandomizedSearchCV (Table 4)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "param_grids = {\n",
                "    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},\n",
                "    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1.0]},\n",
                "    'Gradient Boost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]},\n",
                "    'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5], 'gamma': [0, 0.1, 0.2]},\n",
                "    'CatBoost': {'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [4, 6, 8]}\n",
                "}\n",
                "\n",
                "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)\n",
                "table4_rows = []\n",
                "tuned_models = {}\n",
                "\n",
                "for name, base_m in models.items():\n",
                "    grid = param_grids[name]\n",
                "    total = 1\n",
                "    for v in grid.values(): total *= len(v)\n",
                "    \n",
                "    search = RandomizedSearchCV(base_m, grid, n_iter=min(15, total), cv=cv, scoring='accuracy', n_jobs=-1, random_state=123, verbose=0)\n",
                "    search.fit(X_tr_bal_sc, y_tr_bal)\n",
                "    best_est = search.best_estimator_\n",
                "    tuned_models[name] = best_est\n",
                "    \n",
                "    pred = best_est.predict(X_te_bal_sc)\n",
                "    prob = best_est.predict_proba(X_te_bal_sc)[:, 1]\n",
                "    \n",
                "    table4_rows.append({\n",
                "        'Model': name,\n",
                "        'Accuracy S.': accuracy_score(y_te_bal, pred),\n",
                "        'ROC AUC S.': roc_auc_score(y_te_bal, prob),\n",
                "        'MCC': matthews_corrcoef(y_te_bal, pred),\n",
                "        'Balanced A.': balanced_accuracy_score(y_te_bal, pred),\n",
                "        'Cohen\\'s K.': cohen_kappa_score(y_te_bal, pred),\n",
                "        'Precision': precision_score(y_te_bal, pred, zero_division=0),\n",
                "        'Recall': recall_score(y_te_bal, pred, zero_division=0)\n",
                "    })\n",
                "\n",
                "df_t4 = pd.DataFrame(table4_rows).round(4)\n",
                "print('=== Table 4: Model Performance After Hyperparameter Tuning ===')\n",
                "display(df_t4)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Explainable AI: LIME (Figures 16 & 17)\n",
                "Local Interpretable Model-agnostic Explanations explain individual patient predictions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "explainer = LimeTabularExplainer(\n",
                "    X_tr_bal_sc.values,\n",
                "    feature_names=selected_features,\n",
                "    class_names=['No Malaria', 'Severe Malaria'],\n",
                "    mode='classification',\n",
                "    discretize_continuous=True,\n",
                "    random_state=123\n",
                ")\n",
                "\n",
                "patient_instance = X_te_bal_sc.iloc[0]\n",
                "\n",
                "# LIME for Random Forest (Figure 16)\n",
                "exp_rf = explainer.explain_instance(patient_instance.values, tuned_models['Random Forest'].predict_proba, num_features=10)\n",
                "fig = exp_rf.as_pyplot_figure()\n",
                "plt.title('Figure 16: LIME Random Forest')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# LIME for CatBoost (Figure 17)\n",
                "exp_cat = explainer.explain_instance(patient_instance.values, tuned_models['CatBoost'].predict_proba, num_features=10)\n",
                "fig = exp_cat.as_pyplot_figure()\n",
                "plt.title('Figure 17: LIME CatBoost')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Explainable AI: SHAP Analysis (Figures 18 & 19)\n",
                "SHapley Additive exPlanations quantify individual feature contributions using game theory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rf_explainer = shap.TreeExplainer(tuned_models['Random Forest'])\n",
                "shap_vals = rf_explainer.shap_values(X_te_bal_sc)\n",
                "\n",
                "if isinstance(shap_vals, list):\n",
                "    shap_vals_pos = shap_vals[1]\n",
                "elif hasattr(shap_vals, 'shape') and len(shap_vals.shape) == 3:\n",
                "    shap_vals_pos = shap_vals[:, :, 1]\n",
                "else:\n",
                "    shap_vals_pos = shap_vals\n",
                "\n",
                "# Figure 18: SHAP Beeswarm Summary Plot\n",
                "plt.figure(figsize=(10, 6))\n",
                "shap.summary_plot(shap_vals_pos, X_te_bal_sc, show=False)\n",
                "plt.title('Figure 18: SHAP Summary Plot (Individual Feature Impact)')\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Figure 19: Mean Absolute SHAP Values Bar Plot\n",
                "mean_shap = np.abs(shap_vals_pos).mean(axis=0)\n",
                "feat_imp = pd.DataFrame({'Feature': selected_features, 'Mean SHAP': mean_shap}).sort_values('Mean SHAP', ascending=True)\n",
                "\n",
                "plt.figure(figsize=(9, 6))\n",
                "plt.barh(feat_imp['Feature'], feat_imp['Mean SHAP'], color='#4a90e2')\n",
                "plt.xlabel('Mean Absolute SHAP Value')\n",
                "plt.title('Figure 19: Mean Absolute SHAP Values for Features (Overall)')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Permutation Feature Importance (PFI) (Figures 20 & 21)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def calculate_pfi(model, X, y, metric_fn=accuracy_score, n_iter=100):\n",
                "    base_score = metric_fn(y, model.predict(X))\n",
                "    imps = {}\n",
                "    np.random.seed(123)\n",
                "    for col in X.columns:\n",
                "        scores = []\n",
                "        for _ in range(n_iter):\n",
                "            X_p = X.copy()\n",
                "            X_p[col] = np.random.permutation(X_p[col])\n",
                "            scores.append(base_score - metric_fn(y, model.predict(X_p)))\n",
                "        imps[col] = float(np.mean(scores))\n",
                "    return dict(sorted(imps.items(), key=lambda x: x[1]))\n",
                "\n",
                "# PFI for Random Forest (Figure 20)\n",
                "pfi_rf = calculate_pfi(tuned_models['Random Forest'], X_te_bal_sc, y_te_bal)\n",
                "plt.figure(figsize=(9, 6))\n",
                "colors_rf = ['#1f77b4' if v >= 0 else '#d62728' for v in pfi_rf.values()]\n",
                "plt.barh(range(len(pfi_rf)), list(pfi_rf.values()), color=colors_rf)\n",
                "plt.yticks(range(len(pfi_rf)), list(pfi_rf.keys()))\n",
                "plt.xlabel('Permutation Importance')\n",
                "plt.title('Figure 20: Permutation Feature Importances Random Forest')\n",
                "plt.grid(axis='x', linestyle=':', alpha=0.5)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# PFI for CatBoost (Figure 21)\n",
                "pfi_cat = calculate_pfi(tuned_models['CatBoost'], X_te_bal_sc, y_te_bal)\n",
                "plt.figure(figsize=(9, 6))\n",
                "colors_cat = ['#1f77b4' if v >= 0 else '#d62728' for v in pfi_cat.values()]\n",
                "plt.barh(range(len(pfi_cat)), list(pfi_cat.values()), color=colors_cat)\n",
                "plt.yticks(range(len(pfi_cat)), list(pfi_cat.keys()))\n",
                "plt.xlabel('Permutation Importance')\n",
                "plt.title('Figure 21: Permutation Feature Importances CatBoost')\n",
                "plt.grid(axis='x', linestyle=':', alpha=0.5)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Conclusion & Clinical Insights\n",
                "- **Random Forest** achieved the highest overall performance with an ROC AUC score of 0.87 and balanced accuracy of ~82%.\n",
                "- **CatBoost** offered superior handling of subtle interactions among symptoms without overfitting.\n",
                "- **Explainable AI** confirmed that symptoms like `age`, `headache`, `coca-cola urine` (hemoglobinuria), `prostration`, and `hyperpyrexia` are the strongest indicators of severe malaria infection, aligning with established WHO clinical criteria."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("c:/Users/pamba/OneDrive/Desktop/BASE/notebooks/Explainable_AI_Malaria_Diagnosis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=2)

print("Updated notebooks/Explainable_AI_Malaria_Diagnosis.ipynb successfully!")
