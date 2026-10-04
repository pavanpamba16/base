"""
========================================================================================
Explainable AI for enhanced accuracy in malaria diagnosis using ensemble machine learning models
BMC Medical Informatics and Decision Making (2025) 25:162
Authors: Olushina Olawale Awe, Peter Njoroge Mwangi, Samuel Kotva Goudoungou,
         Ruth Victoria Esho, and Olanrewaju Samuel Oyejide
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import lime
from lime.lime_tabular import LimeTabularExplainer
import shap

# Output directory setup
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Malaria-Data.csv")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def main():
    print("=" * 80)
    print(" BASE PAPER IMPLEMENTATION: MALARIA DIAGNOSIS USING ENSEMBLE ML & XAI")
    print(" Publication: BMC Medical Informatics and Decision Making (2025) 25:162")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. DATA LOADING & SPEARMAN CORRELATION FEATURE SELECTION (Figure 2)
    # -------------------------------------------------------------------------
    print("\n[1] Loading dataset & performing Spearman Rank Correlation analysis...")
    df = pd.read_csv(DATA_PATH)
    print(f"    Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    target_col = "severe_maleria"
    corr_matrix = df.corr(method="spearman")
    target_corr = corr_matrix[target_col].drop(target_col)

    # Plot Figure 2: Spearman correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(target_corr.to_frame(), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Figure 2: Correlation Matrix of Malaria Dataset", fontsize=12, fontweight="bold")
    plt.xlabel("Target Variable (severe_malaria)")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig2_spearman_correlation.png"), dpi=300)
    plt.close()
    print("    Saved Figure 2 -> outputs/fig2_spearman_correlation.png")

    # Feature elimination: correlation threshold 0.05 eliminates 'sex' (0.00), retains 16 features
    selected_features = target_corr[abs(target_corr) >= 0.05].index.tolist()
    print(f"    Features retained after eliminating 'sex' ({len(selected_features)}): {selected_features}")

    df_selected = df[selected_features + [target_col]].copy()

    # -------------------------------------------------------------------------
    # 2. CLASS IMBALANCE & OVERSAMPLING (Figures 3 & 4)
    # -------------------------------------------------------------------------
    print("\n[2] Handling class imbalance...")
    # Plot Figure 3: Class distribution before balancing
    plt.figure(figsize=(6, 5))
    counts_before = df_selected[target_col].value_counts().sort_index()
    plt.bar([str(i) for i in counts_before.index], counts_before.values, color=["#a05162", "#997b88"], width=0.6)
    plt.title("Figure 3: Target Classes Before Balancing", fontsize=12, fontweight="bold")
    plt.xlabel("severe_malaria")
    plt.ylabel("count")
    for i, v in enumerate(counts_before.values):
        plt.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig3_class_distribution_before.png"), dpi=300)
    plt.close()
    print(f"    Before balancing -> Class 0: {counts_before.get(0, 0)}, Class 1: {counts_before.get(1, 0)}")

    # Oversample minority class 1
    class_0 = df_selected[df_selected[target_col] == 0]
    class_1 = df_selected[df_selected[target_col] == 1]
    class_1_over = class_1.sample(len(class_0), replace=True, random_state=123)
    df_oversampled = pd.concat([class_1_over, class_0], axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)

    # Plot Figure 4: Class distribution after oversampling
    plt.figure(figsize=(6, 5))
    counts_after = df_oversampled[target_col].value_counts().sort_index()
    plt.bar([str(i) for i in counts_after.index], counts_after.values, color=["#43799d", "#3b6998"], width=0.6)
    plt.title("Figure 4: Label Distribution After Oversampling", fontsize=12, fontweight="bold")
    plt.xlabel("severe_malaria")
    plt.ylabel("Count")
    for i, v in enumerate(counts_after.values):
        plt.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig4_class_distribution_after.png"), dpi=300)
    plt.close()
    print(f"    After oversampling -> Class 0: {counts_after.get(0, 0)}, Class 1: {counts_after.get(1, 0)}")

    # Train/Test splits (70:30 ratio as specified in paper)
    X_imb = df_selected[selected_features]
    y_imb = df_selected[target_col]
    X_tr_imb, X_te_imb, y_tr_imb, y_te_imb = train_test_split(X_imb, y_imb, test_size=0.30, random_state=101)

    scaler_imb = StandardScaler()
    X_tr_imb_sc = pd.DataFrame(scaler_imb.fit_transform(X_tr_imb), columns=selected_features)
    X_te_imb_sc = pd.DataFrame(scaler_imb.transform(X_te_imb), columns=selected_features)

    X_bal = df_oversampled[selected_features]
    y_bal = df_oversampled[target_col]
    X_tr_bal, X_te_bal, y_tr_bal, y_te_bal = train_test_split(X_bal, y_bal, test_size=0.30, random_state=101)

    scaler_bal = StandardScaler()
    X_tr_bal_sc = pd.DataFrame(scaler_bal.fit_transform(X_tr_bal), columns=selected_features)
    X_te_bal_sc = pd.DataFrame(scaler_bal.transform(X_te_bal), columns=selected_features)

    # -------------------------------------------------------------------------
    # 3. BASELINE MODELS BEFORE BALANCING (Table 2, Figures 5-9)
    # -------------------------------------------------------------------------
    print("\n[3] Training 5 Ensemble Models Before Balancing (Table 2)...")
    base_models = {
        "Random Forest": RandomForestClassifier(random_state=123),
        "AdaBoost": AdaBoostClassifier(random_state=123),
        "Gradient Boost": GradientBoostingClassifier(random_state=123),
        "XGBoost": XGBClassifier(random_state=123, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(random_state=123, verbose=0)
    }

    t2_records = []
    for name, model in base_models.items():
        model.fit(X_tr_imb_sc, y_tr_imb)
        preds = model.predict(X_te_imb_sc)
        probs = model.predict_proba(X_te_imb_sc)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_te_imb, preds)
        plt.figure(figsize=(5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", cbar=True)
        plt.title(f"Confusion Matrix for {name} (Before Balancing)", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f"cm_{name.lower().replace(' ', '_')}_before_balancing.png"), dpi=300)
        plt.close()

        t2_records.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te_imb, preds),
            "ROC AUC": roc_auc_score(y_te_imb, probs),
            "MCC": matthews_corrcoef(y_te_imb, preds),
            "B. Acc": balanced_accuracy_score(y_te_imb, preds),
            "Cohen's K.": cohen_kappa_score(y_te_imb, preds),
            "Precision": precision_score(y_te_imb, preds, zero_division=0),
            "Recall": recall_score(y_te_imb, preds, zero_division=0),
            "F1 Score": f1_score(y_te_imb, preds, zero_division=0)
        })

    df_t2 = pd.DataFrame(t2_records).round(3)
    df_t2.to_csv(os.path.join(OUTPUTS_DIR, "table2_before_balancing.csv"), index=False)
    print("\n--- TABLE 2: PERFORMANCE RESULTS BEFORE BALANCING ---")
    print(df_t2.to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. MODELS AFTER OVERSAMPLING (Table 3, Figures 10-15)
    # -------------------------------------------------------------------------
    print("\n[4] Training 5 Ensemble Models After Oversampling (Table 3)...")
    t3_records = []
    models_stage2 = {
        "Random Forest": RandomForestClassifier(random_state=123),
        "AdaBoost": AdaBoostClassifier(random_state=123),
        "Gradient Boost": GradientBoostingClassifier(random_state=123),
        "XGBoost": XGBClassifier(random_state=123, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(random_state=123, verbose=0)
    }

    plt.figure(figsize=(9, 7))
    for name, model in models_stage2.items():
        model.fit(X_tr_bal_sc, y_tr_bal)
        preds = model.predict(X_te_bal_sc)
        probs = model.predict_proba(X_te_bal_sc)[:, 1]

        # Confusion Matrix
        cm = confusion_matrix(y_te_bal, preds)
        plt.figure(figsize=(5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", cbar=True)
        plt.title(f"Confusion Matrix for {name} (After Oversampling)", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, f"cm_{name.lower().replace(' ', '_')}_after_oversampling.png"), dpi=300)
        plt.close()

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_te_bal, probs)
        auc_val = roc_auc_score(y_te_bal, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.2f})", linewidth=2)

        t3_records.append({
            "Model": name,
            "Accuracy S.": accuracy_score(y_te_bal, preds),
            "ROC AUC S.": roc_auc_score(y_te_bal, probs),
            "MCC": matthews_corrcoef(y_te_bal, preds),
            "B. Acc": balanced_accuracy_score(y_te_bal, preds),
            "Cohen's K.": cohen_kappa_score(y_te_bal, preds),
            "Precision": precision_score(y_te_bal, preds, zero_division=0),
            "Recall": recall_score(y_te_bal, preds, zero_division=0),
            "F1 S.": f1_score(y_te_bal, preds, zero_division=0)
        })

    plt.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Figure 15: ROC Curve for Classifiers (After Oversampling)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig15_roc_curve.png"), dpi=300)
    plt.close()

    df_t3 = pd.DataFrame(t3_records).round(3)
    df_t3.to_csv(os.path.join(OUTPUTS_DIR, "table3_after_oversampling.csv"), index=False)
    print("\n--- TABLE 3: PERFORMANCE RESULTS AFTER OVERSAMPLING ---")
    print(df_t3.to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. HYPERPARAMETER TUNING VIA RANDOMIZEDSEARCHCV (Table 4)
    # -------------------------------------------------------------------------
    print("\n[5] Hyperparameter Tuning with RandomizedSearchCV 5-Fold CV (Table 4)...")
    param_grids = {
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "AdaBoost": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.1, 1.0],
        },
        "Gradient Boost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "gamma": [0, 0.1, 0.2],
        },
        "CatBoost": {
            "iterations": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [4, 6, 8],
        },
    }

    t4_records = []
    best_tuned_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    for name, base_clf in base_models.items():
        print(f"    Tuning {name}...")
        grid = param_grids[name]
        total_combs = 1
        for v in grid.values():
            total_combs *= len(v)
        n_iter = min(15, total_combs)

        search = RandomizedSearchCV(
            estimator=base_clf,
            param_distributions=grid,
            n_iter=n_iter,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            random_state=123,
            verbose=0
        )
        search.fit(X_tr_bal_sc, y_tr_bal)
        best_model = search.best_estimator_
        best_tuned_models[name] = best_model

        preds = best_model.predict(X_te_bal_sc)
        probs = best_model.predict_proba(X_te_bal_sc)[:, 1]

        t4_records.append({
            "Model": name,
            "Accuracy S.": accuracy_score(y_te_bal, preds),
            "ROC AUC S.": roc_auc_score(y_te_bal, probs),
            "MCC": matthews_corrcoef(y_te_bal, preds),
            "Balanced A.": balanced_accuracy_score(y_te_bal, preds),
            "Cohen's K.": cohen_kappa_score(y_te_bal, preds),
            "Precision": precision_score(y_te_bal, preds, zero_division=0),
            "Recall": recall_score(y_te_bal, preds, zero_division=0)
        })

    df_t4 = pd.DataFrame(t4_records).round(4)
    df_t4.to_csv(os.path.join(OUTPUTS_DIR, "table4_hyperparameter_tuning.csv"), index=False)
    print("\n--- TABLE 4: MODEL PERFORMANCE AFTER HYPERPARAMETER TUNING ---")
    print(df_t4.to_string(index=False))

    # -------------------------------------------------------------------------
    # 6. EXPLAINABLE AI (XAI): LIME, SHAP, PERMUTATION FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    print("\n[6] Computing Explainable AI Frameworks...")
    best_rf = best_tuned_models["Random Forest"]
    best_cat = best_tuned_models["CatBoost"]

    # 6.1 LIME (Figures 16 & 17)
    print("    Generating LIME Local Explanations (Figures 16 & 17)...")
    explainer = LimeTabularExplainer(
        training_data=X_tr_bal_sc.values,
        feature_names=selected_features,
        class_names=["No Malaria", "Severe Malaria"],
        mode="classification",
        discretize_continuous=True,
        random_state=123
    )

    test_instance = X_te_bal_sc.iloc[0]

    # Figure 16: LIME Random Forest
    exp_rf = explainer.explain_instance(test_instance.values, best_rf.predict_proba, num_features=10)
    fig = exp_rf.as_pyplot_figure()
    plt.title("Figure 16: LIME Random Forest", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "lime_random_forest.png"), dpi=300)
    plt.close(fig)

    # Figure 17: LIME CatBoost
    exp_cat = explainer.explain_instance(test_instance.values, best_cat.predict_proba, num_features=10)
    fig = exp_cat.as_pyplot_figure()
    plt.title("Figure 17: LIME CatBoost", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "lime_catboost.png"), dpi=300)
    plt.close(fig)

    # 6.2 SHAP (Figures 18 & 19)
    print("    Computing SHAP Values & Visualizations (Figures 18 & 19)...")
    tree_explainer = shap.TreeExplainer(best_rf)
    shap_vals = tree_explainer.shap_values(X_te_bal_sc)

    if isinstance(shap_vals, list):
        shap_vals_pos = shap_vals[1]
    elif hasattr(shap_vals, "shape") and len(shap_vals.shape) == 3:
        shap_vals_pos = shap_vals[:, :, 1]
    else:
        shap_vals_pos = shap_vals

    # Figure 18: SHAP Beeswarm summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_pos, X_te_bal_sc, show=False)
    plt.title("Figure 18: SHAP Individual Summary Plot", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig18_shap_summary.png"), dpi=300)
    plt.close()

    # Figure 19: Mean absolute SHAP bar plot
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": selected_features, "Mean SHAP": mean_abs_shap}).sort_values("Mean SHAP", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(shap_df["Feature"], shap_df["Mean SHAP"], color="#4a90e2")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Figure 19: Mean Absolute SHAP Values for Features (Overall)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig19_shap_overall.png"), dpi=300)
    plt.close()

    # 6.3 Permutation Feature Importance (Figures 20 & 21)
    print("    Calculating Permutation Feature Importance (Figures 20 & 21)...")

    def calc_pfi(model, X, y, metric_fn=accuracy_score, n_iter=100):
        base_score = metric_fn(y, model.predict(X))
        imps = {}
        np.random.seed(123)
        for col in X.columns:
            scores = []
            for _ in range(n_iter):
                X_p = X.copy()
                X_p[col] = np.random.permutation(X_p[col])
                scores.append(base_score - metric_fn(y, model.predict(X_p)))
            imps[col] = float(np.mean(scores))
        return dict(sorted(imps.items(), key=lambda x: x[1]))

    # Figure 20: PFI Random Forest
    pfi_rf = calc_pfi(best_rf, X_te_bal_sc, y_te_bal)
    plt.figure(figsize=(9, 6))
    colors_rf = ["#1f77b4" if v >= 0 else "#d62728" for v in pfi_rf.values()]
    plt.barh(range(len(pfi_rf)), list(pfi_rf.values()), color=colors_rf)
    plt.yticks(range(len(pfi_rf)), list(pfi_rf.keys()))
    plt.xlabel("Permutation Importance")
    plt.title("Figure 20: Permutation Feature Importances Random Forest", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig20_pfi_rf.png"), dpi=300)
    plt.close()

    # Figure 21: PFI CatBoost
    pfi_cat = calc_pfi(best_cat, X_te_bal_sc, y_te_bal)
    plt.figure(figsize=(9, 6))
    colors_cat = ["#1f77b4" if v >= 0 else "#d62728" for v in pfi_cat.values()]
    plt.barh(range(len(pfi_cat)), list(pfi_cat.values()), color=colors_cat)
    plt.yticks(range(len(pfi_cat)), list(pfi_cat.keys()))
    plt.xlabel("Permutation Importance")
    plt.title("Figure 21: Permutation Feature Importances CatBoost", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig21_pfi_catboost.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print(" BASE PAPER REPRODUCTION COMPLETED SUCCESSFULLY!")
    print(f" All 23 Figures & CSV Benchmark Tables are in: {OUTPUTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
