import os
import sys
import pandas as pd
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import OUTPUTS_DIR, MODELS_DIR, FEATURE_COLUMNS, TARGET_COLUMN
from src.data_preprocessing import prepare_datasets
from src.models import get_base_models, tune_hyperparameters, save_artifact
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_combined_roc_curves
from src.explainability import (
    get_lime_explainer,
    explain_instance_lime,
    compute_shap_explanations,
    compute_permutation_importance
)


def run_pipeline():
    print("=" * 70)
    print(" MALARIA DIAGNOSIS & EXPLAINABLE AI (XAI) PIPELINE")
    print(" Replicating Awe et al., BMC Medical Informatics & Decision Making (2025)")
    print("=" * 70)

    # 1. Data Preparation
    print("\n[Step 1/5] Loading and Preprocessing Data...")
    data = prepare_datasets()
    imb_data = data["imbalanced"]
    over_data = data["oversampled"]
    features = data["features"]

    print(f"  Features retained ({len(features)}): {features}")
    print(f"  Imbalanced split -> Train: {imb_data['X_train'].shape[0]}, Test: {imb_data['X_test'].shape[0]}")
    print(f"  Oversampled split -> Train: {over_data['X_train'].shape[0]}, Test: {over_data['X_test'].shape[0]}")

    # Save fitted scaler
    save_artifact(over_data["scaler"], "scaler.pkl")
    save_artifact(features, "features.pkl")

    # 2. Stage 1: Models Before Balancing (Table 2 replication)
    print("\n[Step 2/5] Training Baseline Models on Imbalanced Data (Table 2)...")
    base_models_stage1 = get_base_models()
    table2_results = []
    
    for name, model in base_models_stage1.items():
        model.fit(imb_data["X_train_scaled"], imb_data["y_train"])
        metrics = evaluate_model(model, imb_data["X_test_scaled"], imb_data["y_test"], model_name=name)
        plot_confusion_matrix(imb_data["y_test"], metrics["y_pred"], name, stage_name="before_balancing")
        table2_results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "ROC AUC": metrics["ROC AUC"],
            "MCC": metrics["MCC"],
            "B. Acc": metrics["Balanced Acc"],
            "Cohen's K.": metrics["Cohen's Kappa"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"]
        })
        
    df_table2 = pd.DataFrame(table2_results).round(3)
    df_table2.to_csv(os.path.join(OUTPUTS_DIR, "table2_before_balancing.csv"), index=False)
    print("\n--- Table 2: Performance Evaluation Metric Results Before Balancing ---")
    print(df_table2.to_string(index=False))

    # 3. Stage 2: Models After Oversampling (Table 3 replication)
    print("\n[Step 3/5] Training Models After Oversampling (Table 3)...")
    base_models_stage2 = get_base_models()
    table3_results = []
    models_stage2_fitted = {}

    for name, model in base_models_stage2.items():
        model.fit(over_data["X_train_scaled"], over_data["y_train"])
        models_stage2_fitted[name] = model
        metrics = evaluate_model(model, over_data["X_test_scaled"], over_data["y_test"], model_name=name)
        plot_confusion_matrix(over_data["y_test"], metrics["y_pred"], name, stage_name="after_oversampling")
        table3_results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "ROC AUC": metrics["ROC AUC"],
            "MCC": metrics["MCC"],
            "B. Acc": metrics["Balanced Acc"],
            "Cohen's K.": metrics["Cohen's Kappa"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"]
        })

    df_table3 = pd.DataFrame(table3_results).round(3)
    df_table3.to_csv(os.path.join(OUTPUTS_DIR, "table3_after_oversampling.csv"), index=False)
    print("\n--- Table 3: Performance Evaluation Metric Results After Oversampling ---")
    print(df_table3.to_string(index=False))

    # Plot ROC curve for classifiers (Fig 15)
    plot_combined_roc_curves(
        models_stage2_fitted,
        over_data["X_test_scaled"],
        over_data["y_test"],
        title="ROC Curve for Classifiers (After Oversampling)",
        filename="fig15_roc_curve.png"
    )

    # 4. Stage 3: Hyperparameter Tuning (Table 4 replication)
    print("\n[Step 4/5] Hyperparameter Tuning via RandomizedSearchCV (Table 4)...")
    base_models_stage3 = get_base_models()
    table4_results = []
    best_tuned_models = {}

    for name, model in base_models_stage3.items():
        print(f"  Tuning {name}...")
        best_est, best_params = tune_hyperparameters(
            name, model, over_data["X_train_scaled"], over_data["y_train"], n_iter=15, cv_splits=5
        )
        best_tuned_models[name] = best_est
        metrics = evaluate_model(best_est, over_data["X_test_scaled"], over_data["y_test"], model_name=name)
        
        # Save model artifact
        save_artifact(best_est, f"{name.lower().replace(' ', '_')}_model.pkl")

        table4_results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "ROC AUC": metrics["ROC AUC"],
            "MCC": metrics["MCC"],
            "Balanced A.": metrics["Balanced Acc"],
            "Cohen's K.": metrics["Cohen's Kappa"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "Best Params": str(best_params)
        })

    df_table4 = pd.DataFrame(table4_results).round(4)
    df_table4.to_csv(os.path.join(OUTPUTS_DIR, "table4_hyperparameter_tuning.csv"), index=False)
    print("\n--- Table 4: Model Performance After Hyperparameter Tuning ---")
    print(df_table4.drop(columns=["Best Params"]).to_string(index=False))

    # 5. Explainable AI Frameworks (LIME, SHAP, Permutation Feature Importance)
    print("\n[Step 5/5] Generating Explainable AI (XAI) Visualizations...")
    best_rf = best_tuned_models["Random Forest"]
    best_cat = best_tuned_models["CatBoost"]

    # 5.1 LIME
    print("  Generating LIME explanations (Figures 16 & 17)...")
    # Save training data for LIME initialization
    over_data["X_train_scaled"].to_csv(os.path.join(MODELS_DIR, "lime_train_data.csv"), index=False)
    lime_explainer = get_lime_explainer(over_data["X_train_scaled"])
    
    instance_0 = over_data["X_test_scaled"].iloc[0]
    explain_instance_lime(lime_explainer, instance_0, best_rf.predict_proba, model_name="Random Forest", save_plot=True)
    explain_instance_lime(lime_explainer, instance_0, best_cat.predict_proba, model_name="CatBoost", save_plot=True)

    # 5.2 SHAP
    print("  Generating SHAP summary and feature importance plots (Figures 18 & 19)...")
    compute_shap_explanations(best_rf, over_data["X_train_scaled"], over_data["X_test_scaled"], model_name="Random Forest")

    # 5.3 Permutation Feature Importance
    print("  Computing Permutation Feature Importance for Random Forest & CatBoost (Figures 20 & 21)...")
    compute_permutation_importance(best_rf, over_data["X_test_scaled"], over_data["y_test"], model_name="Random Forest")
    compute_permutation_importance(best_cat, over_data["X_test_scaled"], over_data["y_test"], model_name="CatBoost")

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" All models saved to: {MODELS_DIR}")
    print(f" All figures & tables saved to: {OUTPUTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
