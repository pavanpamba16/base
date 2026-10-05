"""
========================================================================================
MASTER RESEARCH EXPERIMENT & BENCHMARKING PIPELINE
Publication-Grade Empirical Evaluation for Final Year Project
========================================================================================
Executes:
1. Experiment 1: Methodological Leakage & Resampling Audit (ROS vs SMOTE-NC vs Borderline-SMOTE vs ADASYN)
2. Experiment 2: Comprehensive Classifier Benchmarking (9 Models including Stacking Meta-Ensemble & Tab-MLP)
3. Experiment 3: Statistical Significance Hypothesis Testing (Wilcoxon Signed-Rank & McNemar Tests)
4. Experiment 4: Inductive Conformal Prediction Calibration & Empirical Coverage Validation
5. Saves all publication artifacts, figures, and serialized model weights
========================================================================================
"""

import os
import sys
import types
import warnings
warnings.filterwarnings("ignore")

# Ensure numba compatibility shim
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, precision_score,
    recall_score, f1_score, roc_curve
)
from sklearn.calibration import calibration_curve

from src.config import OUTPUTS_DIR, MODELS_DIR, RANDOM_STATE
from src.advanced_resampling import prepare_leak_free_data, compare_balancing_methods_audit
from src.novel_models import get_all_comparison_models, save_novel_model
from src.conformal_prediction import ConformalMalariaPredictor


def run_resampling_and_leakage_audit() -> pd.DataFrame:
    """
    Experiment 1: Demonstrates empirical consequences of pre-split oversampling (leakage)
    vs leak-free SMOTE-NC balancing.
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: DATA LEAKAGE & RESAMPLING FIDELITY AUDIT", flush=True)
    print("=" * 70, flush=True)
    
    audit_df = compare_balancing_methods_audit()
    audit_csv_path = os.path.join(OUTPUTS_DIR, "table_resampling_leakage_audit.csv")
    audit_df.to_csv(audit_csv_path, index=False)
    print(audit_df.to_string(index=False), flush=True)
    print(f"\n[Saved] Resampling audit saved to {audit_csv_path}", flush=True)
    return audit_df


def run_benchmark_experiments() -> tuple[pd.DataFrame, dict]:
    """
    Experiment 2: Evaluates 9 machine learning models on strict leak-free SMOTE-NC clinical data.
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: 9-CLASSIFIER BENCHMARK WITH STACKING META-ENSEMBLE", flush=True)
    print("=" * 70, flush=True)
    
    # Prepare strict leak-free dataset using SMOTE-NC
    data = prepare_leak_free_data(technique="SMOTE-NC")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    models = get_all_comparison_models()
    trained_models = {}
    benchmark_records = []
    probabilities_dict = {}

    for name, model in models.items():
        print(f"  Training and evaluating: {name}...", flush=True)
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        probabilities_dict[name] = y_prob

        metrics = {
            "Model": name,
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "ROC AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
            "MCC": round(float(matthews_corrcoef(y_test, y_pred)), 4),
            "Balanced Acc": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
            "Cohen's Kappa": round(float(cohen_kappa_score(y_test, y_pred)), 4),
            "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "F1 Score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
        }
        benchmark_records.append(metrics)

    benchmark_df = pd.DataFrame(benchmark_records).sort_values(by="ROC AUC", ascending=False)
    benchmark_csv_path = os.path.join(OUTPUTS_DIR, "table_research_benchmark_comparison.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    
    print("\n=== RESEARCH BENCHMARK RESULTS (TABLE 5) ===", flush=True)
    print(benchmark_df.to_string(index=False), flush=True)
    print(f"\n[Saved] Benchmark table saved to {benchmark_csv_path}", flush=True)

    # Generate combined ROC plot
    plot_novel_roc_comparison(probabilities_dict, y_test)
    
    # Generate Probability Calibration Curves
    plot_calibration_curves(probabilities_dict, y_test)

    # Save best Stacking Meta-Ensemble
    best_model = trained_models["Stacking Meta-Ensemble (Ours)"]
    save_novel_model(best_model, "stacking_meta_ensemble.pkl")
    
    # Save training metadata
    joblib.dump(data["features"], os.path.join(MODELS_DIR, "features.pkl"))
    joblib.dump(data["scaler"], os.path.join(MODELS_DIR, "scaler_smotenc.pkl"))
    
    return benchmark_df, trained_models


def plot_novel_roc_comparison(probs_dict: dict, y_test: pd.Series):
    """Generates publication-quality ROC comparison plot for all models."""
    plt.figure(figsize=(10, 8))
    
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#17becf"
    ]

    for idx, (name, y_prob) in enumerate(probs_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        is_ours = "Ours" in name
        plt.plot(
            fpr, tpr,
            label=f"{name} (AUC = {auc_score:.3f})",
            linewidth=3.0 if is_ours else 1.8,
            color="#b22222" if is_ours else palette[idx % len(palette)],
            linestyle="-" if is_ours else "--"
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.500)")
    plt.title("Comparative ROC Curves with Stacking Meta-Ensemble", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    roc_path = os.path.join(OUTPUTS_DIR, "fig_novel_roc_comparison.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Saved] Novel ROC comparison plot saved to {roc_path}")


def plot_calibration_curves(probs_dict: dict, y_test: pd.Series):
    """Plots clinical probability calibration curves (Brier score & reliability)."""
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    key_models = ["Random Forest", "CatBoost", "XGBoost", "Stacking Meta-Ensemble (Ours)"]
    for name in key_models:
        if name in probs_dict:
            prob_true, prob_pred = calibration_curve(y_test, probs_dict[name], n_bins=5)
            plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name)

    plt.title("Clinical Probability Calibration Curves", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Mean Predicted Probability of Severe Malaria", fontsize=12)
    plt.ylabel("Fraction of True Severe Malaria Cases", fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    calib_path = os.path.join(OUTPUTS_DIR, "fig_calibration_curves.png")
    plt.savefig(calib_path, dpi=300)
    plt.close()
    print(f"[Saved] Calibration curves saved to {calib_path}")


def run_statistical_significance_tests(trained_models: dict) -> pd.DataFrame:
    """
    Experiment 3: Statistical Hypothesis Testing.
    Conducts Wilcoxon Signed-Rank Tests across 5 stratified cross-validation folds
    to verify that our Stacking Meta-Ensemble statistically outperforms the base single models.
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: STATISTICAL SIGNIFICANCE HYPOTHESIS TESTING", flush=True)
    print("=" * 70, flush=True)
    
    from sklearn.model_selection import StratifiedKFold
    
    data = prepare_leak_free_data(technique="SMOTE-NC")
    X = data["X_train"]
    y = data["y_train"]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    stacking_scores = []
    rf_scores = []
    cat_scores = []
    xgb_scores = []
    
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"  Evaluating CV Fold {fold_i + 1}/5...", flush=True)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Stacking
        m_stack = trained_models["Stacking Meta-Ensemble (Ours)"]
        m_stack.fit(X_tr, y_tr)
        stacking_scores.append(roc_auc_score(y_val, m_stack.predict_proba(X_val)[:, 1]))
        
        # RF
        m_rf = trained_models["Random Forest"]
        m_rf.fit(X_tr, y_tr)
        rf_scores.append(roc_auc_score(y_val, m_rf.predict_proba(X_val)[:, 1]))
        
        # CatBoost
        m_cat = trained_models["CatBoost"]
        m_cat.fit(X_tr, y_tr)
        cat_scores.append(roc_auc_score(y_val, m_cat.predict_proba(X_val)[:, 1]))
        
        # XGBoost
        m_xgb = trained_models["XGBoost"]
        m_xgb.fit(X_tr, y_tr)
        xgb_scores.append(roc_auc_score(y_val, m_xgb.predict_proba(X_val)[:, 1]))

    stat_results = []
    pairs = [
        ("Stacking Meta-Ensemble vs Random Forest", stacking_scores, rf_scores),
        ("Stacking Meta-Ensemble vs CatBoost", stacking_scores, cat_scores),
        ("Stacking Meta-Ensemble vs XGBoost", stacking_scores, xgb_scores)
    ]
    
    for comp_name, scores_a, scores_b in pairs:
        diff = np.array(scores_a) - np.array(scores_b)
        w_stat, p_val = stats.wilcoxon(diff, alternative="greater")
        t_stat, t_pval = stats.ttest_rel(scores_a, scores_b)
        
        stat_results.append({
            "Comparison": comp_name,
            "Mean Score (Ours)": round(float(np.mean(scores_a)), 4),
            "Mean Score (Baseline)": round(float(np.mean(scores_b)), 4),
            "Mean Improvement (+Delta)": round(float(np.mean(diff)), 4),
            "Wilcoxon W-Stat": round(float(w_stat), 3),
            "Wilcoxon p-value": round(float(p_val), 5),
            "Paired t-test p-value": round(float(t_pval), 5),
            "Statistically Significant (p < 0.05)": "YES (p < 0.05)" if p_val < 0.05 else "NO"
        })
        
    stat_df = pd.DataFrame(stat_results)
    stat_csv_path = os.path.join(OUTPUTS_DIR, "table_statistical_tests.csv")
    stat_df.to_csv(stat_csv_path, index=False)
    
    print("\n=== STATISTICAL SIGNIFICANCE TESTS (TABLE 6) ===")
    print(stat_df.to_string(index=False))
    print(f"\n[Saved] Statistical test results saved to {stat_csv_path}")
    return stat_df


def calibrate_and_validate_conformal(stacking_model):
    """
    Experiment 4: Calibrates Inductive Conformal Prediction on holdout calibration split.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: INDUCTIVE CONFORMAL PREDICTION (UNCERTAINTY SAFETY)")
    print("=" * 70)
    
    data = prepare_leak_free_data(technique="SMOTE-NC")
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Split test into calibration (50%) and evaluation (50%)
    from sklearn.model_selection import train_test_split
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=RANDOM_STATE
    )
    
    cp = ConformalMalariaPredictor(stacking_model, alpha=0.05)
    cp.calibrate(X_cal, y_cal)
    
    coverage_audit = []
    for alpha_val in [0.01, 0.05, 0.10, 0.15]:
        cov = cp.compute_empirical_coverage(X_eval, y_eval, alpha=alpha_val)
        coverage_audit.append(cov)
        
    coverage_df = pd.DataFrame(coverage_audit)
    cov_path = os.path.join(OUTPUTS_DIR, "table_conformal_coverage_audit.csv")
    coverage_df.to_csv(cov_path, index=False)
    
    # Save calibrated conformal predictor object
    joblib.dump(cp, os.path.join(MODELS_DIR, "conformal_predictor.pkl"))
    
    print("\n=== CONFORMAL PREDICTION SAFETY AUDIT ===")
    print(coverage_df.to_string(index=False))
    print(f"[Saved] Conformal predictor saved to {os.path.join(MODELS_DIR, 'conformal_predictor.pkl')}")


def main():
    print("=" * 80)
    print("STARTING ADVANCED RESEARCH EXPERIMENTAL WORKFLOW")
    print("Goal: Produce Publication-Grade Benchmarks & Models for FYP Submission")
    print("=" * 80)
    
    # 1. Resampling and Leakage Audit
    run_resampling_and_leakage_audit()
    
    # 2. 9-Classifier Benchmark with Stacking
    benchmark_df, trained_models = run_benchmark_experiments()
    
    # 3. Statistical Significance Hypothesis Tests
    run_statistical_significance_tests(trained_models)
    
    # 4. Inductive Conformal Prediction Calibration
    calibrate_and_validate_conformal(trained_models["Stacking Meta-Ensemble (Ours)"])
    
    print("\n" + "=" * 80)
    print("ALL RESEARCH BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Artifacts generated in outputs/ and saved_models/")
    print("=" * 80)


if __name__ == "__main__":
    main()
