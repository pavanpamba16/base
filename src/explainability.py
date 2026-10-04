import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import types

# Ensure numba compatibility shim if numba DLL is blocked by OS security policy
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

import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score

try:
    from src.config import OUTPUTS_DIR, CLASS_NAMES
except ImportError:
    from config import OUTPUTS_DIR, CLASS_NAMES


def get_lime_explainer(X_train_df: pd.DataFrame, class_names: list = CLASS_NAMES) -> LimeTabularExplainer:
    """Initializes and returns a LimeTabularExplainer instance."""
    explainer = LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=list(X_train_df.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=123
    )
    return explainer


def explain_instance_lime(explainer: LimeTabularExplainer, instance: pd.Series, predict_proba_fn, model_name: str = "Model", num_features: int = 10, save_plot: bool = True):
    """
    Explains an individual patient prediction using LIME.
    Generates local feature contribution bar chart matching Figures 16 & 17.
    """
    exp = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=predict_proba_fn,
        num_features=num_features
    )
    
    if save_plot:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME Explanation for {model_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        filename = f"lime_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(OUTPUTS_DIR, filename)
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        
    return exp


def compute_shap_explanations(model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str = "Random Forest"):
    """
    Computes SHAP values using TreeExplainer.
    Generates:
    - Beeswarm summary plot (Fig 18)
    - Mean absolute SHAP values bar plot (Fig 19)
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle SHAP multi-class vs binary output formats across versions
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        shap_vals_pos = shap_values[:, :, 1]
    else:
        shap_vals_pos = shap_values

    # 1. SHAP Beeswarm summary plot (Fig 18)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_vals_pos, X_test, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = os.path.join(OUTPUTS_DIR, "fig18_shap_summary.png")
    plt.savefig(beeswarm_path, dpi=300)
    plt.close()

    # 2. Mean Absolute SHAP values bar plot (Fig 19)
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean SHAP Value": mean_abs_shap
    }).sort_values(by="Mean SHAP Value", ascending=True)

    plt.figure(figsize=(9, 7))
    plt.barh(shap_df["Feature"], shap_df["Mean SHAP Value"], color="#4a90e2", edgecolor="navy", alpha=0.85)
    plt.xlabel("Mean Absolute SHAP Value", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.title("Mean Absolute SHAP Values for Features", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUTS_DIR, "fig19_shap_overall.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    return {
        "explainer": explainer,
        "shap_values_pos": shap_vals_pos,
        "mean_shap_df": shap_df
    }


def compute_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series, metric_fn=accuracy_score, num_iterations: int = 100, model_name: str = "Random Forest"):
    """
    Calculates Permutation Feature Importance (PFI) by shuffling features and measuring performance drop.
    Matches Figures 20 (RF) and 21 (CatBoost).
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    baseline_score = metric_fn(y_test, model.predict(X_test))
    importances = {}
    
    np.random.seed(123)
    for col in X_test.columns:
        scores = []
        for _ in range(num_iterations):
            X_perm = X_test.copy()
            X_perm[col] = np.random.permutation(X_perm[col])
            perm_score = metric_fn(y_test, model.predict(X_perm))
            scores.append(baseline_score - perm_score)
        importances[col] = float(np.mean(scores))

    # Sort ascending for horizontal bar chart
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=False))
    
    # Plotting
    plt.figure(figsize=(9, 6.5))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in sorted_importances.values()]
    plt.barh(range(len(sorted_importances)), list(sorted_importances.values()), color=colors, edgecolor="black", linewidth=0.5)
    plt.yticks(range(len(sorted_importances)), list(sorted_importances.keys()), fontsize=10)
    plt.xlabel("Permutation Importance", fontsize=11)
    plt.title(f"Permutation Feature Importances {model_name}", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    
    clean_name = "rf" if "random" in model_name.lower() else "catboost"
    filename = f"fig20_pfi_{clean_name}.png" if clean_name == "rf" else f"fig21_pfi_{clean_name}.png"
    filepath = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return sorted_importances
