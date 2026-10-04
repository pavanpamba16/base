import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

try:
    from src.config import OUTPUTS_DIR
except ImportError:
    from config import OUTPUTS_DIR


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Computes full suite of classification metrics as reported in the base paper."""
    y_pred = model.predict(X_test)
    
    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred

    metrics = {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "ROC AUC": float(roc_auc_score(y_test, y_prob)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "Balanced Acc": float(balanced_accuracy_score(y_test, y_pred)),
        "Cohen's Kappa": float(cohen_kappa_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, stage_name: str = "after_oversampling", save_dir: str = OUTPUTS_DIR):
    """Generates styled confusion matrix heatmap matching the publication figures."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        cbar=True,
        xticklabels=[0, 1],
        yticklabels=[0, 1]
    )
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    filename = f"cm_{model_name.lower().replace(' ', '_')}_{stage_name}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_combined_roc_curves(models_dict: dict, X_test, y_test, title: str = "ROC Curve for Classifiers", filename: str = "fig15_roc_curve.png", save_dir: str = OUTPUTS_DIR):
    """
    Plots multi-model ROC curves on a single plot matching Figure 15 in the paper.
    """
    plt.figure(figsize=(9, 7))
    
    colors = {
        "Random Forest": "#4a7bb0",
        "AdaBoost": "#e69f00",
        "Gradient Boost": "#56b4e9",
        "XGBoost": "#cc79a7",
        "CatBoost": "#d55e00"
    }

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = colors.get(name, None)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})", color=color, linewidth=2)
        
    plt.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.7)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath
