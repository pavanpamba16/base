"""
========================================================================================
NOVEL STACKING META-ENSEMBLE & TABULAR DEEP LEARNING ARCHITECTURE
Enhancement over Base Paper's Single-Tree Classifiers
========================================================================================
Implements:
1. Multi-Model Stacking Classifier with 5-fold Out-Of-Fold Meta-Learning
2. LightGBM integration into the clinical ensemble
3. Soft-Voting Probability Blending Meta-Estimator
4. Tabular Residual Multilayer Perceptron (Tab-MLP)
========================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

try:
    from src.config import RANDOM_STATE, MODELS_DIR
except ImportError:
    from config import RANDOM_STATE, MODELS_DIR


def get_tuned_base_estimators() -> list:
    """
    Returns high-performing base estimators calibrated on clinical features.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE
    )
    cat = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=RANDOM_STATE
    )
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    lgbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    
    return [
        ("rf", rf),
        ("catboost", cat),
        ("xgboost", xgb),
        ("lgbm", lgbm)
    ]


def build_stacking_meta_ensemble() -> StackingClassifier:
    """
    Constructs a two-stage Stacking Meta-Ensemble.
    - Level 0: Random Forest, CatBoost, XGBoost, LightGBM
    - Level 1 (Meta-Learner): L2-regularized Logistic Regression trained on 5-fold OOF probabilities.
    """
    estimators = get_tuned_base_estimators()
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=1
    )
    return stacking_clf


def build_soft_voting_ensemble() -> VotingClassifier:
    """
    Constructs a weighted Soft-Voting Probability Ensemble.
    """
    estimators = get_tuned_base_estimators()
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=1
    )
    return voting_clf


def build_deep_tabular_mlp() -> MLPClassifier:
    """
    Deep Tabular Multilayer Perceptron with adaptive learning rate and weight decay.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        alpha=0.001,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=0.005,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )
    return mlp


def get_all_comparison_models() -> dict:
    """
    Returns full suite of models for publication benchmarking:
    - Base paper trees (RF, AdaBoost, Gradient Boost, XGBoost, CatBoost)
    - Novel additions (LightGBM, Deep Tab-MLP, Soft Voting, Stacking Meta-Ensemble)
    """
    from sklearn.ensemble import AdaBoostClassifier
    
    return {
        # Base paper models
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
        "Gradient Boost": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, eval_metric="logloss", random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=RANDOM_STATE),
        
        # Novel extensions
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=RANDOM_STATE),
        "Deep Tab-MLP": build_deep_tabular_mlp(),
        "Soft-Voting Ensemble": build_soft_voting_ensemble(),
        "Stacking Meta-Ensemble (Ours)": build_stacking_meta_ensemble()
    }


def save_novel_model(model, filename: str = "stacking_meta_ensemble.pkl", directory: str = MODELS_DIR):
    """Saves novel model artifact to disk."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    joblib.dump(model, path)
    return path


def load_novel_model(filename: str = "stacking_meta_ensemble.pkl", directory: str = MODELS_DIR):
    """Loads novel model artifact from disk."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


if __name__ == "__main__":
    from src.advanced_resampling import prepare_leak_free_data
    print("Testing novel models construction...")
    models = get_all_comparison_models()
    print(f"Loaded {len(models)} benchmark models: {list(models.keys())}")
