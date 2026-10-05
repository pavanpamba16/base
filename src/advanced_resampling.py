"""
========================================================================================
ADVANCED TABULAR RESAMPLING & GENERATIVE FIDELITY SUITE
Addresses the Data Leakage & Synthetic Sampling Flaws of the Base 2025 BMC Paper
========================================================================================
Implements:
1. Leak-Free Cross-Validation Splitter (ensuring test sets remain completely unseen)
2. SMOTE-NC (Synthetic Minority Over-sampling for Nominal/Continuous features)
3. Borderline-SMOTE and ADASYN
4. Empirical Synthetic Fidelity & Privacy Auditing (Distance-to-Closest-Record, DCR)
========================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, SMOTENC
from scipy.spatial.distance import cdist

try:
    from src.config import (
        DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, CATEGORICAL_INDICES,
        RANDOM_STATE, SPLIT_RANDOM_STATE
    )
    from src.data_preprocessing import load_raw_data
except ImportError:
    from config import (
        DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, CATEGORICAL_INDICES,
        RANDOM_STATE, SPLIT_RANDOM_STATE
    )
    from data_preprocessing import load_raw_data


def get_resampling_strategies(categorical_indices: list = CATEGORICAL_INDICES, random_state: int = RANDOM_STATE) -> dict:
    """Returns dictionary of modern tabular oversampling transformers."""
    return {
        "Random Over-Sampling (ROS)": RandomOverSampler(random_state=random_state),
        "SMOTE": SMOTE(random_state=random_state),
        "SMOTE-NC": SMOTENC(categorical_features=categorical_indices, random_state=random_state),
        "Borderline-SMOTE": BorderlineSMOTE(random_state=random_state),
        "ADASYN": ADASYN(random_state=random_state)
    }


def compute_synthetic_fidelity_metrics(X_real: pd.DataFrame, X_synthetic: pd.DataFrame) -> dict:
    """
    Evaluates synthetic data fidelity and privacy against real clinical cohort:
    1. Mean Distance to Closest Record (DCR) - measures novelty vs exact duplication (memorization).
    2. Minimum DCR - checks if any synthetic row is an exact carbon-copy of a patient.
    3. Mean Absolute Correlation Difference (MACD) - measures preservation of inter-feature covariance.
    """
    if len(X_synthetic) == 0:
        return {"Mean DCR": 0.0, "Min DCR": 0.0, "MACD": 0.0}

    # Normalize before computing pairwise Euclidean distances
    scaler = StandardScaler()
    X_real_norm = scaler.fit_transform(X_real)
    X_syn_norm = scaler.transform(X_synthetic)

    # Compute Euclidean distance from each synthetic point to closest real point
    distances = cdist(X_syn_norm, X_real_norm, metric="euclidean")
    min_distances = np.min(distances, axis=1)

    mean_dcr = float(np.mean(min_distances))
    min_dcr = float(np.min(min_distances))

    # Correlation difference
    corr_real = X_real.corr().fillna(0).values
    corr_syn = X_synthetic.corr().fillna(0).values
    macd = float(np.mean(np.abs(corr_real - corr_syn)))

    return {
        "Mean DCR": round(mean_dcr, 4),
        "Min DCR": round(min_dcr, 4),
        "MACD": round(macd, 4)
    }


def prepare_leak_free_data(
    technique: str = "SMOTE-NC",
    test_size: float = 0.30,
    random_state: int = SPLIT_RANDOM_STATE
) -> dict:
    """
    STRICT LEAK-FREE PIPELINE:
    1. Loads data and filters informative clinical features.
    2. Splits raw data into pure train and test partitions BEFORE any oversampling.
    3. Fits resampling strictly on X_train, y_train.
    4. Evaluates strictly on untouched, un-augmented X_test.
    """
    df = load_raw_data()
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[features].copy()
    y = df[TARGET_COLUMN].copy()

    # Step 1: Honest split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, stratify=y, random_state=random_state
    )

    strategies = get_resampling_strategies(random_state=RANDOM_STATE)
    sampler = strategies.get(technique, strategies["SMOTE-NC"])

    # Step 2: Apply resampling ONLY on train partition
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

    # Identify purely synthetic rows generated for the minority class
    minority_real_count = int((y_train == 1).sum())
    synthetic_minority_rows = X_train_resampled[y_train_resampled == 1].iloc[minority_real_count:]

    fidelity_metrics = compute_synthetic_fidelity_metrics(
        X_real=X_train[y_train == 1],
        X_synthetic=synthetic_minority_rows
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_resampled), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    return {
        "technique": technique,
        "features": features,
        "X_train": X_train_resampled,
        "y_train": y_train_resampled,
        "X_train_scaled": X_train_scaled,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "fidelity": fidelity_metrics
    }


def compare_balancing_methods_audit():
    """Generates comparative tabular summary of oversampling methods and fidelity."""
    strategies = [
        "Random Over-Sampling (ROS)",
        "SMOTE",
        "SMOTE-NC",
        "Borderline-SMOTE",
        "ADASYN"
    ]
    results = []
    for strat in strategies:
        data = prepare_leak_free_data(technique=strat)
        results.append({
            "Method": strat,
            "Train Samples (Balanced)": len(data["X_train"]),
            "Minority Class 1": int((data["y_train"] == 1).sum()),
            "Majority Class 0": int((data["y_train"] == 0).sum()),
            "Mean Distance to Closest Record (DCR)": data["fidelity"]["Mean DCR"],
            "Min DCR (Memorization Risk)": data["fidelity"]["Min DCR"],
            "Correlation Preservation (MACD)": data["fidelity"]["MACD"]
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    audit_df = compare_balancing_methods_audit()
    print("=== SYNTHETIC GENERATIVE BALANCING AUDIT ===")
    print(audit_df.to_string(index=False))
