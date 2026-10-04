import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from src.config import DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, RANDOM_STATE, SPLIT_RANDOM_STATE, OUTPUTS_DIR
except ImportError:
    from config import DATA_PATH, TARGET_COLUMN, FEATURE_COLUMNS, RANDOM_STATE, SPLIT_RANDOM_STATE, OUTPUTS_DIR


def load_raw_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df


def calculate_spearman_correlation(df: pd.DataFrame, target_col: str = TARGET_COLUMN, threshold: float = 0.05, save_plot: bool = True) -> tuple[pd.Series, list[str]]:
    """
    Calculate Spearman Rank Correlation of all features with the target variable.
    Filters out features with |correlation| < threshold (e.g. sex).
    """
    corr_matrix = df.corr(method="spearman")
    target_corr = corr_matrix[target_col].drop(target_col)
    
    # Selected features based on threshold
    selected = target_corr[abs(target_corr) >= threshold].index.tolist()
    
    if save_plot:
        plt.figure(figsize=(10, 8))
        sorted_corr = target_corr.to_frame()
        sns.heatmap(sorted_corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title("Spearman Correlation of Features with severe_malaria")
        plt.xlabel("Target Variable")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "fig2_spearman_correlation.png"), dpi=300)
        plt.close()
        
    return target_corr, selected


def plot_class_distributions(df_before: pd.DataFrame, df_after: pd.DataFrame, target_col: str = TARGET_COLUMN):
    """Plot target class distribution before and after oversampling (Figures 3 and 4 in paper)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Before balancing
    plt.figure(figsize=(6, 5))
    counts_before = df_before[target_col].value_counts().sort_index()
    colors = ["#a05162", "#997b88"]
    plt.bar([str(i) for i in counts_before.index], counts_before.values, color=colors, width=0.6)
    plt.title("Count of Malaria Result (Before Balancing)")
    plt.xlabel("severe_malaria")
    plt.ylabel("count")
    for i, v in enumerate(counts_before.values):
        plt.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig3_class_distribution_before.png"), dpi=300)
    plt.close()
    
    # After oversampling
    plt.figure(figsize=(6, 5))
    counts_after = df_after[target_col].value_counts().sort_index()
    colors_after = ["#43799d", "#3b6998"]
    plt.bar([str(i) for i in counts_after.index], counts_after.values, color=colors_after, width=0.6)
    plt.title("Label Distribution after Oversampling")
    plt.xlabel("severe_malaria")
    plt.ylabel("Count")
    for i, v in enumerate(counts_after.values):
        plt.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "fig4_class_distribution_after.png"), dpi=300)
    plt.close()


def prepare_datasets(test_size: float = 0.30, random_state: int = SPLIT_RANDOM_STATE):
    """
    Executes the complete data preparation workflow:
    1. Loads dataset
    2. Drops non-informative features (sex) via Spearman correlation
    3. Generates imbalanced splits (for Table 2 baseline)
    4. Generates oversampled balanced dataset (for Tables 3 & 4)
    5. Standardizes features using StandardScaler
    Returns dictionary with all datasets and transformers.
    """
    df = load_raw_data()
    
    # Calculate Spearman correlation and plot Fig 2
    target_corr, selected_feats = calculate_spearman_correlation(df, threshold=0.05)
    
    # Features excluding sex (retaining the 16 clinical/demographic features)
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    df_selected = df[features + [TARGET_COLUMN]].copy()
    
    # 1. Imbalanced Dataset (Original)
    X_imbalanced = df_selected[features]
    y_imbalanced = df_selected[TARGET_COLUMN]
    
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imbalanced, y_imbalanced, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    scaler_imb = StandardScaler()
    X_train_imb_scaled = pd.DataFrame(scaler_imb.fit_transform(X_train_imb), columns=features, index=X_train_imb.index)
    X_test_imb_scaled = pd.DataFrame(scaler_imb.transform(X_test_imb), columns=features, index=X_test_imb.index)
    
    # 2. Oversampled Dataset
    class_0 = df_selected[df_selected[TARGET_COLUMN] == 0]
    class_1 = df_selected[df_selected[TARGET_COLUMN] == 1]
    
    # Resample minority class 1 to match majority class 0
    class_1_over = class_1.sample(len(class_0), replace=True, random_state=RANDOM_STATE)
    df_oversampled = pd.concat([class_1_over, class_0], axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Save plots
    plot_class_distributions(df_selected, df_oversampled)
    
    X_bal = df_oversampled[features]
    y_bal = df_oversampled[TARGET_COLUMN]
    
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
        X_bal, y_bal, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    scaler_bal = StandardScaler()
    X_train_bal_scaled = pd.DataFrame(scaler_bal.fit_transform(X_train_bal), columns=features, index=X_train_bal.index)
    X_test_bal_scaled = pd.DataFrame(scaler_bal.transform(X_test_bal), columns=features, index=X_test_bal.index)
    
    return {
        "features": features,
        "imbalanced": {
            "X_train": X_train_imb,
            "X_test": X_test_imb,
            "X_train_scaled": X_train_imb_scaled,
            "X_test_scaled": X_test_imb_scaled,
            "y_train": y_train_imb,
            "y_test": y_test_imb,
            "scaler": scaler_imb,
            "df": df_selected
        },
        "oversampled": {
            "X_train": X_train_bal,
            "X_test": X_test_bal,
            "X_train_scaled": X_train_bal_scaled,
            "X_test_scaled": X_test_bal_scaled,
            "y_train": y_train_bal,
            "y_test": y_test_bal,
            "scaler": scaler_bal,
            "df": df_oversampled
        }
    }


if __name__ == "__main__":
    data = prepare_datasets()
    print("Preprocessed features:", data["features"])
    print("Imbalanced Train shape:", data["imbalanced"]["X_train"].shape)
    print("Imbalanced Test shape:", data["imbalanced"]["X_test"].shape)
    print("Oversampled Train shape:", data["oversampled"]["X_train"].shape)
    print("Oversampled Test shape:", data["oversampled"]["X_test"].shape)
