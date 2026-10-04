import os
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

try:
    from src.config import RANDOM_STATE, PARAM_GRIDS, MODELS_DIR
except ImportError:
    from config import RANDOM_STATE, PARAM_GRIDS, MODELS_DIR


def get_base_models() -> dict:
    """Returns un-tuned ensemble models with fixed random seeds."""
    return {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Gradient Boost": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
    }


def tune_hyperparameters(model_name: str, base_estimator, X_train, y_train, n_iter: int = 15, cv_splits: int = 5) -> tuple:
    """
    Performs RandomizedSearchCV hyperparameter optimization as described in the paper.
    """
    param_grid = PARAM_GRIDS.get(model_name, {})
    if not param_grid:
        print(f"No tuning grid for {model_name}. Fitting base model.")
        base_estimator.fit(X_train, y_train)
        return base_estimator, {}

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Adjust n_iter if grid has fewer combinations
    total_combs = 1
    for v in param_grid.values():
        total_combs *= len(v)
    actual_n_iter = min(n_iter, total_combs)

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_grid,
        n_iter=actual_n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def save_artifact(obj, filename: str, directory: str = MODELS_DIR):
    """Save model or scaler object to disk."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    joblib.dump(obj, path)
    return path


def load_artifact(filename: str, directory: str = MODELS_DIR):
    """Load model or scaler object from disk."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)
