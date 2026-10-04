import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Malaria-Data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Random seeds
RANDOM_STATE = 123
SPLIT_RANDOM_STATE = 101

# Dataset definitions
TARGET_COLUMN = "severe_maleria"
EXCLUDED_FEATURES = ["sex"]  # Excluded based on Spearman Rank Correlation < 0.05

FEATURE_COLUMNS = [
    "age",
    "fever",
    "cold",
    "rigor",
    "fatigue",
    "headace",
    "bitter_tongue",
    "vomitting",
    "diarrhea",
    "Convulsion",
    "Anemia",
    "jundice",
    "cocacola_urine",
    "hypoglycemia",
    "prostraction",
    "hyperpyrexia"
]

FEATURE_DESCRIPTIONS = {
    "age": "Age of the patient in years (3 - 77)",
    "fever": "Presence of fever (1: Yes, 0: No)",
    "cold": "Presence of cold symptoms (1: Yes, 0: No)",
    "rigor": "Presence of rigor/shivering (1: Yes, 0: No)",
    "fatigue": "Presence of fatigue or tiredness (1: Yes, 0: No)",
    "headace": "Presence of headache (1: Yes, 0: No)",
    "bitter_tongue": "Presence of bitter taste in mouth (1: Yes, 0: No)",
    "vomitting": "Presence of vomiting (1: Yes, 0: No)",
    "diarrhea": "Presence of diarrhea (1: Yes, 0: No)",
    "Convulsion": "Presence of convulsions/seizures (1: Yes, 0: No)",
    "Anemia": "Reduced RBC count or hemoglobin (1: Yes, 0: No)",
    "jundice": "Yellowing of skin and eyes (1: Yes, 0: No)",
    "cocacola_urine": "Dark-colored Coca-Cola urine (1: Yes, 0: No)",
    "hypoglycemia": "Low blood sugar levels (1: Yes, 0: No)",
    "prostraction": "Extreme weakness / unable to sit or stand (1: Yes, 0: No)",
    "hyperpyrexia": "Extremely high body temperature > 39°C (1: Yes, 0: No)",
}

CLASS_NAMES = ["No Malaria", "Severe Malaria"]

# Hyperparameter search grids from paper
PARAM_GRIDS = {
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
