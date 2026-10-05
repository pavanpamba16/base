"""
========================================================================================
CONFORMAL PREDICTION & UNCERTAINTY QUANTIFICATION FOR CLINICAL TRIAGE
Guaranteed Coverage Sets under User-Defined Confidence (1 - alpha)
========================================================================================
Implements:
1. Inductive Conformal Prediction (ICP) for Binary Clinical Classification
2. Non-Conformity Scoring based on Softmax Probability Margins
3. Triage Safety Filter:
   - Singleton {0}: Confidently Negative
   - Singleton {1}: Confidently Severe Malaria
   - Pair {0, 1}: Clinically Uncertain -> Trigger Immediate Microscopic Smear
   - Empty {}: Out-of-Distribution Presentation
========================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ConformalMalariaPredictor:
    """
    Wraps any probabilistic malaria classifier with Inductive Conformal Prediction.
    """
    def __init__(self, base_model, alpha: float = 0.05):
        """
        alpha: Error rate tolerance (e.g. 0.05 gives 95% confidence coverage guarantee).
        """
        self.model = base_model
        self.alpha = alpha
        self.cal_scores = None

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Computes non-conformity scores on a calibration hold-out split.
        Score = 1 - P(True Class | X)
        """
        probs = self.model.predict_proba(X_cal)
        n_samples = len(y_cal)
        
        # Calculate non-conformity score for true label
        scores = np.zeros(n_samples)
        for i, (p, y) in enumerate(zip(probs, y_cal)):
            scores[i] = 1.0 - p[int(y)]
            
        self.cal_scores = np.sort(scores)
        return self

    def predict_conformal_set(self, X_test: pd.DataFrame, alpha: float = None) -> list[dict]:
        """
        Generates conformal prediction sets for each patient instance.
        """
        if self.cal_scores is None:
            raise ValueError("Conformal predictor must be calibrated with calibrate(X_cal, y_cal) first.")

        if alpha is None:
            alpha = self.alpha

        n_cal = len(self.cal_scores)
        # Quantile index with conservative finite-sample correction
        q_idx = int(np.ceil((n_cal + 1) * (1.0 - alpha))) - 1
        q_idx = min(max(0, q_idx), n_cal - 1)
        q_threshold = self.cal_scores[q_idx]

        probs = self.model.predict_proba(X_test)
        results = []

        for i, prob in enumerate(probs):
            prob_0, prob_1 = prob[0], prob[1]
            
            # Non-conformity score if label were 0 vs 1
            score_0 = 1.0 - prob_0
            score_1 = 1.0 - prob_1
            
            prediction_set = []
            if score_0 <= q_threshold:
                prediction_set.append(0)
            if score_1 <= q_threshold:
                prediction_set.append(1)

            # Clinical interpretation
            if prediction_set == [0]:
                status = "CONFIDENT_NEGATIVE"
                guidance = "Low clinical risk. Regular outpatient management."
            elif prediction_set == [1]:
                status = "CONFIDENT_SEVERE"
                guidance = "High clinical certainty of Severe Malaria. Immediate parenteral artesunate required."
            elif prediction_set == [0, 1]:
                status = "UNCERTAIN_BORDERLINE"
                guidance = "Algorithmic uncertainty high. Mandatory urgent microscopic thin/thick blood film examination."
            else:
                status = "OUT_OF_DISTRIBUTION"
                guidance = "Atypical symptom combination. Comprehensive laboratory differential workup advised."

            results.append({
                "patient_index": i,
                "prob_severe": float(prob_1),
                "prob_negative": float(prob_0),
                "prediction_set": prediction_set,
                "set_size": len(prediction_set),
                "clinical_status": status,
                "confidence_level": f"{int((1 - alpha) * 100)}%",
                "clinical_guidance": guidance
            })

        return results

    def compute_empirical_coverage(self, X_eval: pd.DataFrame, y_eval: pd.Series, alpha: float = 0.05) -> dict:
        """
        Validates mathematical validity: empirical coverage should be >= 1 - alpha.
        """
        res = self.predict_conformal_set(X_eval, alpha=alpha)
        covered = sum(y_eval.iloc[i] in r["prediction_set"] for i, r in enumerate(res))
        empirical_coverage = covered / len(y_eval)
        avg_set_size = np.mean([r["set_size"] for r in res])
        uncertain_ratio = sum(r["set_size"] == 2 for r in res) / len(y_eval)

        return {
            "Nominal Confidence": f"{int((1 - alpha) * 100)}%",
            "Empirical Coverage": round(empirical_coverage, 4),
            "Average Set Size": round(avg_set_size, 3),
            "Uncertain Ratio (Set Size 2)": round(uncertain_ratio, 4)
        }
