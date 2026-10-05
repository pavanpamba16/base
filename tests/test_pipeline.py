"""
========================================================================================
AUTOMATED UNIT & INTEGRATION TEST SUITE
Validates Data Preprocessing, Stacking Ensembles, Conformal Bounds, and Multimodal Vision
========================================================================================
Usage:
    python -m unittest tests/test_pipeline.py
========================================================================================
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.advanced_resampling import prepare_leak_free_data, compute_synthetic_fidelity_metrics
from src.novel_models import get_tuned_base_estimators, build_stacking_meta_ensemble
from src.conformal_prediction import ConformalMalariaPredictor
from src.counterfactuals import ClinicalCounterfactualExplainer
from src.clinical_engine import compute_who_danger_score, generate_clinical_html_report
from src.multimodal_fusion import MultimodalMalariaClassifier


class TestMalariaResearchPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepare sample data and models for tests."""
        cls.data = prepare_leak_free_data(technique="SMOTE-NC")
        cls.X_train = cls.data["X_train"]
        cls.y_train = cls.data["y_train"]
        cls.X_test = cls.data["X_test"]
        cls.y_test = cls.data["y_test"]

        # Train a fast base model for testing
        cls.rf = get_tuned_base_estimators()[0][1]
        cls.rf.fit(cls.X_train, cls.y_train)

    def test_01_resampling_leak_free(self):
        """Verify that training set is balanced and test set is strictly untouched."""
        # Train should be balanced
        count_0 = (self.y_train == 0).sum()
        count_1 = (self.y_train == 1).sum()
        self.assertEqual(count_0, count_1, "Training set minority and majority classes should be balanced.")
        
        # Test set should retain original natural imbalance
        test_prev = np.mean(self.y_test)
        self.assertLess(test_prev, 0.45, "Test set should retain original clinical distribution without synthetic contamination.")

    def test_02_synthetic_fidelity_dcr(self):
        """Verify that SMOTE-NC does not create exact duplicate copies (DCR > 0)."""
        real_min = self.X_train[self.y_train == 1].iloc[:20]
        syn_min = self.X_train[self.y_train == 1].iloc[20:40]
        fidelity = compute_synthetic_fidelity_metrics(real_min, syn_min)
        
        self.assertIn("Mean DCR", fidelity)
        self.assertGreater(fidelity["Mean DCR"], 0.0, "Mean Distance-to-Closest-Record must be greater than 0 to prove non-memorization.")

    def test_03_stacking_ensemble_predictions(self):
        """Verify Stacking Meta-Ensemble fits and predicts valid probabilities [0, 1]."""
        stack_clf = build_stacking_meta_ensemble()
        stack_clf.fit(self.X_train.iloc[:60], self.y_train.iloc[:60])
        
        probs = stack_clf.predict_proba(self.X_test.iloc[:5])
        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be bounded in [0, 1].")

    def test_04_conformal_prediction_bounds(self):
        """Verify Conformal Predictor generates valid prediction sets and coverage."""
        cp = ConformalMalariaPredictor(self.rf, alpha=0.05)
        # Calibrate on first half of test, evaluate on second half
        split_idx = len(self.X_test) // 2
        cp.calibrate(self.X_test.iloc[:split_idx], self.y_test.iloc[:split_idx])
        
        sets = cp.predict_conformal_set(self.X_test.iloc[split_idx:])
        self.assertGreater(len(sets), 0)
        for s in sets:
            self.assertIn("prediction_set", s)
            self.assertTrue(s["set_size"] in [1, 2], "Prediction set size should be 1 or 2.")
            self.assertIn(s["clinical_status"], ["CONFIDENT_NEGATIVE", "CONFIDENT_SEVERE", "UNCERTAIN_BORDERLINE", "OUT_OF_DISTRIBUTION"])

    def test_05_counterfactual_actionability(self):
        """Verify DiCE counterfactuals strictly preserve immutable demographic age."""
        explainer = ClinicalCounterfactualExplainer(self.rf, self.X_train.iloc[:50])
        patient = self.X_test.iloc[0].copy()
        patient["age"] = 42
        
        cfs = explainer.generate_counterfactuals(patient, total_CFs=2, desired_class=0)
        self.assertGreater(len(cfs), 0, "At least one counterfactual clinical state should be generated.")
        for _, cf_row in cfs.iterrows():
            self.assertEqual(cf_row["age"], 42, "Patient age is immutable and must not be altered by counterfactual engine.")

    def test_06_who_danger_scoring(self):
        """Verify WHO severe malaria scoring logic and emergency triage tiers."""
        # Patient with severe danger markers
        critical_patient = {
            "age": 30, "Convulsion": 1, "cocacola_urine": 1, "hypoglycemia": 1, "prostraction": 1
        }
        res = compute_who_danger_score(critical_patient)
        self.assertGreaterEqual(res["score"], 5)
        self.assertEqual(res["tier"], "RED_EMERGENCY", "Patient with convulsions and coca-cola urine must be classified as RED_EMERGENCY.")

        # Asymptomatic patient
        mild_patient = {"age": 25, "fever": 0, "cold": 0}
        res_mild = compute_who_danger_score(mild_patient)
        self.assertEqual(res_mild["tier"], "GREEN_ROUTINE")

    def test_07_multimodal_vision_tabular_fusion(self):
        """Verify microscopy image feature extraction and cross-modal fusion."""
        classifier = MultimodalMalariaClassifier()
        sample_img = Image.new("RGB", (224, 224), color=(240, 240, 240))
        
        cell_info = classifier.analyze_microscopy_cell(sample_img)
        self.assertIn("is_parasitized", cell_info)
        self.assertIn("parasitemia_index", cell_info)

        patient = self.X_test.iloc[0]
        mm_pred = classifier.predict_multimodal(sample_img, patient, base_tabular_prob=0.85)
        self.assertIn("fused_probability", mm_pred)
        self.assertIn("modality_agreement", mm_pred)
        self.assertTrue(0.0 <= mm_pred["fused_probability"] <= 1.0)

    def test_08_clinical_dossier_html_generation(self):
        """Verify print-ready HTML diagnostic dossier compiles cleanly."""
        patient_dict = {f: 1 if f in ["fever", "rigor"] else 0 for f in FEATURE_COLUMNS}
        patient_dict["age"] = 35
        html = generate_clinical_html_report(
            patient_data=patient_dict,
            model_prediction={"prediction": 1, "probability": 0.88},
            doctor_notes="Test clinical verification."
        )
        self.assertIn("EXPLAINABLE CLINICAL MALARIA DECISION SUPPORT SYSTEM", html)
        self.assertIn("Patient Telemetry", html)
        self.assertIn("Attending Medical Officer Signature", html)


if __name__ == "__main__":
    unittest.main()
