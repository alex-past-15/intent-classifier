"""Проверяем сохранённые модели. Без данных и моделей эти проверки пропускаются."""
import unittest

import pandas as pd

from experiment import DATA, REPORTS
from predict import Predictor


class SavedModelTests(unittest.TestCase):
    @unittest.skipUnless((REPORTS / "bert_tiny.json").exists(), "Train both models first")
    def test_saved_inference_matches_recorded_test_predictions(self):
        for name in ["tfidf", "bert_tiny"]:
            predictor = Predictor(name)
            sample = pd.read_csv(REPORTS / f"{name}_predictions.csv").iloc[0]
            result = predictor.predict(sample.text)
            self.assertEqual(result["top3"][0]["label"], sample.predicted)
            self.assertAlmostEqual(result["top3"][0]["score"], sample.confidence, places=5)
            self.assertEqual(result["decision"] == "automatic", bool(sample.accepted))

    @unittest.skipUnless((DATA / "test.csv").exists(), "Prepare data first")
    def test_actual_splits_have_no_normalized_overlap(self):
        from itertools import combinations
        frames = {s: pd.read_csv(DATA / f"{s}.csv") for s in ["train", "validation", "calibration", "test"]}
        for a, b in combinations(frames, 2):
            self.assertTrue(set(frames[a].key).isdisjoint(frames[b].key))
        self.assertEqual(len(frames["test"]), 3080)
        self.assertTrue(all(f.label.nunique() == 77 for f in frames.values()))

    @unittest.skipUnless((REPORTS / "tfidf.json").exists(), "Train first")
    def test_demo_runs_and_predicts(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("lost_or_stolen_card" in s.value for s in app.success))


if __name__ == "__main__":
    unittest.main()
