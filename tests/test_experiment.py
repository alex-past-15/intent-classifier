import unittest
import numpy as np
from experiment import choose_threshold, normalize, selective_metrics


class EvaluationTests(unittest.TestCase):
    def test_abstains_when_target_cannot_be_reached(self):
        p = np.array([[.9, .1], [.8, .2]])
        threshold = choose_threshold(["b", "b"], p, ["a", "b"], min_accepted=1)
        self.assertIsNone(threshold)
        result = selective_metrics(["b", "b"], p, ["a", "b"], threshold)
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["accepted_accuracy"])

    def test_threshold_maximizes_coverage_and_respects_ties(self):
        p = np.array([[.9, .1], [.9, .1], [.6, .4], [.55, .45]])
        self.assertEqual(choose_threshold(["a", "a", "b", "b"], p, ["a", "b"], min_accepted=2), .9)
        self.assertIsNone(choose_threshold(["a", "b", "b", "b"], p, ["a", "b"], min_accepted=2))

    def test_minimum_sample_count(self):
        self.assertIsNone(choose_threshold(["a"], np.array([[.99, .01]]), ["a", "b"]))

    def test_duplicate_normalization(self):
        self.assertEqual(normalize(" WHERE is my card?!"), normalize("where is my card"))


if __name__ == "__main__":
    unittest.main()
