"""Dependency-free tests for the deterministic HACE hedge detector."""

import unittest

from src.hedging.detector import HedgeDetector
from src.hedging.features import HedgeFeatureExtractor


class HedgeDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = HedgeDetector()

    def test_required_hedge_cases(self) -> None:
        cases = [
            ("The company may report stronger earnings.", ["may"]),
            ("The company might experience weaker growth.", ["might"]),
            ("The results suggest that revenue could increase.", ["suggest", "could"]),
            ("Revenue was approximately $2 billion.", ["approximately"]),
            ("Depending on market conditions, earnings could improve.", ["depending on", "could"]),
            ("The company is subject to regulatory approval.", ["subject to"]),
        ]
        for text, terms in cases:
            with self.subTest(text=text):
                result = self.detector.detect(text)
                self.assertTrue(result["hedge_flag"])
                self.assertEqual(result["detected_terms"], terms)
                self.assertEqual(result["hedge_count"], len(terms))

        result = self.detector.detect("This may be a good investment.")
        self.assertTrue(result.hedge_flag)
        self.assertIn("may", result.detected_terms)

    def test_empty_and_non_hedged_text(self) -> None:
        self.assertEqual(self.detector.detect("The company reported record revenue.").to_dict(), {
            "hedge_flag": False, "hedge_probability": 0.0,
            "detected_terms": [], "hedge_count": 0,
        })
        self.assertFalse(self.detector.detect("   ").hedge_flag)

    def test_word_boundary_case_and_punctuation(self) -> None:
        result = self.detector.detect("The word maybe should not match the cue may.")
        self.assertEqual(result.detected_terms, ["may"])
        self.assertNotIn("maybe", result.detected_terms)
        self.assertFalse(self.detector.detect("Maybe revenue rises.").hedge_flag)
        self.assertEqual(
            self.detector.detect("MAY, potentially; improve.").detected_terms,
            ["may", "potentially"],
        )

    def test_duplicate_cues_and_probability(self) -> None:
        self.assertEqual(self.detector.detect("It may, and may not, improve.").hedge_count, 2)
        text = "The company may potentially appear subject to delays."
        result = self.detector.detect(text)
        self.assertEqual(result, self.detector.detect(text))
        self.assertGreaterEqual(result.hedge_probability, 0.0)
        self.assertLessEqual(result.hedge_probability, 1.0)

    def test_invalid_input_and_feature_extractor(self) -> None:
        with self.assertRaisesRegex(TypeError, "text must be a string"):
            self.detector.detect(None)  # type: ignore[arg-type]
        self.assertEqual(HedgeFeatureExtractor().extract("Revenue may increase."), {
            "hedge_flag": 1, "hedge_probability": 0.3, "hedge_count": 1,
        })


if __name__ == "__main__":
    unittest.main()
