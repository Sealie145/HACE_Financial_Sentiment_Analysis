"""API contract tests for the HACE mock-demo backend."""

import unittest

from fastapi.testclient import TestClient

from backend.main import app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_root_and_health(self) -> None:
        root = self.client.get("/")
        health = self.client.get("/health")
        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.json()["status"], "running")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

    def test_predict_returns_real_hedging_and_four_mock_experts(self) -> None:
        response = self.client.post("/predict", json={"text": "The company may report stronger earnings."})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text"], "The company may report stronger earnings.")
        self.assertIn("sentiment", payload)
        self.assertIn("confidence", payload)
        self.assertIn("hedging", payload)
        self.assertIn("experts", payload)
        self.assertEqual(payload["hedging"]["detected_terms"], ["may"])
        self.assertTrue(payload["hedging"]["hedge_flag"])
        self.assertGreaterEqual(payload["hedging"]["hedge_probability"], 0.0)
        self.assertLessEqual(payload["hedging"]["hedge_probability"], 1.0)
        self.assertEqual(set(payload["experts"]), {"fiqa", "twitter", "phrasebank", "finance_news"})

    def test_empty_text_is_rejected(self) -> None:
        self.assertEqual(self.client.post("/predict", json={"text": "   "}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
