"""Unit tests for the deterministic travel policy manager."""

from __future__ import annotations

import unittest

from host_agent.policy_manager import TravelPolicyManager


class TravelPolicyManagerTest(unittest.TestCase):
    """Ensure the policy heuristics behave predictably."""

    def setUp(self) -> None:
        self.manager = TravelPolicyManager()

    def test_classify_rental_request_requires_weather(self) -> None:
        classification = self.manager.classify_request(
            "Plan a long weekend stay in a hotel in Denver"
        )
        self.assertTrue(classification.need_rentals)
        self.assertTrue(
            classification.need_weather,
            "Rental planning should enforce a weather check",
        )
        self.assertEqual(classification.location_hint, "Denver")

    def test_classify_weather_only_request(self) -> None:
        classification = self.manager.classify_request("What is the weather in Austin?")
        self.assertFalse(classification.need_rentals)
        self.assertTrue(classification.need_weather)
        self.assertEqual(classification.location_hint, "Austin")

    def test_should_block_rentals_on_hazardous_forecast(self) -> None:
        forecast = "Severe storm warning with dangerous winds expected."
        self.assertTrue(self.manager.should_block_rentals(forecast))
        self.assertFalse(
            self.manager.should_block_rentals("Clear skies and warm sunshine all weekend")
        )


if __name__ == "__main__":
    unittest.main()
