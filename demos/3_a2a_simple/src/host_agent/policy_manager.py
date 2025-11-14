"""Policy utilities used by the host agent LangGraph."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyClassification:
    """Structured result describing how the host should react to a message."""

    need_weather: bool
    need_rentals: bool
    location_hint: Optional[str]
    note: Optional[str]


class TravelPolicyManager:
    """Encapsulates deterministic policy checks for the host agent.

    The host agent enforces a simple but illustrative policy: when users ask
    about hotel rentals we must first obtain a weather assessment for the
    destination. Hazardous forecasts block downstream rental suggestions.
    """

    WEATHER_KEYWORDS = (
        "weather",
        "forecast",
        "temperature",
        "rain",
        "snow",
    )
    RENTAL_KEYWORDS = (
        "hotel",
        "stay",
        "rental",
        "lodging",
        "apartment",
        "condo",
        "cabin",
    )
    HAZARD_KEYWORDS = (
        "storm",
        "warning",
        "advisory",
        "hazard",
        "flood",
        "blizzard",
        "hurricane",
        "tornado",
        "heat wave",
        "dangerous",
    )

    _LOCATION_PATTERN = re.compile(
        r"\b(?:in|near|around|for)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})"
    )

    def classify_request(self, message: str) -> PolicyClassification:
        """Categorise a user query into weather and rental intents.

        Args:
            message: Raw user utterance.

        Returns:
            PolicyClassification conveying whether weather and rental agents
            should be consulted along with a short explanatory note.
        """

        lowered = message.lower()
        mentions_weather = any(keyword in lowered for keyword in self.WEATHER_KEYWORDS)
        mentions_rental = any(keyword in lowered for keyword in self.RENTAL_KEYWORDS)

        need_rentals = mentions_rental
        # The policy requires weather checks whenever we consider rentals.
        need_weather = mentions_weather or need_rentals

        note: Optional[str] = None
        if need_rentals:
            note = (
                "Policy: hotel planning must include a fresh weather review "
                "before sharing listings."
            )
        elif need_weather:
            note = "Policy: provide a concise weather outlook from the specialist."

        location_hint = self._extract_location_hint(message)

        return PolicyClassification(
            need_weather=need_weather,
            need_rentals=need_rentals,
            location_hint=location_hint,
            note=note,
        )

    def should_block_rentals(self, weather_report: str) -> bool:
        """Return ``True`` when the policy detects hazardous weather terms."""

        lowered = weather_report.lower()
        return any(keyword in lowered for keyword in self.HAZARD_KEYWORDS)

    def _extract_location_hint(self, message: str) -> Optional[str]:
        """Extract a lightweight location hint from the user message."""

        match = self._LOCATION_PATTERN.search(message)
        if match:
            return match.group(1)
        return None
