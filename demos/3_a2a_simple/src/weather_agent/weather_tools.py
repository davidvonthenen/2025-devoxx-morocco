"""Utility functions for querying weather.gov data and formatting responses."""

from __future__ import annotations

import json
from typing import Any

import httpx
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

BASE_URL = "https://api.weather.gov"
USER_AGENT = "weather-agent"
# Weather.gov endpoints occasionally stall when the upstream service is busy.
# Use a generous timeout so long-running requests can complete instead of
# raising client-side timeout errors.
REQUEST_TIMEOUT = 120.0
GEOCODE_TIMEOUT = 10.0

_http_client = httpx.AsyncClient(
    base_url=BASE_URL,
    headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
    timeout=REQUEST_TIMEOUT,
    follow_redirects=True,
)
_geolocator = Nominatim(user_agent=USER_AGENT)


async def close_client() -> None:
    """Release the shared HTTP client resources."""
    await _http_client.aclose()


async def get_alerts(state: str) -> str:
    if not isinstance(state, str) or len(state) != 2 or not state.isalpha():
        return "Invalid input. Please provide a two-letter US state code (e.g., CA)."

    endpoint = f"/alerts/active/area/{state.upper()}"
    data = await _get_weather_response(endpoint)

    if data is None:
        return f"Failed to retrieve weather alerts for {state.upper()}."

    features = data.get("features")
    if not features:
        return f"No active weather alerts found for {state.upper()}."

    alerts = [format_alert(feature) for feature in features]
    return "\n---\n".join(alerts)


async def get_forecast(latitude: float, longitude: float) -> str:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return (
            "Invalid latitude or longitude provided. Latitude must be between -90 and 90, "
            "Longitude between -180 and 180."
        )

    point_endpoint = f"/points/{latitude:.4f},{longitude:.4f}"
    points_data = await _get_weather_response(point_endpoint)

    if points_data is None or "properties" not in points_data:
        return (
            f"Unable to retrieve NWS gridpoint information for {latitude:.4f},{longitude:.4f}."
        )

    forecast_url = points_data["properties"].get("forecast")
    if not forecast_url:
        return (
            f"Could not find the NWS forecast endpoint for {latitude:.4f},{longitude:.4f}."
        )

    forecast_data = None
    try:
        response = await _http_client.get(forecast_url)
        response.raise_for_status()
        forecast_data = response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        forecast_data = None

    if forecast_data is None or "properties" not in forecast_data:
        return "Failed to retrieve detailed forecast data from NWS."

    periods = forecast_data["properties"].get("periods")
    if not periods:
        return "No forecast periods found for this location from NWS."

    forecasts = [format_forecast_period(period) for period in periods[:5]]
    return "\n---\n".join(forecasts)


async def get_forecast_by_city(city: str, state: str) -> str:
    if not city or not isinstance(city, str):
        return "Invalid city name provided."
    if not state or not isinstance(state, str) or len(state) != 2 or not state.isalpha():
        return "Invalid state code. Please provide the two-letter US state abbreviation (e.g., CA)."

    query = f"{city.strip()}, {state.strip().upper()}, USA"

    try:
        location = _geolocator.geocode(query, timeout=GEOCODE_TIMEOUT)
    except GeocoderTimedOut:
        return (
            f"Could not get coordinates for '{city.strip()}, {state.strip().upper()}': "
            "The location service timed out."
        )
    except GeocoderServiceError:
        return (
            f"Could not get coordinates for '{city.strip()}, {state.strip().upper()}': "
            "The location service returned an error."
        )
    except Exception:
        return (
            f"An unexpected error occurred while finding coordinates for "
            f"'{city.strip()}, {state.strip().upper()}'."
        )

    if location is None:
        return (
            f"Could not find coordinates for '{city.strip()}, {state.strip().upper()}'. "
            "Please check the spelling or try a nearby city."
        )

    return await get_forecast(location.latitude, location.longitude)


async def _get_weather_response(endpoint: str) -> dict[str, Any] | None:
    try:
        response = await _http_client.get(endpoint)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


def format_alert(feature: dict[str, Any]) -> str:
    props = feature.get("properties", {})
    return (
        "Event: {event}\n"
        "Area: {area}\n"
        "Severity: {severity}\n"
        "Certainty: {certainty}\n"
        "Urgency: {urgency}\n"
        "Effective: {effective}\n"
        "Expires: {expires}\n"
        "Description: {description}\n"
        "Instructions: {instructions}"
    ).format(
        event=props.get("event", "Unknown Event"),
        area=props.get("areaDesc", "N/A"),
        severity=props.get("severity", "N/A"),
        certainty=props.get("certainty", "N/A"),
        urgency=props.get("urgency", "N/A"),
        effective=props.get("effective", "N/A"),
        expires=props.get("expires", "N/A"),
        description=props.get("description", "No description provided.").strip(),
        instructions=props.get("instruction", "No instructions provided.").strip(),
    )


def format_forecast_period(period: dict[str, Any]) -> str:
    return (
        f"{period.get('name', 'Unknown Period')}:\n"
        f"  Temperature: {period.get('temperature', 'N/A')}°{period.get('temperatureUnit', 'F')}\n"
        f"  Wind: {period.get('windSpeed', 'N/A')} {period.get('windDirection', 'N/A')}\n"
        f"  Short Forecast: {period.get('shortForecast', 'N/A')}\n"
        f"  Detailed Forecast: {period.get('detailedForecast', 'No detailed forecast provided.').strip()}"
    )
