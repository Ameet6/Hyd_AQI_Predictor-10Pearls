"""
Raw data fetching from Open-Meteo.
Covers both current/live data and historical data, for both weather and
air quality, from one provider — no API key required.
Deliberately does NOT know about MongoDB or feature engineering.
"""

import requests

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_FIELDS = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
AIR_QUALITY_FIELDS = "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,ammonia,us_aqi"


def fetch_current_weather(lat: float, lon: float) -> dict:
    params = {"latitude": lat, "longitude": lon, "current": WEATHER_FIELDS}
    resp = requests.get(WEATHER_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["current"]


def fetch_current_air_quality(lat: float, lon: float) -> dict:
    params = {"latitude": lat, "longitude": lon, "current": AIR_QUALITY_FIELDS}
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["current"]


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """start_date/end_date format: 'YYYY-MM-DD'. Returns hourly arrays."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_FIELDS,
    }
    resp = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["hourly"]


def fetch_historical_air_quality(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """start_date/end_date format: 'YYYY-MM-DD'. Returns hourly arrays."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": AIR_QUALITY_FIELDS,
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["hourly"]