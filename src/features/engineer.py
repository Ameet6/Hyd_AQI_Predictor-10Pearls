"""
Feature engineering.
Takes raw Open-Meteo dicts (already fetched) and turns them into one
feature document. No network calls, no DB calls in here — that's what
makes it easy to unit test: feed it fake weather/air_quality dicts,
check the output.
"""

from datetime import datetime
from src import config


def build_feature_document(
    weather: dict,
    air_quality: dict,
    timestamp: datetime,
    previous_aqi: float = None,
) -> dict:
    """
    Combine raw weather + air quality dicts (Open-Meteo format) into one
    feature document.

    previous_aqi: the US AQI value from the last stored reading for this
    city, or None if this is the first reading ever. Passed in rather than
    looked up here so this function stays free of any MongoDB dependency.
    """
    current_aqi = air_quality["us_aqi"]
    aqi_change_rate = 0 if previous_aqi is None else current_aqi - previous_aqi

    doc = {
        "city": config.CITY_NAME,
        "lat": config.LAT,
        "lon": config.LON,
        "timestamp": timestamp,

        # --- pollutant features ---
        "aqi": current_aqi,  # US AQI scale (0-500)
        "pm2_5": air_quality.get("pm2_5"),
        "pm10": air_quality.get("pm10"),
        "co": air_quality.get("carbon_monoxide"),
        "no2": air_quality.get("nitrogen_dioxide"),
        "so2": air_quality.get("sulphur_dioxide"),
        "o3": air_quality.get("ozone"),
        "nh3": air_quality.get("ammonia"),

        # --- weather features (affect pollutant dispersion) ---
        "temperature": weather.get("temperature_2m"),
        "humidity": weather.get("relative_humidity_2m"),
        "pressure": weather.get("surface_pressure"),
        "wind_speed": weather.get("wind_speed_10m"),

        # --- time-based features ---
        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,
        "day_of_week": timestamp.weekday(),  # 0 = Monday

        # --- derived feature ---
        "aqi_change_rate": aqi_change_rate,
    }
    return doc