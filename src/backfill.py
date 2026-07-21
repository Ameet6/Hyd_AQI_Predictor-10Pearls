"""
Pearls AQI Predictor — Historical Backfill
--------------------------------------------
Fetches ~2 years of historical weather + air quality data from Open-Meteo,
in monthly chunks, engineers features for every hour, and bulk-upserts
into MongoDB. This generates the training dataset for the model.

Run manually, once (or re-run any time to fill in missing months):
    python -m src.backfill
"""

import sys
from datetime import datetime, timedelta, timezone

from pymongo import UpdateOne
from pymongo.errors import PyMongoError
import requests

from src import config, db
from src.features.fetch import fetch_historical_weather, fetch_historical_air_quality
from src.features.engineer import build_feature_document

# How far back to backfill.
BACKFILL_DAYS = 730  # ~2 years
CHUNK_DAYS = 30  # fetch one month at a time


def month_chunks(start: datetime, end: datetime, chunk_days: int):
    """Yield (chunk_start, chunk_end) date pairs covering start..end."""
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        yield current, chunk_end
        current = chunk_end


def build_docs_for_chunk(weather_hourly: dict, air_quality_hourly: dict, running_previous_aqi):
    """
    Zip weather + air quality hourly arrays together by index (they share
    the same 'time' array since we requested the same date range), and
    build one feature document per hour.

    Returns (list_of_docs, updated_previous_aqi) so the caller can carry
    the last AQI value into the next chunk for a continuous change-rate.
    """
    docs = []
    times = weather_hourly["time"]  # e.g. "2024-07-21T00:00"
    previous_aqi = running_previous_aqi

    for i, time_str in enumerate(times):
        timestamp = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)

        weather_at_i = {
            "temperature_2m": weather_hourly["temperature_2m"][i],
            "relative_humidity_2m": weather_hourly["relative_humidity_2m"][i],
            "surface_pressure": weather_hourly["surface_pressure"][i],
            "wind_speed_10m": weather_hourly["wind_speed_10m"][i],
        }
        air_quality_at_i = {
            "us_aqi": air_quality_hourly["us_aqi"][i],
            "pm2_5": air_quality_hourly["pm2_5"][i],
            "pm10": air_quality_hourly["pm10"][i],
            "carbon_monoxide": air_quality_hourly["carbon_monoxide"][i],
            "nitrogen_dioxide": air_quality_hourly["nitrogen_dioxide"][i],
            "sulphur_dioxide": air_quality_hourly["sulphur_dioxide"][i],
            "ozone": air_quality_hourly["ozone"][i],
            "ammonia": air_quality_hourly["ammonia"][i],
        }

        # Skip hours where AQI is missing (can't train on a row with no target).
        if air_quality_at_i["us_aqi"] is None:
            continue

        doc = build_feature_document(weather_at_i, air_quality_at_i, timestamp, previous_aqi)
        docs.append(doc)
        previous_aqi = doc["aqi"]

    return docs, previous_aqi


def store_docs(docs: list, collection):
    """Bulk upsert on (city, hour_bucket) — same dedup key as the live pipeline."""
    if not docs:
        return 0

    operations = []
    for doc in docs:
        hour_bucket = doc["timestamp"].replace(minute=0, second=0, microsecond=0)
        doc["hour_bucket"] = hour_bucket
        filter_key = {"city": doc["city"], "hour_bucket": hour_bucket}
        operations.append(UpdateOne(filter_key, {"$set": doc}, upsert=True))

    result = collection.bulk_write(operations)
    return result.upserted_count + result.modified_count


def run():
    config.validate()
    client = db.get_client()
    try:
        collection = db.get_collection(config.FEATURES_COLLECTION, client)
        collection.create_index([("city", 1), ("hour_bucket", 1)], unique=True)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=BACKFILL_DAYS)

        previous_aqi = None
        total_written = 0

        for chunk_start, chunk_end in month_chunks(start_date, end_date, CHUNK_DAYS):
            start_str = chunk_start.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")
            print(f"Fetching {start_str} to {end_str}...")

            weather_hourly = fetch_historical_weather(config.LAT, config.LON, start_str, end_str)
            air_quality_hourly = fetch_historical_air_quality(config.LAT, config.LON, start_str, end_str)

            docs, previous_aqi = build_docs_for_chunk(weather_hourly, air_quality_hourly, previous_aqi)
            written = store_docs(docs, collection)
            total_written += written
            print(f"  -> {len(docs)} hours processed, {written} written to MongoDB")

        print(f"Backfill complete. Total rows written: {total_written}")

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}", file=sys.stderr)
        sys.exit(1)
    except PyMongoError as e:
        print(f"MongoDB error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    run()