"""
Pearls AQI Predictor — Feature Pipeline
----------------------------------------
Orchestrates: fetch raw data -> engineer features -> upsert into MongoDB.

Run manually for now:
    python -m src.features.pipeline

Later, GitHub Actions runs this every hour.
"""

import sys
from datetime import datetime, timezone

from pymongo.errors import PyMongoError
import requests

from src import config, db
from src.features.fetch import fetch_current_weather, fetch_current_air_quality
from src.features.engineer import build_feature_document


def get_previous_aqi(collection, city: str):
    """Look up the most recent stored reading for this city, if any."""
    previous = collection.find_one({"city": city}, sort=[("timestamp", -1)])
    return previous["aqi"] if previous else None


def store_feature_document(doc: dict, collection):
    """
    Upsert on (city, hour_bucket) so re-running within the same hour
    doesn't create duplicate rows.
    """
    hour_bucket = doc["timestamp"].replace(minute=0, second=0, microsecond=0)
    doc["hour_bucket"] = hour_bucket
    filter_key = {"city": doc["city"], "hour_bucket": hour_bucket}

    result = collection.update_one(filter_key, {"$set": doc}, upsert=True)
    if result.upserted_id:
        print(f"Inserted new feature row for {doc['city']} at {hour_bucket}")
    else:
        print(f"Updated existing feature row for {doc['city']} at {hour_bucket}")


def run():
    config.validate()
    client = db.get_client()
    try:
        collection = db.get_collection(config.FEATURES_COLLECTION, client)
        collection.create_index([("city", 1), ("hour_bucket", 1)], unique=True)

        weather = fetch_current_weather(config.LAT, config.LON)
        air_quality = fetch_current_air_quality(config.LAT, config.LON)

        previous_aqi = get_previous_aqi(collection, config.CITY_NAME)
        now = datetime.now(timezone.utc)
        doc = build_feature_document(weather, air_quality, now, previous_aqi)

        store_feature_document(doc, collection)
        print("Feature pipeline run complete.")
        print(doc)

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