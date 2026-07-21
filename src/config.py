"""
Central config for the whole project.
Every other script imports from here instead of reading os.environ directly,
so there's exactly one place that knows about secrets and constants.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Target city — change these three lines to forecast a different city.
CITY_NAME = "Hyderabad"
LAT = 25.3960
LON = 68.3578

DB_NAME = "aqi_predictor"
FEATURES_COLLECTION = "features"
MODELS_COLLECTION = "models"  # used later by the training pipeline


def validate():
    """Call this at the top of any script that needs the DB to work."""
    if not MONGO_URI:
        sys.exit(
            "Missing MONGO_URI. Fill in your real value in .env before running this script."
        )