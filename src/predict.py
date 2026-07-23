"""
Prediction helpers.
Loads the active model for a given horizon from the MongoDB model registry,
and generates predictions from the latest feature row. Used by the
dashboard (and reusable for any other script that needs predictions).
"""

import io
import joblib
import pandas as pd

from src import config, db

HORIZONS_HOURS = [24, 48, 72]


def get_latest_features(collection) -> dict:
    """Fetch the single most recent feature document for our city."""
    doc = collection.find_one({"city": config.CITY_NAME}, sort=[("timestamp", -1)])
    return doc


def load_active_model(horizon_hours: int, models_collection) -> dict:
    """
    Fetch the currently active model + its metadata for a given horizon.
    Returns None if no active model exists yet for that horizon (e.g.
    training pipeline hasn't run yet).
    """
    doc = models_collection.find_one({
        "city": config.CITY_NAME,
        "horizon_hours": horizon_hours,
        "is_active": True,
    })
    if doc is None:
        return None

    model = joblib.load(io.BytesIO(doc["model_binary"]))
    return {
        "model": model,
        "algorithm": doc["algorithm"],
        "feature_cols": doc["feature_cols"],
        "metrics": doc["metrics"],
        "trained_at": doc["trained_at"],
    }


def predict_all_horizons(latest_features: dict, models_collection) -> list:
    """
    Run all 3 horizon models on the latest feature row.
    Returns a list of dicts: [{horizon_hours, day, predicted_aqi, algorithm, metrics}, ...]
    """
    results = []
    for horizon_hours in HORIZONS_HOURS:
        model_info = load_active_model(horizon_hours, models_collection)
        if model_info is None:
            results.append({
                "horizon_hours": horizon_hours,
                "day": horizon_hours // 24,
                "predicted_aqi": None,
                "algorithm": None,
                "metrics": None,
            })
            continue

        # Build a single-row DataFrame matching the model's expected feature columns
        X = pd.DataFrame([{col: latest_features.get(col) for col in model_info["feature_cols"]}])
        prediction = model_info["model"].predict(X)[0]

        results.append({
            "horizon_hours": horizon_hours,
            "day": horizon_hours // 24,
            "predicted_aqi": round(float(prediction), 1),
            "algorithm": model_info["algorithm"],
            "metrics": model_info["metrics"],
        })

    return results


def aqi_category(aqi_value: float) -> tuple:
    """
    Map a US AQI value to its official health category, a display color,
    and a matching text color that stays legible on that background.
    Returns (label, bg_color, text_color).
    """
    if aqi_value is None:
        return ("Unknown", "#94a3b8", "#ffffff")
    if aqi_value <= 50:
        return ("Good", "#16a34a", "#ffffff")
    if aqi_value <= 100:
        return ("Moderate", "#eab308", "#1a2332")  # amber needs dark text, not white
    if aqi_value <= 150:
        return ("Unhealthy for Sensitive Groups", "#f97316", "#ffffff")
    if aqi_value <= 200:
        return ("Unhealthy", "#dc2626", "#ffffff")
    if aqi_value <= 300:
        return ("Very Unhealthy", "#9333ea", "#ffffff")
    return ("Hazardous", "#7f1d1d", "#ffffff")