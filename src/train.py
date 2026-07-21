"""
Pearls AQI Predictor — Training Pipeline
-------------------------------------------
Loads feature data from MongoDB, builds a 72-hour-ahead AQI target,
does a time-based train/test split, trains multiple models, evaluates
with RMSE/MAE/R2, and saves the best model to the model registry:
  - the model binary and metadata -> MongoDB "models" collection

Run manually for now:
    python -m src.train

Later, GitHub Actions runs this daily.
"""

import io
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bson.binary import Binary
from pymongo.errors import PyMongoError

from src import config, db

HORIZON_HOURS = 72          # how far ahead we're forecasting (3 days)
TRAIN_SPLIT_RATIO = 0.85    # fraction of the timeline used for training

FEATURE_COLS = [
    "aqi", "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day", "month", "day_of_week", "aqi_change_rate",
]


def load_data(collection) -> pd.DataFrame:
    """Pull all feature documents for our city, sorted chronologically."""
    cursor = collection.find({"city": config.CITY_NAME}).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))
    df = df.drop(columns=[c for c in ["nh3", "_id"] if c in df.columns])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_target_and_split(df: pd.DataFrame):
    """Shift AQI forward by HORIZON_HOURS to create the target, then split
    chronologically (train = earlier data, test = later/unseen data)."""
    df["aqi_target"] = df["aqi"].shift(-HORIZON_HOURS)
    df = df.dropna(subset=["aqi_target"]).reset_index(drop=True)

    split_index = int(len(df) * TRAIN_SPLIT_RATIO)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train each candidate model, evaluate on the test set, return results."""
    X_train, y_train = train_df[FEATURE_COLS], train_df["aqi_target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["aqi_target"]

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "ridge_regression": Ridge(alpha=1.0),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
    }

    results = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
        }
        print(f"{name}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}")
        results.append({"name": name, "model": model, "metrics": metrics})

    return results


def select_best(results: list) -> dict:
    """Pick the model with the highest R2 (higher = explains more variance = better)."""
    return max(results, key=lambda r: r["metrics"]["r2"])


def save_model_registry(best: dict, models_collection):
    """
    Serialize the model to bytes in memory (no local file), and store both
    the model bytes and its metadata as one MongoDB document. This means
    the model is reachable from anywhere — your laptop, GitHub Actions,
    or the dashboard — since it lives in the database, not on disk.
    """
    buffer = io.BytesIO()
    joblib.dump(best["model"], buffer)
    model_bytes = buffer.getvalue()

    trained_at = datetime.now(timezone.utc)

    metadata = {
        "city": config.CITY_NAME,
        "algorithm": best["name"],
        "trained_at": trained_at,
        "horizon_hours": HORIZON_HOURS,
        "feature_cols": FEATURE_COLS,
        "metrics": best["metrics"],
        "model_binary": Binary(model_bytes),
        "is_active": True,  # this is now the "current" model to use for predictions
    }

    # Deactivate any previously active model for this city, so there's
    # only ever one "is_active: True" model at a time — the dashboard
    # will always query for that one.
    models_collection.update_many(
        {"city": config.CITY_NAME, "is_active": True},
        {"$set": {"is_active": False}},
    )
    models_collection.insert_one(metadata)

    size_kb = len(model_bytes) / 1024
    print(f"Model serialized: {size_kb:.1f} KB")
    print(f"Registered in MongoDB 'models' collection as active model for {config.CITY_NAME}")


def run():
    print("Starting training pipeline...")
    config.validate()
    client = db.get_client()
    try:
        features_collection = db.get_collection(config.FEATURES_COLLECTION, client)
        models_collection = db.get_collection(config.MODELS_COLLECTION, client)

        df = load_data(features_collection)
        print(f"Loaded {len(df)} rows")

        train_df, test_df = build_target_and_split(df)
        print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

        results = train_and_evaluate(train_df, test_df)
        best = select_best(results)
        print(f"\nBest model: {best['name']} (R2={best['metrics']['r2']:.3f})")

        save_model_registry(best, models_collection)

    except PyMongoError as e:
        print(f"MongoDB error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    run()