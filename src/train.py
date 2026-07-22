"""
Pearls AQI Predictor — Training Pipeline (multi-horizon)
------------------------------------------------------------
For each forecast horizon (24h, 48h, 72h = Day 1/2/3), this:
  1. Builds a target column shifted by that many hours
  2. Does a time-based train/test split
  3. Trains multiple candidate models, evaluates with RMSE/MAE/R2
  4. Saves the best model (as binary) + metadata to the MongoDB model registry,
     tagged with its horizon, so the dashboard can fetch "the Day 2 model"
     separately from "the Day 1 model".

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

# One model per horizon — Day 1, Day 2, Day 3
HORIZONS_HOURS = [24, 48, 72]
TRAIN_SPLIT_RATIO = 0.85

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


def build_target_and_split(df: pd.DataFrame, horizon_hours: int):
    """Shift AQI forward by horizon_hours to create the target for THIS
    horizon, then split chronologically. Uses a copy so each horizon's
    target column doesn't interfere with the others."""
    horizon_df = df.copy()
    horizon_df["aqi_target"] = horizon_df["aqi"].shift(-horizon_hours)
    horizon_df = horizon_df.dropna(subset=["aqi_target"]).reset_index(drop=True)

    split_index = int(len(horizon_df) * TRAIN_SPLIT_RATIO)
    train_df = horizon_df.iloc[:split_index]
    test_df = horizon_df.iloc[split_index:]
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
        results.append({"name": name, "model": model, "metrics": metrics})

    return results


def select_best(results: list) -> dict:
    return max(results, key=lambda r: r["metrics"]["r2"])


def save_model_registry(best: dict, horizon_hours: int, models_collection):
    """Serialize the model to bytes and store it + metadata in MongoDB,
    tagged with its horizon so it can be fetched independently later."""
    buffer = io.BytesIO()
    joblib.dump(best["model"], buffer)
    model_bytes = buffer.getvalue()

    trained_at = datetime.now(timezone.utc)

    metadata = {
        "city": config.CITY_NAME,
        "algorithm": best["name"],
        "trained_at": trained_at,
        "horizon_hours": horizon_hours,
        "feature_cols": FEATURE_COLS,
        "metrics": best["metrics"],
        "model_binary": Binary(model_bytes),
        "is_active": True,
    }

    # Deactivate the previous active model for THIS city + THIS horizon only
    # (leaves the other horizons' active models untouched).
    models_collection.update_many(
        {"city": config.CITY_NAME, "horizon_hours": horizon_hours, "is_active": True},
        {"$set": {"is_active": False}},
    )
    models_collection.insert_one(metadata)

    size_kb = len(model_bytes) / 1024
    print(f"  Best: {best['name']} (R2={best['metrics']['r2']:.3f}), "
          f"serialized {size_kb:.1f} KB, saved as active {horizon_hours}h model")


def run():
    print("Starting multi-horizon training pipeline...")
    config.validate()
    client = db.get_client()
    try:
        features_collection = db.get_collection(config.FEATURES_COLLECTION, client)
        models_collection = db.get_collection(config.MODELS_COLLECTION, client)

        df = load_data(features_collection)
        print(f"Loaded {len(df)} rows\n")

        for horizon_hours in HORIZONS_HOURS:
            print(f"--- Horizon: {horizon_hours}h (Day {horizon_hours // 24}) ---")
            train_df, test_df = build_target_and_split(df, horizon_hours)
            print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

            results = train_and_evaluate(train_df, test_df)
            for r in results:
                m = r["metrics"]
                print(f"  {r['name']}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R2={m['r2']:.3f}")

            best = select_best(results)
            save_model_registry(best, horizon_hours, models_collection)
            print()

        print("Multi-horizon training complete.")

    except PyMongoError as e:
        print(f"MongoDB error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    run()