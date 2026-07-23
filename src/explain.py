"""
SHAP explanation helpers.
Explains WHY a model produced a given prediction, by computing each
feature's contribution (positive = pushed AQI up, negative = pushed it down).

Uses shap.TreeExplainer, which works directly on tree-based models
(Random Forest, Gradient Boosting) without needing a background dataset,
computing exact Shapley values from the tree structure itself.
"""

import pandas as pd
import shap

FEATURE_LABELS = {
    "aqi": "Current AQI",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "co": "Carbon Monoxide",
    "no2": "Nitrogen Dioxide",
    "so2": "Sulfur Dioxide",
    "o3": "Ozone",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "wind_speed": "Wind Speed",
    "hour": "Hour of Day",
    "day": "Day of Month",
    "month": "Month",
    "day_of_week": "Day of Week",
    "aqi_change_rate": "AQI Trend",
}


def explain_prediction(model, feature_cols: list, feature_row: dict) -> list:
    """
    Compute SHAP values for a single prediction.

    Returns a list of dicts, sorted by absolute impact (largest first):
        [{"feature": "pressure", "value": 1004.2, "shap_value": 3.1}, ...]

    shap_value > 0 means that feature pushed the prediction UP (worse AQI).
    shap_value < 0 means it pushed the prediction DOWN (better AQI).
    """
    X = pd.DataFrame([{col: feature_row.get(col) for col in feature_cols}])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]  # [0] = the single row we passed in

    contributions = [
        {"feature": feature_cols[i], "value": X.iloc[0, i], "shap_value": float(shap_values[i])}
        for i in range(len(feature_cols))
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions