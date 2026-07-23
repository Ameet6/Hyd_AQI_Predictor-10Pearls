"""
Hazardous AQI alert logic.
Checks current AQI and all forecasted days against two escalating
thresholds (100 = caution, 150 = warning), and returns the info needed
to render a banner — or None if air quality is fine across the board.
"""

CAUTION_THRESHOLD = 100
WARNING_THRESHOLD = 150


def check_alert(latest_aqi: float, forecasts: list) -> dict:
    """
    Look at current AQI + all forecasted days, find the worst (highest)
    value, and determine the alert tier.

    Returns None if everything is below the caution threshold.
    Otherwise returns:
        {
            "tier": "warning" | "caution",
            "worst_aqi": float,
            "worst_when": "today" | "Day 1" | "Day 2" | "Day 3",
            "message": str,
        }
    """
    readings = [("today", latest_aqi)]
    for f in forecasts:
        if f["predicted_aqi"] is not None:
            readings.append((f"Day {f['day']}", f["predicted_aqi"]))

    worst_when, worst_aqi = max(readings, key=lambda r: r[1])

    if worst_aqi >= WARNING_THRESHOLD:
        tier = "warning"
        message = (
            f"Unhealthy air quality expected ({worst_when}, AQI {worst_aqi:.0f}). "
            "Limit outdoor activity — this affects the general public, not just sensitive groups."
        )
    elif worst_aqi >= CAUTION_THRESHOLD:
        tier = "caution"
        message = (
            f"Elevated AQI expected ({worst_when}, AQI {worst_aqi:.0f}). "
            "Sensitive groups (children, elderly, respiratory/heart conditions) should limit prolonged outdoor exertion."
        )
    else:
        return None

    return {"tier": tier, "worst_aqi": worst_aqi, "worst_when": worst_when, "message": message}