import numpy as np
import pandas as pd
import pytest
from smartcity.data import hourly_frame
from smartcity.model import feature_frame, feature_names, scores, HORIZON


def test_hourly_frame_keeps_utc_and_nulls():
    value = {"hourly": {"time": ["2026-11-01T01:00", "2026-11-01T02:00"],
                        "pm2_5": [10.2, None]}}
    frame = hourly_frame(value, ["pm2_5"])
    assert len(frame) == 2
    assert str(frame.time.dt.tz) == "UTC"
    assert pd.isna(frame.iloc[1]["pm2_5"])


def test_feature_target_is_24_hours_ahead():
    hours = pd.date_range("2026-01-01", periods=160, freq="h", tz="UTC")
    frame = pd.DataFrame({"time": hours})
    for col in ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
                "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "wind_direction_10m", "precipitation"]:
        frame[col] = np.arange(160, dtype=float)
    transformed = feature_frame(frame)
    assert transformed["target"].iloc[0] == HORIZON
    assert transformed["pm2_5_lag_24"].iloc[24] == 0
    assert len(feature_names()) > 10


def test_missing_hours_not_mislabeled_as_exact_horizon():
    hours = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC").delete(15)
    frame = pd.DataFrame({"time": hours, "pm2_5": np.arange(99)})
    for col in ["pm10", "nitrogen_dioxide", "ozone", "temperature_2m",
                "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
                "precipitation"]:
        frame[col] = 1.0
    result = feature_frame(frame)
    assert pd.isna(result.iloc[15]["pm2_5"])


def test_scores_perfect():
    r = scores([1, 2, 3], [1, 2, 3])
    assert r["mae"] == 0 and r["rmse"] == 0 and r["r2"] == 1


def test_invalid_dates():
    from smartcity.data import fetch_history
    with pytest.raises(ValueError):
        fetch_history("2026-11-20", "2026-11-01")
