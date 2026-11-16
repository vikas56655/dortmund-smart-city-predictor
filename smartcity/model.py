"""Leakage-aware 24-hour-ahead forecasting with chronological holdout.

The prediction target is CAMS MODELED PM2.5 at t+24h, NOT station-observed PM2.5.
Only variables available at t or earlier enter the model. Baseline = persistence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

HORIZON = 24
LAGS = (1, 3, 6, 24, 48)
FEATURES = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "precipitation", "hour_sin", "hour_cos",
            "weekday_sin", "weekday_cos"]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time").drop_duplicates("time").set_index("time")
    # Introduce missing rows so a 24-row shift always means exactly 24 hours.
    frame = frame.asfreq("h")
    frame["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame.index.dayofweek / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame.index.dayofweek / 7)
    for lag in LAGS:
        frame[f"pm2_5_lag_{lag}"] = frame["pm2_5"].shift(lag)
    frame["target"] = frame["pm2_5"].shift(-HORIZON)
    return frame


def feature_names() -> list[str]:
    return FEATURES + [f"pm2_5_lag_{lag}" for lag in LAGS]


def scores(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted))}


def train(csv_path: str, output_dir: str, seed: int = 42) -> dict:
    data = pd.read_csv(csv_path)
    frame = feature_frame(data).dropna(subset=feature_names() + ["target"])
    if len(frame) < 300:
        raise ValueError("Need at least 300 complete hourly rows after feature creation")
    # Splits are chronological, and purge 24 boundary hours so train targets
    # cannot overlap the subsequent evaluation period's forecast origins.
    n = len(frame)
    train_end, val_end = int(n * .65), int(n * .82)
    train_df = frame.iloc[:max(train_end - HORIZON, 0)]
    val_df = frame.iloc[train_end:max(val_end - HORIZON, train_end)]
    test_df = frame.iloc[val_end:]
    if min(map(len, [train_df, val_df, test_df])) < 30:
        raise ValueError("Not enough data after temporal splits and purge")

    # Evaluate two fixed-capacity models on validation, test only selected one.
    candidates = [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": .05},
        {"n_estimators": 400, "max_depth": 4, "learning_rate": .035},
    ]
    validations = []
    for params in candidates:
        model = XGBRegressor(**params, subsample=.85, colsample_bytree=.9,
                             objective="reg:squarederror", random_state=seed, n_jobs=2)
        model.fit(train_df[feature_names()], train_df["target"])
        pred = model.predict(val_df[feature_names()])
        validations.append((scores(val_df["target"], pred)["mae"], params, model))
    val_mae, best_params, selected = min(validations, key=lambda item: item[0])
    # Freeze hyperparameters, fit using train+validation origins while preserving
    # the strict holdout window at val_end.
    refit_df = frame.iloc[:max(val_end - HORIZON, 0)]
    final = XGBRegressor(**best_params, subsample=.85, colsample_bytree=.9,
                         objective="reg:squarederror", random_state=seed, n_jobs=2)
    final.fit(refit_df[feature_names()], refit_df["target"])
    predictions = final.predict(test_df[feature_names()])
    persistence = test_df["pm2_5"].to_numpy()
    result = {
        "source": "Open-Meteo CAMS modeled air quality + historical reanalysis weather",
        "target": "modeled PM2.5, 24-hour-ahead, µg/m³",
        "horizon_hours": HORIZON,
        "seed": seed,
        "best_params": best_params,
        "validation_mae": val_mae,
        "n_rows": {"train": len(train_df), "validation": len(val_df),
                   "test": len(test_df)},
        "test": {"xgboost": scores(test_df["target"], predictions),
                 "persistence": scores(test_df["target"], persistence)},
        "test_period": {"start": test_df.index.min().isoformat(),
                        "end": test_df.index.max().isoformat()},
        "note": "Historical retrospective experiment; not a station-measurement benchmark.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "xgboost.joblib")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"forecast_origin": test_df.index.astype(str),
                  "target_time": (test_df.index + pd.Timedelta(hours=HORIZON)).astype(str),
                  "actual": test_df.target.to_numpy(),
                  "xgboost": predictions, "persistence": persistence}).to_csv(
                      out / "predictions.csv", index=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/history.csv")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    train(args.input, args.output_dir)
