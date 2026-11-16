"""Open-Meteo CAMS modeled air-quality and weather API ingestion.

Important: Open-Meteo air quality is CAMS MODELED data, not UBA station measurements.
Timestamps always use UTC for correct hourly joins and DST behavior.
"""
from __future__ import annotations
import argparse
import json
from datetime import date, timedelta
from pathlib import Path
import requests
import pandas as pd

LAT, LON = 51.5136, 7.4653
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_COLUMNS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
WEATHER_COLUMNS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                   "wind_direction_10m", "precipitation"]


def request_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=50,
                            headers={"User-Agent": "DortmundAirIntelligence/1.0"})
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise ValueError(data.get("reason", "API error"))
    return data


def hourly_frame(payload: dict, columns: list[str]) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})
    for name in columns:
        if name not in hourly:
            raise ValueError(f"Missing requested hourly variable: {name}")
        frame[name] = pd.to_numeric(hourly[name], errors="coerce")
    if frame["time"].duplicated().any():
        raise ValueError("Duplicate source timestamps")
    return frame.sort_values("time").reset_index(drop=True)


def fetch_history(start: str, end: str) -> pd.DataFrame:
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("Start date must not exceed end date")
    common = {"latitude": LAT, "longitude": LON,
              "start_date": start, "end_date": end,
              "timezone": "GMT"}
    air = request_json(AIR_URL, {**common, "hourly": ",".join(AIR_COLUMNS),
                                 "domains": "cams_europe"})
    weather = request_json(WEATHER_URL, {**common, "hourly": ",".join(WEATHER_COLUMNS)})
    result = hourly_frame(air, AIR_COLUMNS).merge(
        hourly_frame(weather, WEATHER_COLUMNS), on="time", how="inner",
        validate="one_to_one",
    )
    if result.empty:
        raise ValueError("No matching hourly timestamps across providers")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=(date.today() - timedelta(days=90)).isoformat())
    parser.add_argument("--end", default=(date.today() - timedelta(days=6)).isoformat())
    parser.add_argument("--output", default="data/history.csv")
    args = parser.parse_args()
    frame = fetch_history(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(json.dumps({"rows": len(frame), "start": str(frame.time.min()),
                      "end": str(frame.time.max()), "saved": str(output)}, indent=2))


if __name__ == "__main__":
    main()
