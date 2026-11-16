"""Scheduled snapshot of modeled near-current air quality and weather.

This is a forecast/model-feed snapshot, not an official measurement feed.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from smartcity.data import AIR_URL, LAT, LON, request_json

def main():
    payload = request_json(AIR_URL, {
        "latitude": LAT, "longitude": LON, "timezone": "GMT",
        "current": "pm2_5,pm10,nitrogen_dioxide,ozone,european_aqi",
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,european_aqi",
        "past_days": 7, "forecast_days": 3, "domains": "cams_europe",
    })
    out = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": "Open-Meteo / Copernicus CAMS Europe",
        "data_kind": "modeled estimate and provider forecast; not a station measurement",
        "latitude": LAT, "longitude": LON,
        "current": payload.get("current", {}),
        "hourly": payload["hourly"],
    }
    target = Path("data/latest.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False),
                      encoding="utf-8")
    print(f"Wrote {target}; current timestamp {out['current'].get('time', 'unknown')}")

if __name__ == "__main__":
    main()
