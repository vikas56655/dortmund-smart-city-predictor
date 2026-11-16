# Dortmund Air Intelligence 🌍

**Real-data environmental analytics + reproducible 24-hour PM2.5 forecasting + an interactive city dashboard.**

[**↗ Open the live dashboard**](https://vikas56655.github.io/smart-city/) · [GitHub Actions](https://github.com/vikas56655/dortmund-smart-city-predictor/actions)

> **Data integrity:** The public dashboard currently visualizes *Open-Meteo/Copernicus CAMS modeled air-quality estimates and provider forecasts*, not UBA/LANUK ground-station observations. Its forward-looking curve is a **CAMS forecast**, not an output from the repository's XGBoost model. The XGBoost experiment runs separately and must be trained/evaluated before any performance claim. No synthetic observations or fabricated benchmark scores are shown.

## Dashboard — six interactive analyses

1. **Air quality timeline** — historical modeled hourly concentrations and future CAMS projection.
2. **Pollutant comparison** — PM2.5, PM10, NO₂, O₃ on aligned timestamps.
3. **Daily concentration profile** — local day-of-week average pollutant concentration.
4. **Hourly heatmap** — PM2.5 by Dortmund-local hour and day.
5. **Weather correlation** — weather and air quality co-evolution, clearly labeling weather-source estimates.
6. **Forecast trajectory** — upcoming modeled CAMS pollution levels against recent hours.
7. **Geographic context** — a map showing the approximate **model grid location** for Dortmund, not an official station sensor.

The site fetches recent data from the Open-Meteo APIs and offers an archived snapshot fallback via GitHub. The dashboard shows timestamps and sources.

## Data

- Air quality: [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api), based on Copernicus Atmosphere Monitoring Service (CAMS) models.
- Weather: [Open-Meteo Forecast API](https://open-meteo.com/en/docs), and [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) for retrospective experiments.
- Official measurements that could be incorporated later: [German UBA Air Data API](https://luftdaten.umweltbundesamt.de/api/air-data/v4/doc), [Dortmund measuring stations](https://www.dortmund.de/themen/umwelt-nachhaltigkeit-und-klimaschutz/luft-und-luftreinhaltung/luftqualitaet/messstationen-in-dortmund/).
- Coordinates: Dortmund center, ~51.5136° N, 7.4653° E.
- Units: pollutants in µg/m³; hourly data have hourly timestamps.

**Research limitation:** CAMS is not observed ground truth and is less suitable as an independently validated benchmark than quality-assured monitoring-station data. Live data may be delayed or unavailable. Forecast/weather providers may revise their historical series.

## Run the forecasting experiment

Python 3.10+:

~~~bash
git clone https://github.com/vikas56655/dortmund-smart-city-predictor.git
cd dortmund-smart-city-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m smartcity.data --start 2026-08-01 --end 2026-10-15
python -m smartcity.model --input data/history.csv --output-dir artifacts
~~~

Historical API availability, quotas and allowed date range can change; choose a supported interval if a request fails. The trainer uses origin-time variables and lagged PM2.5, a **24-hour-ahead** target, chronological train/validation/test partitions, purges boundaries, compares **XGBoost to persistence**, and writes `artifacts/metrics.json` and `artifacts/predictions.csv`.

No retrospective test metrics are claimed in this README because no full historical run was performed in this publication step. Model artifacts are ignored to avoid committing misleading results or heavyweight binaries.

## Automatic data collection

`.github/workflows/update-data.yml` requests current CAMS estimates every six hours and commits `data/latest.json`. GitHub Actions schedule intervals are approximate; scheduled workflows can be delayed, suspended, or denied write permissions. The live dashboard prefers direct API calls and uses the snapshot as a fallback.

For local snapshot: `python -m smartcity.snapshot`.

## Tests

~~~bash
pip install -r requirements-dev.txt
pytest -q
~~~

CI tests use synthetic *test fixtures only* and do not pretend to be real pollution data.

## Structure

~~~text
smartcity/
  data.py          # date-bounded CAMS + historical weather ETL
  model.py         # 24h XGBoost vs persistence experiment
  snapshot.py      # current CAMS data snapshot collector
dashboard/
  index.html       # static dashboard hosted on GitHub Pages
  app.js           # 6 visualizations and data-source handling
  style.css        # responsive dark design
.github/workflows/
  ci.yml
  update-data.yml
tests/
  test_pipeline.py
~~~

## Scientific roadmap

- Connect official UBA/LANUK NRW station data and validate station-specific pollutant coverage.
- Re-run predictions against actual station measurements.
- Include issue-time archived weather forecasts to avoid retrospective look-ahead.
- Add rolling-origin cross-validation, uncertainty intervals and calibrated alert thresholds.
- Compare against more baselines and publish reproducible version-pinned experiments.

## Credits

Open-Meteo (CC BY 4.0 attribution where applicable), Copernicus CAMS data providers; upstream terms and attribution apply. © 2026 Vikas Singh. MIT-licensed source code.
