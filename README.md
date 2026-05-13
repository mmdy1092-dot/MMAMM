# 🛢️ Hormuz Oil Price Prediction API

> **Predict Brent Crude oil prices (USD/bbl)** from Strait of Hormuz shipping disruption indicators — powered by a Random Forest Regressor with **R² = 0.9924**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Request & Response Examples](#request--response-examples)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Future Upgrades](#future-upgrades)

---

## Overview

This REST API wraps a trained **Random Forest Regressor** that predicts Brent crude oil prices based on real-time Strait of Hormuz disruption data — ship transit counts, oil throughput, war-risk insurance premiums, vessel attack counts, and rerouting delays.

| Metric  | Value         |
|---------|---------------|
| R²      | **0.9924**    |
| MAE     | 0.92 USD/bbl  |
| RMSE    | 1.25 USD/bbl  |
| Dataset | 125 daily observations (Jan–May 2026) |

---

## Features

- ⚡ **FastAPI** — async, auto-documented, production-ready
- 🔮 **Single & batch predictions** — analyse one scenario or hundreds at once
- 📊 **90% confidence intervals** — from individual Random Forest trees
- 📝 **Interactive docs** — Swagger UI at `/docs`, ReDoc at `/redoc`
- 🐳 **Docker-ready** — one command to containerise
- ✅ **Test suite** — 20+ pytest tests included

---

## Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hormuz-oil-price-api.git
cd hormuz-oil-price-api
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Start the server

```bash
uvicorn main:app --reload --port 8000
```

### 5 — Open the interactive docs

```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/health` | Detailed server + model status |
| `GET`  | `/model/info` | Model metadata and training metrics |
| `GET`  | `/model/features` | Feature list with descriptions and units |
| `POST` | `/predict` | Predict price for a single scenario |
| `POST` | `/predict/batch` | Predict prices for up to 100 scenarios |
| `GET`  | `/predict/scenarios` | Load 4 pre-built example scenarios |

---

## Request & Response Examples

### Single Prediction

**Request**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "daily_ship_transits": 105,
    "oil_throughput_mbpd": 17.0,
    "war_risk_insurance_pct": 0.1,
    "vessels_attacked_cumulative": 0,
    "cape_reroute_extra_days": 0,
    "is_war_crisis": 0,
    "days_since_closure": 0,
    "transit_pct_of_prewar_avg": 100.0,
    "vessels_stranded_in_gulf": 0,
    "lng_throughput_bcfd": 3.5,
    "diesel_usd_gallon_us": 3.60
  }'
```

**Response**
```json
{
  "predicted_brent_usd_bbl": 74.30,
  "confidence_interval_90": {
    "lower": 72.10,
    "upper": 76.50
  },
  "model_r2": 0.9924,
  "model_mae": 0.9231,
  "model_rmse": 1.2473
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      { "label": "Normal", "daily_ship_transits": 105, "oil_throughput_mbpd": 17.0, ... },
      { "label": "Crisis", "daily_ship_transits": 12,  "oil_throughput_mbpd": 4.5,  ... }
    ]
  }'
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **20 tests passed** ✅

---

## Docker

### Build and run

```bash
docker build -t hormuz-api .
docker run -p 8000:8000 hormuz-api
```

### With Docker Compose

```bash
docker compose up
```

---

## Project Structure

```
hormuz-oil-price-api/
├── main.py              # FastAPI app, all route handlers
├── schemas.py           # Pydantic request/response models
├── utils.py             # Feature helpers
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── docker-compose.yml   # Compose config
├── .gitignore
├── README.md
├── model/
│   ├── rf_model.pkl     # Trained Random Forest (joblib)
│   └── metrics.json     # Training metrics
└── tests/
    └── test_api.py      # Full pytest test suite
```

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| n_estimators | 300 |
| max_features | sqrt |
| min_samples_leaf | 2 |
| Train / Test split | 75% / 25% |
| Random state | 42 |

### Feature Importance (Top 5)

| Feature | Description |
|---------|-------------|
| `diesel_usd_gallon_us` | US retail diesel price — proxy for global energy cost |
| `war_risk_insurance_pct` | Insurance premium spike signals escalation |
| `vessels_attacked_cumulative` | Accumulated attack severity |
| `oil_throughput_mbpd` | Direct supply constraint |
| `transit_pct_of_prewar_avg` | Normalised flow rate |

---

## Future Upgrades

- **XGBoost / LightGBM** — faster, regularised gradient boosting with SHAP explainability
- **LSTM / GRU** — deep learning for temporal autocorrelation in price sequences
- **Temporal Fusion Transformer** — state-of-the-art multi-horizon forecasting
- **TimeSeriesSplit CV** — replace random split with walk-forward validation
- **SHAP integration** — per-prediction feature attribution endpoint (`/explain`)
- **Streaming predictions** — SSE endpoint for real-time price feeds

---

## License

MIT — see `LICENSE` for details.
