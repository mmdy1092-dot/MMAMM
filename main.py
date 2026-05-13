"""
╔══════════════════════════════════════════════════════════════╗
║   Hormuz Oil Price Prediction API                            ║
║   FastAPI · Random Forest · Brent Crude USD/bbl              ║
╚══════════════════════════════════════════════════════════════╝

Run locally:
    uvicorn main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import joblib
import json
import numpy as np
from pathlib import Path

from schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from utils import build_feature_array, FEATURE_NAMES

# ── Model is loaded once at startup ─────────────────────────────
MODEL_PATH   = Path("model/rf_model.pkl")
METRICS_PATH = Path("model/metrics.json")

ml = {}   # shared state: {"model": ..., "metrics": ...}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts when the server starts; clean up on shutdown."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    ml["model"]   = joblib.load(MODEL_PATH)
    ml["metrics"] = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    print("✅  Model loaded successfully")
    yield
    ml.clear()


# ── App definition ──────────────────────────────────────────────
app = FastAPI(
    title       = "🛢️  Hormuz Oil Price Prediction API",
    description = (
        "Predict **Brent Crude oil prices (USD/bbl)** based on Strait of Hormuz "
        "shipping disruption indicators.\n\n"
        "Built with a Random Forest Regressor trained on 2026 disruption data.\n\n"
        "**R² = 0.9924 | MAE = 0.92 USD/bbl | RMSE = 1.25 USD/bbl**"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten for production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Routes ──────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {
        "status":  "online",
        "message": "Hormuz Oil Price Prediction API is running 🛢️",
        "docs":    "/docs",
        "version": app.version,
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health + model status."""
    return {
        "status":       "healthy",
        "model_loaded": "model" in ml,
        "model_type":   type(ml.get("model", "")).__name__,
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return model metadata and training metrics."""
    return {
        "model":    "Random Forest Regressor",
        "target":   "brent_crude_usd_bbl",
        "features": FEATURE_NAMES,
        "metrics":  ml.get("metrics", {}),
        "description": (
            "Predicts Brent crude prices from Strait of Hormuz "
            "shipping and geopolitical disruption indicators."
        ),
    }


@app.get("/model/features", tags=["Model"])
def model_features():
    """List all features accepted by the model with descriptions."""
    return {
        "features": [
            {"name": "daily_ship_transits",        "type": "int",   "unit": "ships/day",  "description": "Number of vessels transiting the strait daily"},
            {"name": "oil_throughput_mbpd",        "type": "float", "unit": "Mbpd",       "description": "Oil throughput in million barrels per day"},
            {"name": "war_risk_insurance_pct",     "type": "float", "unit": "% premium",  "description": "War-risk insurance premium percentage"},
            {"name": "vessels_attacked_cumulative","type": "int",   "unit": "vessels",    "description": "Cumulative vessel attacks since crisis start"},
            {"name": "cape_reroute_extra_days",    "type": "int",   "unit": "days",       "description": "Extra transit days via Cape of Good Hope rerouting"},
            {"name": "is_war_crisis",              "type": "int",   "unit": "0 or 1",     "description": "Binary flag: 0 = pre-war, 1 = active war/crisis"},
            {"name": "days_since_closure",         "type": "int",   "unit": "days",       "description": "Days elapsed since strait disruption began"},
            {"name": "transit_pct_of_prewar_avg",  "type": "float", "unit": "%",          "description": "Ship transits as % of pre-war daily average"},
            {"name": "vessels_stranded_in_gulf",   "type": "int",   "unit": "vessels",    "description": "Number of vessels currently stranded in Gulf"},
            {"name": "lng_throughput_bcfd",        "type": "float", "unit": "bcfd",       "description": "LNG throughput in billion cubic feet per day"},
            {"name": "diesel_usd_gallon_us",       "type": "float", "unit": "USD/gallon", "description": "US retail diesel price per gallon"},
        ]
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    **Predict Brent crude oil price for a single scenario.**

    Supply all 11 feature values describing current strait conditions;
    receive the predicted Brent crude price in USD per barrel.
    """
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    try:
        X = build_feature_array(request)
        prediction = float(ml["model"].predict(X)[0])

        # Confidence interval via individual tree predictions
        tree_preds  = np.array([t.predict(X)[0] for t in ml["model"].estimators_])
        ci_lower    = float(np.percentile(tree_preds, 5))
        ci_upper    = float(np.percentile(tree_preds, 95))

        return PredictResponse(
            predicted_brent_usd_bbl = round(prediction, 2),
            confidence_interval_90  = {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2),
            },
            model_r2   = ml["metrics"].get("r2", None),
            model_mae  = ml["metrics"].get("mae", None),
            model_rmse = ml["metrics"].get("rmse", None),
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """
    **Predict Brent crude prices for multiple scenarios at once.**

    Send a list of up to 100 scenario objects; receive a prediction for each.
    Useful for sensitivity analysis and scenario planning.
    """
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(request.scenarios) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 scenarios per batch request.")

    try:
        results = []
        for i, scenario in enumerate(request.scenarios):
            X = build_feature_array(scenario)
            pred = float(ml["model"].predict(X)[0])
            tree_preds = np.array([t.predict(X)[0] for t in ml["model"].estimators_])
            results.append({
                "scenario_index":         i,
                "scenario_label":         scenario.label or f"Scenario {i+1}",
                "predicted_brent_usd_bbl": round(pred, 2),
                "confidence_interval_90": {
                    "lower": round(float(np.percentile(tree_preds, 5)),  2),
                    "upper": round(float(np.percentile(tree_preds, 95)), 2),
                },
            })

        preds = [r["predicted_brent_usd_bbl"] for r in results]
        return BatchPredictResponse(
            count         = len(results),
            predictions   = results,
            summary       = {
                "min_price": min(preds),
                "max_price": max(preds),
                "avg_price": round(sum(preds) / len(preds), 2),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/predict/scenarios", tags=["Prediction"])
def example_scenarios():
    """Return four pre-built example scenarios for quick testing."""
    return {
        "scenarios": [
            {
                "label": "Normal Operations (Pre-War)",
                "daily_ship_transits":        105,
                "oil_throughput_mbpd":        17.0,
                "war_risk_insurance_pct":     0.1,
                "vessels_attacked_cumulative": 0,
                "cape_reroute_extra_days":     0,
                "is_war_crisis":               0,
                "days_since_closure":          0,
                "transit_pct_of_prewar_avg": 100.0,
                "vessels_stranded_in_gulf":    0,
                "lng_throughput_bcfd":         3.5,
                "diesel_usd_gallon_us":        3.60,
            },
            {
                "label": "Mild Disruption",
                "daily_ship_transits":        55,
                "oil_throughput_mbpd":        11.5,
                "war_risk_insurance_pct":     0.9,
                "vessels_attacked_cumulative": 18,
                "cape_reroute_extra_days":     4,
                "is_war_crisis":               0,
                "days_since_closure":         15,
                "transit_pct_of_prewar_avg":  52.0,
                "vessels_stranded_in_gulf":   10,
                "lng_throughput_bcfd":         2.8,
                "diesel_usd_gallon_us":        4.30,
            },
            {
                "label": "Severe Crisis (Strait Partially Closed)",
                "daily_ship_transits":        12,
                "oil_throughput_mbpd":         4.5,
                "war_risk_insurance_pct":      3.5,
                "vessels_attacked_cumulative": 68,
                "cape_reroute_extra_days":     12,
                "is_war_crisis":               1,
                "days_since_closure":         55,
                "transit_pct_of_prewar_avg":  11.0,
                "vessels_stranded_in_gulf":   28,
                "lng_throughput_bcfd":         1.2,
                "diesel_usd_gallon_us":        5.80,
            },
            {
                "label": "Full Closure + Maximum Rerouting",
                "daily_ship_transits":         0,
                "oil_throughput_mbpd":         0.5,
                "war_risk_insurance_pct":      5.8,
                "vessels_attacked_cumulative": 110,
                "cape_reroute_extra_days":     18,
                "is_war_crisis":               1,
                "days_since_closure":          90,
                "transit_pct_of_prewar_avg":   0.0,
                "vessels_stranded_in_gulf":    45,
                "lng_throughput_bcfd":          0.3,
                "diesel_usd_gallon_us":         7.20,
            },
        ]
    }
