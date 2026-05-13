"""
schemas.py — Pydantic models for request & response validation.
All fields include clear descriptions that appear in /docs.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ── Shared input schema ────────────────────────────────────────
class ScenarioInput(BaseModel):
    """All 11 features the model expects. Includes an optional label."""

    label: Optional[str] = Field(
        default=None,
        description="Optional name for this scenario (used in batch responses)",
        examples=["Severe Crisis"]
    )

    # ── Core 5 features (user-facing) ──────────────────────────
    daily_ship_transits: int = Field(
        ..., ge=0, le=200,
        description="Number of vessels transiting the Strait per day",
        examples=[105]
    )
    oil_throughput_mbpd: float = Field(
        ..., ge=0.0, le=25.0,
        description="Oil throughput in million barrels per day (Mbpd)",
        examples=[17.0]
    )
    war_risk_insurance_pct: float = Field(
        ..., ge=0.0, le=15.0,
        description="War-risk insurance premium as a percentage",
        examples=[0.1]
    )
    vessels_attacked_cumulative: int = Field(
        ..., ge=0, le=300,
        description="Total number of vessels attacked since crisis began",
        examples=[0]
    )
    cape_reroute_extra_days: int = Field(
        ..., ge=0, le=30,
        description="Extra sailing days added by rerouting via Cape of Good Hope",
        examples=[0]
    )

    # ── Engineered / contextual features ───────────────────────
    is_war_crisis: int = Field(
        ..., ge=0, le=1,
        description="Binary flag — 0 = pre-war / normal, 1 = active war or crisis",
        examples=[0]
    )
    days_since_closure: int = Field(
        ..., ge=0, le=365,
        description="Days elapsed since the strait disruption began (0 = no disruption)",
        examples=[0]
    )
    transit_pct_of_prewar_avg: float = Field(
        ..., ge=0.0, le=150.0,
        description="Current ship transits as % of the pre-war daily average",
        examples=[100.0]
    )
    vessels_stranded_in_gulf: int = Field(
        ..., ge=0, le=200,
        description="Number of vessels currently stranded inside the Gulf",
        examples=[0]
    )
    lng_throughput_bcfd: float = Field(
        ..., ge=0.0, le=10.0,
        description="LNG throughput in billion cubic feet per day (bcfd)",
        examples=[3.5]
    )
    diesel_usd_gallon_us: float = Field(
        ..., ge=0.5, le=20.0,
        description="US retail diesel price per gallon (USD)",
        examples=[3.60]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "label":                       "Normal Operations",
                "daily_ship_transits":          105,
                "oil_throughput_mbpd":          17.0,
                "war_risk_insurance_pct":        0.1,
                "vessels_attacked_cumulative":    0,
                "cape_reroute_extra_days":         0,
                "is_war_crisis":                   0,
                "days_since_closure":              0,
                "transit_pct_of_prewar_avg":     100.0,
                "vessels_stranded_in_gulf":        0,
                "lng_throughput_bcfd":             3.5,
                "diesel_usd_gallon_us":            3.60,
            }
        }
    }


# ── Alias for single prediction ────────────────────────────────
class PredictRequest(ScenarioInput):
    pass


# ── Response schemas ───────────────────────────────────────────
class PredictResponse(BaseModel):
    predicted_brent_usd_bbl: float = Field(
        description="Predicted Brent crude price in USD per barrel"
    )
    confidence_interval_90: Dict[str, float] = Field(
        description="90% confidence interval derived from individual tree predictions"
    )
    model_r2:   Optional[float] = Field(default=None, description="Model R² on test set")
    model_mae:  Optional[float] = Field(default=None, description="Model MAE on test set (USD/bbl)")
    model_rmse: Optional[float] = Field(default=None, description="Model RMSE on test set (USD/bbl)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_brent_usd_bbl": 74.30,
                "confidence_interval_90":  {"lower": 72.10, "upper": 76.50},
                "model_r2":   0.9924,
                "model_mae":  0.9231,
                "model_rmse": 1.2473,
            }
        }
    }


class BatchPredictRequest(BaseModel):
    scenarios: List[ScenarioInput] = Field(
        ..., min_length=1, max_length=100,
        description="List of scenarios to predict (max 100)"
    )


class BatchPredictResponse(BaseModel):
    count:       int                     = Field(description="Number of predictions returned")
    predictions: List[Dict[str, Any]]    = Field(description="Per-scenario results")
    summary:     Dict[str, float]        = Field(description="Min / max / avg across all predictions")
