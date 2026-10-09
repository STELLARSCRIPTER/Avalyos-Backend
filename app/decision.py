"""
app/decision.py

Core decision-analysis logic. Extracted from main.py so that multiple
endpoints (/analyze, /compare, future /decide/multi) can share the same
signal-blending, weighting, and suggestion logic.

Public API:
    analyze_one(scenario, db) -> dict
        Blend financial + seismic + flood signals into a single verdict.

Input `scenario` is a dict with:
    company: Optional[str]
    sector: Optional[str]
    country: Optional[str]   (ISO-2)
    investment_amount: float
    time_horizon_years: float

Output matches schemas.AnalyzeResponse.
"""
from typing import Optional, List

from sqlalchemy.orm import Session

from . import models
from . import seismic_risk
from . import flood_risk
from .risk_engine import score_scenario, _suggestion_for


# Weights for each signal when all three are available.
# These are the *base* weights; if a signal is unavailable, its share
# is redistributed proportionally across the remaining ones.
BASE_WEIGHTS = {
    "financial": 0.50,
    "seismic": 0.20,
    "flood": 0.30,
}


def _level_from_score(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 65:
        return "Medium"
    return "High"


def _resolve_company_context(
    company_name: Optional[str],
    db: Session,
) -> tuple[Optional[str], Optional[int]]:
    """
    If the company exists in our local DB, return (sector, total_employees).
    Otherwise (None, None).
    """
    if not company_name:
        return None, None
    company = (
        db.query(models.Company)
        .filter(models.Company.name.ilike(company_name))
        .first()
    )
    if not company:
        return None, None
    employees = sum(b.employees or 0 for b in company.branches)
    return company.sector, employees


def _score_seismic(country: Optional[str], db: Session) -> dict:
    """
    Returns {available, score, reason}. Never raises.
    """
    if not country:
        return {"available": False, "score": None, "reason": None}

    try:
        s = seismic_risk.score_country_seismic_risk(db, country, days=180)
        event_count = s.get("event_count", 0)
        if event_count > 0:
            return {
                "available": True,
                "score": s.get("risk_score"),
                "reason": (
                    f"Seismic activity in {country}: {s.get('risk_level')} "
                    f"({event_count} event(s), max magnitude "
                    f"{s.get('max_magnitude') or '—'}) in last 180 days."
                ),
            }
        return {
            "available": True,
            "score": 0.0,
            "reason": (
                f"No seismic events recorded in {country} in the last 180 days."
            ),
        }
    except Exception:
        return {"available": False, "score": None, "reason": None}


def _score_flood(country: Optional[str]) -> dict:
    """
    Returns {available, score, reasons}. Never raises.
    """
    if not country:
        return {"available": False, "score": None, "reasons": []}

    try:
        f = flood_risk.flood_risk_for_country(country)
        if f.get("available") and f.get("risk_score") is not None:
            return {
                "available": True,
                "score": f["risk_score"],
                "reasons": f.get("reasons", []) or [],
            }
    except Exception:
        pass
    return {"available": False, "score": None, "reasons": []}


def analyze_one(scenario: dict, db: Session) -> dict:
    """
    Blend financial + seismic + flood for a single scenario.
    Returns a dict matching schemas.AnalyzeResponse.
    """
    company_name = scenario.get("company")
    sector = scenario.get("sector")
    country = scenario.get("country")
    investment_amount = scenario["investment_amount"]
    time_horizon_years = scenario["time_horizon_years"]

    # --- Resolve company context ---
    local_sector, employees = _resolve_company_context(company_name, db)
    effective_sector = sector or local_sector

    # --- Signal 1: Financial (always available) ---
    financial = score_scenario(
        sector=effective_sector,
        investment_amount=investment_amount,
        time_horizon_years=time_horizon_years,
        employees=employees,
    )
    financial_score = financial["risk_score"]

    # --- Signal 2: Seismic ---
    seismic = _score_seismic(country, db)

    # --- Signal 3: Flood ---
    flood = _score_flood(country)

    # --- Availability and weight normalization ---
    availability = {
        "financial": True,
        "seismic": seismic["available"],
        "flood": flood["available"],
    }
    available_sum = sum(w for k, w in BASE_WEIGHTS.items() if availability[k]) or 1.0
    norm = {
        k: (BASE_WEIGHTS[k] / available_sum if availability[k] else 0.0)
        for k in BASE_WEIGHTS
    }

    # --- Composite ---
    combined = (
        financial_score * norm["financial"]
        + (seismic["score"] or 0.0) * norm["seismic"]
        + (flood["score"] or 0.0) * norm["flood"]
    )
    combined = round(combined, 1)
    level = _level_from_score(combined)

    # --- Reasons ---
    reasons: List[str] = list(financial["reasons"])
    if seismic["reason"]:
        reasons.append(seismic["reason"])
    reasons.extend(flood["reasons"])

    # --- Suggestion ---
    suggestion = _suggestion_for(level)
    if seismic["available"] and seismic["score"] is not None and seismic["score"] >= 35:
        suggestion += (
            " Seismic exposure is non-trivial — consider monitoring and contingency "
            "planning for assets in this region."
        )
    if flood["available"] and flood["score"] is not None and flood["score"] >= 40:
        suggestion += (
            " Historical flood recurrence is significant in this country — "
            "review physical asset placement and supply-chain contingencies."
        )

    # --- Signals payload ---
    signals = [
        {
            "signal": "financial",
            "score": financial_score,
            "weight": round(norm["financial"], 2),
            "available": True,
        },
        {
            "signal": "seismic",
            "score": seismic["score"],
            "weight": round(norm["seismic"], 2),
            "available": seismic["available"],
        },
        {
            "signal": "flood",
            "score": flood["score"],
            "weight": round(norm["flood"], 2),
            "available": flood["available"],
        },
    ]

    return {
        "risk_score": combined,
        "risk_level": level,
        "reasons": reasons,
        "suggestion": suggestion,
        "signals": signals,
    }