from typing import Optional, List, Tuple

# Illustrative sector risk weights (0-100, higher = riskier).
# Replace with real data later.
SECTOR_RISK = {
    "Technology": 55,
    "Automotive": 60,
    "Industrial": 50,
    "Healthcare": 35,
    "Chemicals": 55,
    "Consumer Goods": 30,
    "Food & Beverage": 25,
}
DEFAULT_SECTOR_RISK = 50


def _sector_risk(sector: Optional[str]) -> Tuple[float, str]:
    if not sector:
        return DEFAULT_SECTOR_RISK, "No sector provided — used neutral default risk."
    score = SECTOR_RISK.get(sector, DEFAULT_SECTOR_RISK)
    return score, f"Sector '{sector}' carries a baseline volatility score of {score}/100."


def _capital_exposure_risk(investment_amount: float) -> Tuple[float, str]:
    if investment_amount < 100_000:
        return 20, "Investment amount is small — limited downside exposure."
    if investment_amount < 1_000_000:
        return 40, "Moderate investment amount — meaningful but manageable exposure."
    if investment_amount < 10_000_000:
        return 65, "Large investment amount — significant capital at risk."
    return 85, "Very large investment amount — high capital exposure."


def _time_horizon_risk(years: float) -> Tuple[float, str]:
    if years < 1:
        return 80, "Short time horizon (<1yr) — little room to recover from downturns."
    if years < 3:
        return 60, "Short-to-medium horizon (1-3yrs) — limited recovery buffer."
    if years < 7:
        return 35, "Medium-to-long horizon (3-7yrs) — reasonable recovery buffer."
    return 20, "Long time horizon (7yrs+) — ample room to ride out volatility."


def _scale_risk(employees: Optional[int]) -> Tuple[float, str]:
    if employees is None:
        return 50, "No company scale data available — used neutral default."
    if employees > 20_000:
        return 20, f"Large, established organization (~{employees:,} employees) — more stable."
    if employees > 5_000:
        return 40, f"Mid-large organization (~{employees:,} employees)."
    if employees > 1_000:
        return 60, f"Mid-size organization (~{employees:,} employees) — moderate stability."
    return 80, f"Small organization (~{employees:,} employees) — less stability buffer."


def score_scenario(
    sector: Optional[str],
    investment_amount: float,
    time_horizon_years: float,
    employees: Optional[int] = None,
) -> dict:
    sector_score, sector_reason = _sector_risk(sector)
    capital_score, capital_reason = _capital_exposure_risk(investment_amount)
    time_score, time_reason = _time_horizon_risk(time_horizon_years)
    scale_score, scale_reason = _scale_risk(employees)

    weighted = (
        sector_score * 0.40
        + capital_score * 0.30
        + time_score * 0.15
        + scale_score * 0.15
    )

    if weighted < 35:
        level = "Low"
    elif weighted < 65:
        level = "Medium"
    else:
        level = "High"

    reasons: List[str] = [sector_reason, capital_reason, time_reason, scale_reason]

    suggestion = _suggestion_for(level)

    return {
        "risk_score": round(weighted, 1),
        "risk_level": level,
        "reasons": reasons,
        "suggestion": suggestion,
    }


def _suggestion_for(level: str) -> str:
    if level == "Low":
        return (
            "Conditions look favorable. Proceeding with the full planned exposure "
            "is reasonable, with standard monitoring."
        )
    if level == "Medium":
        return (
            "Moderate risk detected. Consider starting with a smaller pilot "
            "position or phased exposure before committing fully."
        )
    return (
        "High risk detected. Recommend a small pilot or trial exposure first, "
        "with clear stop-loss / exit criteria, before scaling up investment."
    )