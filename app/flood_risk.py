"""
app/flood_risk.py

Scoring wrapper around the pre-computed TDA flood analysis
(app/data/flood_analysis_results.json).

Turns the raw topological / historical data into a per-country
{risk_score, risk_level, reasons, ...} dict — same shape as
seismic_risk.py, so the /analyze endpoint can blend them uniformly.

Data source: app/flood_analysis.py (batch, run offline, writes the JSON).
"""
import json
import os
from typing import Optional

import pycountry

# --------------------------------------------------------------------
# Load and index the pre-computed TDA results once at import time.
# --------------------------------------------------------------------
_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "flood_analysis_results.json",
)

# Manual overrides for UN-style names in the TDA JSON that pycountry
# doesn't resolve cleanly from an ISO-2 code.
_ISO_OVERRIDES = {
    "US": "United States of America",
    "GB": "United Kingdom of Great Britain and Northern Ireland",
    "RU": "Russian Federation",
    "IR": "Iran (Islamic Republic of)",
    "VN": "Viet Nam",
    "TR": "Türkiye",
    "KR": "Republic of Korea",
    "KP": "Democratic People's Republic of Korea",
    "LA": "Lao People's Democratic Republic",
    "CI": "Côte d\u0092Ivoire",  # note: the TDA JSON uses a strange apostrophe
    "CD": "Democratic Republic of the Congo",
    "CG": "Congo",
    "BO": "Bolivia (Plurinational State of)",
    "VE": "Venezuela (Bolivarian Republic of)",
    "TZ": "United Republic of Tanzania",
    "SY": "Syrian Arab Republic",
    "MD": "Republic of Moldova",
    "MK": "North Macedonia",
    "TW": "Taiwan (Province of China)",
    "PS": "State of Palestine",
    "SZ": "Eswatini",
    "CV": "Cabo Verde",
    "TL": "Timor-Leste",
}


def _load_data() -> dict:
    if not os.path.exists(_DATA_PATH):
        return {}
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_DATA = _load_data()


def _index_countries() -> tuple[dict, dict, dict]:
    """
    Returns:
      by_name:  { "United States of America": {...country entry...} }
      by_iso:   { "US": {...same entry...} }
      neighbors: { "United States of America": ["Canada", "Mexico", ...] }
    Neighbors are the 5 closest by Wasserstein distance.
    """
    by_name: dict = {}
    by_iso: dict = {}
    neighbors: dict = {}

    if not _DATA:
        return by_name, by_iso, neighbors

    # Index country_points by name
    points = _DATA.get("step6_country_topology", {}).get("country_points", [])
    for entry in points:
        name = entry.get("country")
        if not name:
            continue
        by_name[name] = entry

    # Build reverse map: full UN-style name -> ISO-2 code
    # Try pycountry first, then fall back to the manual overrides.
    override_reverse = {v: k for k, v in _ISO_OVERRIDES.items()}
    for name in by_name:
        # Direct override
        if name in override_reverse:
            by_iso[override_reverse[name]] = by_name[name]
            continue
        # Try pycountry by name (with common_name / official_name fallbacks)
        try:
            match = pycountry.countries.get(name=name)
            if not match:
                match = pycountry.countries.get(common_name=name)
            if not match:
                match = pycountry.countries.get(official_name=name)
            if match and hasattr(match, "alpha_2"):
                by_iso[match.alpha_2] = by_name[name]
        except Exception:
            pass

    # Index Wasserstein neighbors
    ws = _DATA.get("step7_wasserstein", {})
    countries = ws.get("countries", [])
    matrix = ws.get("distance_matrix", [])
    for i, cname in enumerate(countries):
        if i >= len(matrix):
            continue
        row = matrix[i]
        # Build (distance, other_name) pairs, excluding self (i == j)
        pairs = [
            (row[j], countries[j])
            for j in range(len(countries))
            if j != i and j < len(row) and isinstance(row[j], (int, float))
        ]
        pairs.sort(key=lambda x: x[0])
        neighbors[cname] = [name for _, name in pairs[:5]]

    return by_name, by_iso, neighbors


_BY_NAME, _BY_ISO, _NEIGHBORS = _index_countries()


# --------------------------------------------------------------------
# Normalization — computed once from the whole dataset
# --------------------------------------------------------------------
def _compute_norms() -> tuple[float, float]:
    if not _BY_NAME:
        return 1.0, 1.0
    damages = [e.get("mean_damage_m", 0.0) or 0.0 for e in _BY_NAME.values()]
    deaths = [e.get("mean_deaths", 0.0) or 0.0 for e in _BY_NAME.values()]
    return max(damages) or 1.0, max(deaths) or 1.0


_MAX_DAMAGE, _MAX_DEATHS = _compute_norms()


# --------------------------------------------------------------------
# ISO -> name resolution (public helper)
# --------------------------------------------------------------------
def iso_to_country_name(iso_code: str) -> Optional[str]:
    """
    Given an ISO-2 code (e.g. "US", "IN", "GB"), return the name used
    in the TDA JSON (e.g. "United States of America"). Returns None if
    the code isn't recognized.
    """
    if not iso_code:
        return None
    iso = iso_code.upper().strip()
    if iso in _ISO_OVERRIDES:
        return _ISO_OVERRIDES[iso]
    try:
        match = pycountry.countries.get(alpha_2=iso)
        if match:
            return getattr(match, "name", None)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------
def _tier_from_score(score: float) -> str:
    if score >= 80:
        return "SEVERE"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "ELEVATED"
    if score >= 20:
        return "MODERATE"
    return "LOW"


def _driver_ratio(entry: dict) -> tuple[float, str]:
    """
    Given a country entry, return (ratio, driver_name) where ratio is the
    amplification multiplier for the dominant driver.
    Falls back to 1.0 for unknown drivers.
    """
    driver = (entry.get("dominant_driver") or "").strip()
    if driver == "BoB":
        return float(entry.get("bob_ratio") or 0.0), "BoB"
    if driver == "El Nino":
        return float(entry.get("elnino_ratio") or 0.0), "El Nino"
    if driver == "pIOD":
        return float(entry.get("piod_ratio") or 0.0), "pIOD"
    return 1.0, driver or "unknown"


def flood_risk_for_country(iso_code: str) -> dict:
    """
    Given an ISO-2 code, return a scored dict. If the country isn't in
    the TDA dataset, returns {available: False, ...}.
    """
    name = iso_to_country_name(iso_code) if iso_code else None

    if not name or name not in _BY_NAME:
        return {
            "available": False,
            "country": iso_code or "unknown",
            "country_full_name": name,
            "risk_score": None,
            "risk_level": "UNKNOWN",
            "reasons": [
                f"Flood data not available for country '{iso_code or 'unknown'}'."
            ],
            "dominant_driver": None,
            "mean_damage_m": None,
            "mean_deaths": None,
            "similar_countries": [],
        }

    entry = _BY_NAME[name]

    # --- Components ---
    damage = float(entry.get("mean_damage_m") or 0.0)
    deaths = float(entry.get("mean_deaths") or 0.0)
    damage_norm = min(100.0, (damage / _MAX_DAMAGE) * 100.0)
    deaths_norm = min(100.0, (deaths / _MAX_DEATHS) * 100.0)

    driver_ratio, driver_name = _driver_ratio(entry)
    # Map ratio 0..3 -> 0..100. ratio=1 -> ~33, ratio=2 -> ~66, ratio>=3 -> 100
    driver_score = min(100.0, max(0.0, (driver_ratio / 3.0) * 100.0))

    # --- Similar-country risk (top-5 neighbors' mean damage) ---
    neighbor_names = _NEIGHBORS.get(name, [])
    neighbor_damages = [
        float(_BY_NAME[n].get("mean_damage_m") or 0.0)
        for n in neighbor_names
        if n in _BY_NAME
    ]
    if neighbor_damages:
        neighbor_avg = sum(neighbor_damages) / len(neighbor_damages)
        neighbor_score = min(100.0, (neighbor_avg / _MAX_DAMAGE) * 100.0)
    else:
        neighbor_avg = 0.0
        neighbor_score = 0.0

    # --- Weighted blend ---
    score = (
        damage_norm * 0.40
        + deaths_norm * 0.25
        + driver_score * 0.20
        + neighbor_score * 0.15
    )
    score = round(score, 1)
    level = _tier_from_score(score)

    # --- Reasons ---
    reasons = [
        f"Historical flood damage in {name} averages ${damage:,.0f}K per event "
        f"(normalized {damage_norm:.0f}/100).",
        f"Historical flood deaths in {name} average {deaths:,.0f} "
        f"(normalized {deaths_norm:.0f}/100).",
    ]
    if driver_name and driver_name != "unknown":
        reasons.append(
            f"Dominant climate driver: {driver_name} "
            f"(amplification ratio {driver_ratio:.2f})."
        )
    if neighbor_names:
        reasons.append(
            f"5 structurally similar countries to {name}: "
            f"{', '.join(neighbor_names)} — mean damage ${neighbor_avg:,.0f}K."
        )

    return {
        "available": True,
        "country": iso_code.upper() if iso_code else None,
        "country_full_name": name,
        "risk_score": score,
        "risk_level": level,
        "reasons": reasons,
        "dominant_driver": driver_name if driver_name != "unknown" else None,
        "mean_damage_m": damage,
        "mean_deaths": deaths,
        "similar_countries": neighbor_names,
    }