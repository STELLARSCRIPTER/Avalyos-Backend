"""
app/seismic_risk.py

Live scoring module — same role as risk_engine.py: no external calls,
just queries against data already sitting in Postgres (populated by
seismic_ingest.py) and returns a scored dict.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_

from .models import SeismicEvent


def _tier_from_score(score: float) -> str:
    if score >= 80:
        return "SEVERE"
    if score >= 60:
        return "HIGH"
    if score >= 35:
        return "ELEVATED"
    if score >= 15:
        return "MODERATE"
    return "LOW"


def _score_events(events: list[SeismicEvent]) -> dict:
    """
    Same composite logic discussed earlier: magnitude dominates (energy
    release is exponential on the Richter scale), frequency and tsunami
    flags add secondary weight. Starting heuristic — swap in population
    density / infrastructure exposure per country before treating the
    output as a real risk figure.
    """
    if not events:
        return {
            "risk_score": 0.0,
            "risk_level": "LOW",
            "event_count": 0,
            "max_magnitude": None,
            "avg_magnitude": None,
            "reasons": ["No seismic events recorded in this window."],
        }

    mags = [e.magnitude for e in events]
    max_mag = max(mags)
    avg_mag = sum(mags) / len(mags)
    tsunami_events = sum(1 for e in events if e.tsunami_flag)

    magnitude_component = min(70.0, max_mag ** 2.2)
    frequency_component = min(20.0, len(events) * 1.5)
    tsunami_component = min(10.0, tsunami_events * 5.0)

    raw_score = magnitude_component + frequency_component + tsunami_component
    risk_score = round(min(100.0, raw_score), 1)
    level = _tier_from_score(risk_score)

    reasons = [
        f"{len(events)} recorded event(s) in the lookback window.",
        f"Peak magnitude {max_mag:.1f}, average magnitude {avg_mag:.1f}.",
    ]
    if tsunami_events:
        reasons.append(f"{tsunami_events} event(s) flagged with tsunami potential.")

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "event_count": len(events),
        "max_magnitude": round(max_mag, 2),
        "avg_magnitude": round(avg_mag, 2),
        "reasons": reasons,
    }


def score_country_seismic_risk(db: Session, country: str, days: int = 90) -> dict:
    """Score a single country by name, matching SeismicEvent.country_normalized."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    events = (
        db.query(SeismicEvent)
        .filter(
            and_(
                SeismicEvent.country_normalized.ilike(country),
                SeismicEvent.event_time >= since,
            )
        )
        .all()
    )
    result = _score_events(events)
    result["country"] = country
    result["window_days"] = days
    return result


def score_all_countries(db: Session, days: int = 90, min_events: int = 1) -> list[dict]:
    """Aggregate scores for every country with recorded events in the window."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    events = (
        db.query(SeismicEvent)
        .filter(
            SeismicEvent.event_time >= since,
            SeismicEvent.country_normalized.isnot(None),
        )
        .all()
    )

    by_country: dict[str, list[SeismicEvent]] = {}
    for e in events:
        by_country.setdefault(e.country_normalized, []).append(e)

    results = []
    for country, country_events in by_country.items():
        if len(country_events) < min_events:
            continue
        scored = _score_events(country_events)
        scored["country"] = country
        scored["window_days"] = days
        results.append(scored)

    return sorted(results, key=lambda r: r["risk_score"], reverse=True)


def score_branch_exposure(db: Session, branch, days: int = 90) -> dict:
    """
    Entity-linking step: takes a Branch ORM object and scores seismic risk
    for its country. This turns a generic global feed into a "which of my
    branches sit in elevated risk zones" answer — the link between your
    entity graph and an external risk feed.
    """
    if not branch.country:
        return {
            "branch_code": branch.code,
            "branch_name": branch.name,
            "risk_score": None,
            "risk_level": "UNKNOWN",
            "reasons": ["Branch has no country on record — cannot match to seismic data."],
        }

    country_score = score_country_seismic_risk(db, branch.country, days=days)
    return {
        "branch_code": branch.code,
        "branch_name": branch.name,
        "company": branch.company.name if branch.company else "unknown",
        "country": branch.country,
        **country_score,
    }

