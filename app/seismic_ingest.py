"""
app/seismic_ingest.py

Batch ingestion script — same role as flood_analysis.py, but instead of
dumping to a JSON file, it upserts rows into Postgres via the same
SQLAlchemy Base/session used by the rest of the app.

Run manually (from backend root, same as flood_analysis.py):
    python app/seismic_ingest.py

Re-run periodically (cron / Celery beat later) to keep seismic_events
fresh. Safe to re-run — events are upserted on their USGS id, so
re-fetching an overlapping window just no-ops on existing rows.
"""

import sys
import os
from datetime import datetime, timedelta, timezone

import httpx
import pycountry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import SeismicEvent

USGS_FEED_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Fast-path aliases for places USGS reports that don't cleanly match a
# pycountry name (US states, common informal names, ocean regions).
# Extend this as you see misses in country_normalized IS NULL rows.
COUNTRY_ALIASES = {
    "CA": "United States", "AK": "United States", "NV": "United States",
    "OK": "United States", "TX": "United States", "HI": "United States",
    "USA": "United States", "US": "United States",
    "S. Korea": "Korea, Republic of", "N. Korea": "Korea, Democratic People's Republic of",
}


def _normalize_country(place: str) -> str | None:
    """
    USGS place strings: "10km SW of Ridgecrest, CA" / "94km ESE of Iwo Jima, Japan"
    Take the tail after the last comma, try alias table, then pycountry
    fuzzy search. Returns None if nothing matches (e.g. mid-ocean regions
    like "South Sandwich Islands region") — those rows still get stored,
    just without a country join key.
    """
    if not place:
        return None
    tail = place.split(",")[-1].strip() if "," in place else place.strip()

    if tail in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[tail]

    try:
        return pycountry.countries.get(name=tail).name
    except AttributeError:
        pass

    try:
        matches = pycountry.countries.search_fuzzy(tail)
        return matches[0].name if matches else None
    except LookupError:
        return None


def fetch_earthquakes(min_magnitude: float = 4.0, days: int = 30) -> list[dict]:
    start_time = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    params = {
        "format": "geojson",
        "starttime": start_time,
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(USGS_FEED_URL, params=params)
        resp.raise_for_status()
        return resp.json().get("features", [])


def upsert_events(features: list[dict]) -> tuple[int, int]:
    db = SessionLocal()
    inserted, skipped = 0, 0
    try:
        existing_ids = {
            row.id for row in db.query(SeismicEvent.id).all()
        }
        for f in features:
            event_id = f.get("id")
            if not event_id or event_id in existing_ids:
                skipped += 1
                continue

            props = f.get("properties", {})
            geom = f.get("geometry", {})
            coords = geom.get("coordinates") or [None, None, None]
            lon, lat, depth = (coords + [None, None, None])[:3]

            mag = props.get("mag")
            time_ms = props.get("time")
            place = props.get("place") or "Unknown"

            if mag is None or time_ms is None or lat is None or lon is None:
                skipped += 1
                continue

            db.add(
                SeismicEvent(
                    id=event_id,
                    magnitude=mag,
                    place=place,
                    country_normalized=_normalize_country(place),
                    event_time=datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc),
                    depth_km=depth if depth is not None else 0.0,
                    latitude=lat,
                    longitude=lon,
                    tsunami_flag=int(bool(props.get("tsunami", 0))),
                    url=props.get("url", ""),
                )
            )
            inserted += 1

        db.commit()
        return inserted, skipped
    finally:
        db.close()


def main():
    print("Fetching earthquakes from USGS (min magnitude 4.0, last 30 days)...")
    features = fetch_earthquakes(min_magnitude=4.0, days=30)
    print(f"  Retrieved {len(features)} events from feed")

    inserted, skipped = upsert_events(features)
    print(f"  Inserted {inserted} new events, skipped {skipped} (already present or incomplete)")
    print("Done.")


if __name__ == "__main__":
    main()