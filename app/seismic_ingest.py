"""
app/seismic_ingest.py

Batch ingestion script — same role as flood_analysis.py, but instead of
dumping to a JSON file, it upserts rows into Postgres via the same
SQLAlchemy Base/session used by the rest of the app.

Run modes:

    # Default: last 30 days, min magnitude 4.0
    python app/seismic_ingest.py

    # Custom window (days back from now)
    python app/seismic_ingest.py --days 180

    # Absolute date range (for historical backfill / backtesting)
    python app/seismic_ingest.py --start 2019-01-01 --end 2023-12-31

Re-run periodically (cron / Celery beat later) to keep seismic_events
fresh. Safe to re-run — events are upserted on their USGS id, so
re-fetching an overlapping window just no-ops on existing rows.

Note on historical backfill: USGS archives go back to 1900. Backfilling
2019-2023 at min magnitude 4.0 fetches ~30-50k events. The script fetches
in chunks (default 30 days) to stay under USGS's response size limits.
"""

import argparse
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


def _normalize_country(place: str) -> tuple[str | None, str | None]:
    """
    USGS place strings: "10km SW of Ridgecrest, CA" / "94km ESE of Iwo Jima, Japan"
    Take the tail after the last comma, try alias table, then pycountry
    fuzzy search.

    Returns (country_name, iso_code). Both may be None if nothing matched
    (e.g. mid-ocean regions like "South Sandwich Islands region").
    """
    if not place:
        return None, None
    tail = place.split(",")[-1].strip() if "," in place else place.strip()

    if tail in COUNTRY_ALIASES:
        tail = COUNTRY_ALIASES[tail]

    # Direct name match
    try:
        c = pycountry.countries.get(name=tail)
        if c:
            return c.name, getattr(c, "alpha_2", None)
    except (AttributeError, KeyError):
        pass

    # Fuzzy search
    try:
        matches = pycountry.countries.search_fuzzy(tail)
        if matches:
            m = matches[0]
            return m.name, getattr(m, "alpha_2", None)
    except LookupError:
        pass

    return None, None


def _fetch_chunk(start: datetime, end: datetime, min_magnitude: float) -> list[dict]:
    """Fetch one chunk of events from USGS between start and end."""
    params = {
        "format": "geojson",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(USGS_FEED_URL, params=params)
        resp.raise_for_status()
        return resp.json().get("features", [])


def fetch_earthquakes(
    min_magnitude: float = 4.0,
    days: int = 30,
    start: datetime | None = None,
    end: datetime | None = None,
    chunk_days: int = 30,
) -> list[dict]:
    """
    Fetch earthquakes. If start/end given, fetch that absolute range in
    chunks. Otherwise fetch `days` back from now.
    """
    if start and end:
        all_features: list[dict] = []
        cursor = start
        total_chunks = 0
        while cursor < end:
            next_cursor = min(cursor + timedelta(days=chunk_days), end)
            print(f"  Fetching {cursor.date()} → {next_cursor.date()}...")
            chunk = _fetch_chunk(cursor, next_cursor, min_magnitude)
            all_features.extend(chunk)
            total_chunks += 1
            cursor = next_cursor
        print(f"  Fetched {len(all_features)} events across {total_chunks} chunks")
        return all_features

    # Default: last N days
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    end_time = datetime.now(timezone.utc)
    return _fetch_chunk(start_time, end_time, min_magnitude)


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

            country_name, country_iso = _normalize_country(place)

            db.add(
                SeismicEvent(
                    id=event_id,
                    magnitude=mag,
                    place=place,
                    country_normalized=country_name,
                    country_iso=country_iso,
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
    parser = argparse.ArgumentParser(description="Ingest USGS earthquakes into Postgres")
    parser.add_argument("--days", type=int, default=30,
                        help="Days back from now (default 30, ignored if --start/--end given)")
    parser.add_argument("--start", type=str, default=None,
                        help="Absolute start date, YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None,
                        help="Absolute end date, YYYY-MM-DD")
    parser.add_argument("--min-magnitude", type=float, default=4.0,
                        help="Minimum magnitude (default 4.0)")
    args = parser.parse_args()

    start = None
    end = None
    if args.start and args.end:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        print(f"Fetching USGS events from {args.start} to {args.end} (min mag {args.min_magnitude})...")
    else:
        print(f"Fetching USGS events from last {args.days} days (min mag {args.min_magnitude})...")

    features = fetch_earthquakes(
        min_magnitude=args.min_magnitude,
        days=args.days,
        start=start,
        end=end,
    )
    print(f"  Retrieved {len(features)} events from feed")

    inserted, skipped = upsert_events(features)
    print(f"  Inserted {inserted} new events, skipped {skipped} (already present or incomplete)")
    print("Done.")


if __name__ == "__main__":
    main()