"""
app/backtest_seismic.py

Backtest the seismic signal against historical outcomes.

Method:
  - Training window: 2014-2018. For each country, compute a seismic score
    from events only in this window.
  - Test window: 2019-2023. Label = did that country have >= 1 seismic event
    during this window?
  - Metrics: Pearson correlation between score and event count, AUC.

Signal definition mirrors seismic_risk._score_events:
    risk_score = min(100,
        min(70, max_mag ** 2.2)
      + min(20, event_count * 1.5)
      + min(10, tsunami_events * 5.0)
    )

Run:
  python app/backtest_seismic.py
"""
import argparse
import json
import os
from datetime import datetime, timezone

from sqlalchemy import func

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import SeismicEvent
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "backtest_results_seismic.json")

TRAIN_START = datetime(2014, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2019, 1, 1, tzinfo=timezone.utc)
TEST_START = datetime(2019, 1, 1, tzinfo=timezone.utc)
TEST_END = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _score_from_events(events: list) -> float:
    """
    Mirror of seismic_risk._score_events — same formula, applied to a
    filtered event list. Returns a score in [0, 100].
    """
    if not events:
        return 0.0

    mags = [e.magnitude for e in events if e.magnitude is not None]
    if not mags:
        return 0.0

    max_mag = max(mags)
    tsunami_events = sum(1 for e in events if e.tsunami_flag)

    magnitude_component = min(70.0, max_mag ** 2.2)
    frequency_component = min(20.0, len(events) * 1.5)
    tsunami_component = min(10.0, tsunami_events * 5.0)

    raw = magnitude_component + frequency_component + tsunami_component
    return round(min(100.0, raw), 2)


def _build_dataset(db) -> list[dict]:
    """
    For each country with ISO code:
      - training score from events in TRAIN window
      - test event count from TEST window
    """
    # --- Training window ---
    train_events = (
        db.query(SeismicEvent)
        .filter(
            SeismicEvent.event_time >= TRAIN_START,
            SeismicEvent.event_time < TRAIN_END,
            SeismicEvent.country_iso.isnot(None),
        )
        .all()
    )

    by_iso_train: dict[str, list] = {}
    for e in train_events:
        by_iso_train.setdefault(e.country_iso, []).append(e)

    # --- Test window ---
    test_counts = (
        db.query(SeismicEvent.country_iso, func.count(SeismicEvent.id))
        .filter(
            SeismicEvent.event_time >= TEST_START,
            SeismicEvent.event_time < TEST_END,
            SeismicEvent.country_iso.isnot(None),
        )
        .group_by(SeismicEvent.country_iso)
        .all()
    )
    test_map = {iso: cnt for iso, cnt in test_counts}

    rows = []
    for iso, events in by_iso_train.items():
        score = _score_from_events(events)
        test_count = int(test_map.get(iso, 0))
        rows.append({
            "country_iso": iso,
            "score": score,
            "event_count": test_count,
            "has_event": 1 if test_count > 0 else 0,
        })
    return rows


def _run_metrics(rows: list[dict]) -> dict:
    import pandas as pd

    df = pd.DataFrame(rows)
    n = len(df)
    n_pos = int(df["has_event"].sum())
    n_neg = n - n_pos

    corr_r = corr_p = None
    auc = None

    if df["score"].nunique() > 1 and df["event_count"].nunique() > 1:
        try:
            corr_r, corr_p = pearsonr(df["score"], df["event_count"])
        except Exception:
            pass

    if n_pos > 0 and n_neg > 0:
        try:
            auc = roc_auc_score(df["has_event"], df["score"])
        except Exception:
            pass

    return df, {
        "n_countries": n,
        "n_with_event_2019_2023": n_pos,
        "n_without_event_2019_2023": n_neg,
        "train_window": f"{TRAIN_START.year}-{TRAIN_END.year - 1}",
        "test_window": f"{TEST_START.year}-{TEST_END.year - 1}",
        "pearson_r": round(corr_r, 4) if corr_r is not None else None,
        "pearson_p": round(corr_p, 6) if corr_p is not None else None,
        "auc": round(auc, 4) if auc is not None else None,
        "mean_score_with_event": round(float(df[df["has_event"] == 1]["score"].mean()), 2) if n_pos > 0 else None,
        "mean_score_without_event": round(float(df[df["has_event"] == 0]["score"].mean()), 2) if n_neg > 0 else None,
    }


def _print_report(metrics: dict, df):
    print("=" * 68)
    print("SEISMIC SIGNAL BACKTEST REPORT")
    print("=" * 68)
    print(f"Train window:              {metrics['train_window']}")
    print(f"Test window:               {metrics['test_window']}")
    print(f"Countries tested:          {metrics['n_countries']}")
    print(f"  With event 2019-2023:    {metrics['n_with_event_2019_2023']}")
    print(f"  Without event:           {metrics['n_without_event_2019_2023']}")
    print()
    print("SIGNAL POWER")
    print(f"  Pearson r (score vs event count):  {metrics['pearson_r']}")
    print(f"  Pearson p-value:                   {metrics['pearson_p']}")
    print(f"  AUC (score vs event occurred):     {metrics['auc']}")
    print()
    print("SCORE COMPARISON")
    print(f"  Mean score (with event):    {metrics['mean_score_with_event']}")
    print(f"  Mean score (without):       {metrics['mean_score_without_event']}")
    print()

    top = df.sort_values("score", ascending=False).head(10)
    bottom = df.sort_values("score", ascending=True).head(10)

    print("TOP 10 SCORED COUNTRIES")
    print(f"  {'ISO':<6} {'SCORE':>7} {'EVENTS':>8}")
    for _, r in top.iterrows():
        print(f"  {r['country_iso']:<6} {r['score']:>7.2f} {r['event_count']:>8}")
    print()

    print("BOTTOM 10 SCORED COUNTRIES")
    print(f"  {'ISO':<6} {'SCORE':>7} {'EVENTS':>8}")
    for _, r in bottom.iterrows():
        print(f"  {r['country_iso']:<6} {r['score']:>7.2f} {r['event_count']:>8}")
    print()

    print("INTERPRETATION")
    r = metrics["pearson_r"]
    a = metrics["auc"]
    if r is None or a is None:
        print("  Insufficient data to interpret.")
    else:
        if a >= 0.9:
            print("  AUC >= 0.90: Signal is near-perfect (expected for geographically stable hazard).")
        elif a >= 0.75:
            print("  AUC >= 0.75: Signal has good discriminative power.")
        elif a >= 0.65:
            print("  AUC in [0.65, 0.75): Moderate discriminative power.")
        else:
            print("  AUC < 0.65: Weak discriminative power.")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Backtest the seismic signal")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(DATA_DIR, os.path.basename(output_path))

    print("Loading seismic events from DB...")
    db = SessionLocal()
    try:
        rows = _build_dataset(db)
    finally:
        db.close()
    print(f"  {len(rows)} countries with training-window data")

    df, metrics = _run_metrics(rows)
    _print_report(metrics, df)

    output = {"metrics": metrics, "countries": rows}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()