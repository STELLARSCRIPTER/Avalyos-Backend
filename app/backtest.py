"""
app/backtest.py

Backtest the flood signal against historical outcomes.

Method:
  - For each of the 141 countries in flood_analysis_results.json, we have
    a flood risk score (0-100) computed from long-run historical patterns.
  - Ground truth: did that country experience >=1 flood event in 2019-2023?
    (from EM-DAT, filtered to Disaster Type == 'Flood')
  - Metrics:
      * Pearson correlation between score and event count
      * AUC of score vs binary label (event occurred: yes/no)

We use the full-history score, not a leave-out-window score. This is a
mild temporal-leakage approximation — documented in the README and revisited
in a later pass if results warrant.

Output:
  - app/data/backtest_results.json
  - Printed summary table to stdout

Run:
  python app/backtest.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FLOOD_RESULTS = os.path.join(DATA_DIR, "flood_analysis_results.json")
EMDAT_CSV = os.path.join(DATA_DIR, "public_emdat_project.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "backtest_results.json")

TEST_START_YEAR = 2019
TEST_END_YEAR = 2023


def _load_flood_scores() -> dict:
    """
    Returns { country_name: {score, risk_level, mean_damage_m, mean_deaths} }
    from the TDA results JSON.
    """
    with open(FLOOD_RESULTS, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = data.get("step6_country_topology", {}).get("country_points", [])
    out = {}
    for entry in points:
        name = entry.get("country")
        if not name:
            continue
        # Recompute the same score the runtime uses.
        # Kept in sync with flood_risk.py — if that file's weights change,
        # this must too. (Future refactor: import the scoring fn directly.)
        out[name] = {
            "mean_damage_m": float(entry.get("mean_damage_m") or 0.0),
            "mean_deaths": float(entry.get("mean_deaths") or 0.0),
            "dominant_driver": entry.get("dominant_driver"),
            "bob_ratio": float(entry.get("bob_ratio") or 0.0),
            "elnino_ratio": float(entry.get("elnino_ratio") or 0.0),
            "piod_ratio": float(entry.get("piod_ratio") or 0.0),
        }
    return out


def _compute_scores(raw_scores: dict) -> dict:
    """
    Compute the same composite score that flood_risk.py uses, so this script
    is self-contained and reproducible. Mirrors the formula in flood_risk.py:
        damage_norm  * 0.40
      + deaths_norm  * 0.25
      + driver_score * 0.20
      + neighbor_score * 0.15
    Neighbor scoring is omitted here to keep the backtest independent of
    the Wasserstein matrix — the damage + deaths + driver alone capture ~85%
    of the signal. Documented approximation.
    """
    max_damage = max((v["mean_damage_m"] for v in raw_scores.values()), default=1.0) or 1.0
    max_deaths = max((v["mean_deaths"] for v in raw_scores.values()), default=1.0) or 1.0

    scored = {}
    for country, v in raw_scores.items():
        damage_norm = min(100.0, (v["mean_damage_m"] / max_damage) * 100.0)
        deaths_norm = min(100.0, (v["mean_deaths"] / max_deaths) * 100.0)

        driver = (v["dominant_driver"] or "").strip()
        if driver == "BoB":
            ratio = v["bob_ratio"]
        elif driver == "El Nino":
            ratio = v["elnino_ratio"]
        elif driver == "pIOD":
            ratio = v["piod_ratio"]
        else:
            ratio = 1.0
        driver_score = min(100.0, max(0.0, (ratio / 3.0) * 100.0))

        score = (
            damage_norm * 0.40
            + deaths_norm * 0.25
            + driver_score * 0.20
        )
        scored[country] = round(score, 2)
    return scored


def _load_test_events() -> pd.DataFrame:
    """
    Load EM-DAT, filter to Flood events in the test window.
    Returns df with columns: Country, EventCount
    """
    df = pd.read_csv(EMDAT_CSV, encoding="latin-1", low_memory=False)

    df = df[df["Disaster Type"] == "Flood"].copy()
    df["Start Year"] = pd.to_numeric(df["Start Year"], errors="coerce")
    df = df.dropna(subset=["Start Year", "Country"])

    window = df[
        (df["Start Year"] >= TEST_START_YEAR) & (df["Start Year"] <= TEST_END_YEAR)
    ]

    counts = window.groupby("Country").size().reset_index(name="EventCount")
    return counts


def _build_dataset():
    """
    Join scores with event counts. Returns a pandas DataFrame with:
        country, score, event_count, has_event
    """
    raw = _load_flood_scores()
    scored = _compute_scores(raw)

    events = _load_test_events()
    event_map = dict(zip(events["Country"], events["EventCount"]))

    rows = []
    for country, score in scored.items():
        count = int(event_map.get(country, 0))
        rows.append({
            "country": country,
            "score": float(score),
            "event_count": count,
            "has_event": 1 if count > 0 else 0,
        })
    return pd.DataFrame(rows)


def _run_metrics(df: pd.DataFrame) -> dict:
    """
    Compute correlation and AUC. Returns a dict with all stats.
    """
    n = len(df)
    n_pos = int(df["has_event"].sum())
    n_neg = n - n_pos

    # Only compute metrics if both classes are present
    corr_r = corr_p = None
    auc = None
    if df["score"].nunique() > 1 and df["event_count"].nunique() > 1:
        try:
            corr_r, corr_p = pearsonr(df["score"], df["event_count"])
        except Exception:
            corr_r, corr_p = None, None

    if n_pos > 0 and n_neg > 0:
        try:
            auc = roc_auc_score(df["has_event"], df["score"])
        except Exception:
            auc = None

    return {
        "n_countries": n,
        "n_with_flood_2019_2023": n_pos,
        "n_without_flood_2019_2023": n_neg,
        "test_window": f"{TEST_START_YEAR}-{TEST_END_YEAR}",
        "pearson_r": round(corr_r, 4) if corr_r is not None else None,
        "pearson_p": round(corr_p, 6) if corr_p is not None else None,
        "auc": round(auc, 4) if auc is not None else None,
        "mean_score_flood_countries": round(
            float(df[df["has_event"] == 1]["score"].mean()), 2
        ) if n_pos > 0 else None,
        "mean_score_no_flood_countries": round(
            float(df[df["has_event"] == 0]["score"].mean()), 2
        ) if n_neg > 0 else None,
    }


def _print_report(metrics: dict, df: pd.DataFrame):
    """
    Human-readable CLI report.
    """
    print("=" * 68)
    print("FLOOD SIGNAL BACKTEST REPORT")
    print("=" * 68)
    print(f"Test window:              {metrics['test_window']}")
    print(f"Countries tested:         {metrics['n_countries']}")
    print(f"  With flood events:      {metrics['n_with_flood_2019_2023']}")
    print(f"  Without flood events:   {metrics['n_without_flood_2019_2023']}")
    print()
    print("SIGNAL POWER")
    print(f"  Pearson r (score vs event count):  {metrics['pearson_r']}")
    print(f"  Pearson p-value:                   {metrics['pearson_p']}")
    print(f"  AUC (score vs event occurred):     {metrics['auc']}")
    print()
    print("SCORE COMPARISON")
    print(f"  Mean score (countries with flood): {metrics['mean_score_flood_countries']}")
    print(f"  Mean score (countries without):    {metrics['mean_score_no_flood_countries']}")
    print()

    # Top/bottom score countries
    top = df.sort_values("score", ascending=False).head(10)
    bottom = df.sort_values("score", ascending=True).head(10)

    print("TOP 10 SCORED COUNTRIES")
    print(f"  {'COUNTRY':<35} {'SCORE':>7} {'EVENTS':>8}")
    for _, r in top.iterrows():
        print(f"  {r['country'][:34]:<35} {r['score']:>7.2f} {r['event_count']:>8}")
    print()

    print("BOTTOM 10 SCORED COUNTRIES")
    print(f"  {'COUNTRY':<35} {'SCORE':>7} {'EVENTS':>8}")
    for _, r in bottom.iterrows():
        print(f"  {r['country'][:34]:<35} {r['score']:>7.2f} {r['event_count']:>8}")
    print()

    # Interpretation
    print("INTERPRETATION")
    r = metrics["pearson_r"]
    a = metrics["auc"]
    if r is None or a is None:
        print("  Insufficient data to interpret.")
    else:
        if a >= 0.75:
            print("  AUC >= 0.75: Signal has good discriminative power.")
        elif a >= 0.65:
            print("  AUC in [0.65, 0.75): Signal has moderate discriminative power.")
        elif a >= 0.55:
            print("  AUC in [0.55, 0.65): Signal is weak — score is only slightly informative.")
        else:
            print("  AUC < 0.55: Signal is essentially noise. Revisit weights/features.")

        if r >= 0.3:
            print(f"  Correlation r = {r}: moderate-to-strong positive relationship.")
        elif r >= 0.15:
            print(f"  Correlation r = {r}: weak positive relationship.")
        else:
            print(f"  Correlation r = {r}: little-to-no linear relationship.")
    print("=" * 68)


def main():
    print("Loading TDA scores...")
    raw = _load_flood_scores()
    print(f"  {len(raw)} countries in flood_analysis_results.json")

    print(f"Loading EM-DAT events ({TEST_START_YEAR}-{TEST_END_YEAR})...")
    events = _load_test_events()
    print(f"  {events.shape[0]} countries with >=1 flood event in window")

    df = _build_dataset()
    metrics = _run_metrics(df)

    _print_report(metrics, df)

    output = {
        "metrics": metrics,
        "countries": df.to_dict(orient="records"),
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()