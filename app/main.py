import os
import random
import threading
import time
import json
from collections import deque
from typing import Optional, List
from . import company_search

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import get_db
from . import models, schemas
from .risk_engine import score_scenario
from . import seismic_risk
from . import flood_risk

app = FastAPI(title="AVALYOS Backend", version="2.0")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Simple per-key rate limiting (same approach as before, kept as-is)
# --------------------------------------------------------------------
RATE_LIMIT_PER_MIN = int(os.getenv("AVAL_RATE_PER_MIN", "60"))
_RATE_STORE: dict = {}
_RATE_LOCK = threading.Lock()


def check_rate_limit(key: str):
    if RATE_LIMIT_PER_MIN <= 0:
        return
    with _RATE_LOCK:
        dq = _RATE_STORE.setdefault(key or "__anon__", deque())
        now = time.time()
        while dq and dq[0] <= now - 60:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(status_code=429, detail={"error": "Rate limit exceeded"})
        dq.append(now)


def get_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    required = os.getenv("AVAL_API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail={"error": "Unauthorized"})
    check_rate_limit(x_api_key or "__anon__")
    return x_api_key


# --------------------------------------------------------------------
# Flood risk analysis: pre-computed results file
# --------------------------------------------------------------------
FLOOD_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "flood_analysis_results.json"
)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _branch_to_out(branch: models.Branch) -> schemas.BranchOut:
    return schemas.BranchOut(
        code=branch.code,
        name=branch.name,
        company=branch.company.name if branch.company else "unknown",
        continent=branch.continent or "unknown",
        country=branch.country or "unknown",
        state=branch.state or "unknown",
        city=branch.city,
        sector=branch.sector or "unknown",
        subsector=branch.subsector or "unknown",
        employees=branch.employees or 0,
    )


# --------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------
@app.get("/", summary="Health check")
def health():
    return {"status": "ok", "message": "AVALYOS backend running"}


@app.get("/companies", response_model=List[schemas.CompanyOut])
def list_companies(db: Session = Depends(get_db), api_key: Optional[str] = Depends(get_api_key)):
    companies = db.query(models.Company).all()
    return [
        schemas.CompanyOut(
            name=c.name,
            code=c.code,
            sector=c.sector,
            subsector=c.subsector,
            description=c.description,
            branch_count=len(c.branches),
        )
        for c in companies
    ]


@app.get("/companies/search", summary="Search global companies by name (GLEIF)")
def companies_search(
    q: str,
    limit: int = 20,
    api_key: Optional[str] = Depends(get_api_key),
):
    """
    Search global companies via GLEIF (2.5M+ legal entities, 230+ jurisdictions).
    Free, keyless, CC0-licensed. Minimum query length: 3 characters.
    """
    if not q or len(q.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail={"error": "Query must be at least 3 characters"},
        )
    if limit > 50:
        limit = 50
    if limit < 1:
        limit = 1
    return company_search.search_companies(q, limit=limit)


@app.get("/companies/{name}", response_model=schemas.CompanyDetailOut)
def get_company(name: str, db: Session = Depends(get_db), api_key: Optional[str] = Depends(get_api_key)):
    company = (
        db.query(models.Company)
        .filter(models.Company.name.ilike(name))
        .first()
    )
    if not company:
        raise HTTPException(status_code=404, detail={"error": "Company not found"})
    return schemas.CompanyDetailOut(
        name=company.name,
        code=company.code,
        sector=company.sector,
        subsector=company.subsector,
        description=company.description,
        branch_count=len(company.branches),
        branches=[_branch_to_out(b) for b in company.branches],
    )


@app.get("/branches/{company}", response_model=List[schemas.BranchOut])
def branches_for_company(company: str, db: Session = Depends(get_db), api_key: Optional[str] = Depends(get_api_key)):
    branches = (
        db.query(models.Branch)
        .join(models.Company)
        .filter(models.Company.name.ilike(company))
        .all()
    )
    return [_branch_to_out(b) for b in branches]


@app.get("/sample", response_model=schemas.BranchOut)
def sample_one(db: Session = Depends(get_db), api_key: Optional[str] = Depends(get_api_key)):
    """Return one random branch. Swap this for real Q# sampling later if needed."""
    branch = db.query(models.Branch).order_by(models.Branch.id).all()
    if not branch:
        raise HTTPException(status_code=404, detail={"error": "No branches in database"})
    return _branch_to_out(random.choice(branch))


@app.get("/sample/many/{n}", response_model=schemas.SampleManyOut)
def sample_many(n: int, db: Session = Depends(get_db), api_key: Optional[str] = Depends(get_api_key)):
    if n <= 0:
        raise HTTPException(status_code=400, detail={"error": "n must be > 0"})
    if n > 2000:
        raise HTTPException(status_code=400, detail={"error": "n too large; max 2000"})

    all_branches = db.query(models.Branch).all()
    if not all_branches:
        raise HTTPException(status_code=404, detail={"error": "No branches in database"})

    picks = [random.choice(all_branches) for _ in range(n)]
    samples = [_branch_to_out(b) for b in picks]

    distribution: dict = {}
    for s in samples:
        distribution[s.company] = distribution.get(s.company, 0) + 1

    return schemas.SampleManyOut(samples=samples, distribution=distribution)


@app.post("/analyze", response_model=schemas.AnalyzeResponse)
def analyze_scenario(
    req: schemas.AnalyzeRequest,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key),
):
    """Score a finance scenario: risk level + suggested action.

    Combines three signals:
      - financial  (risk_engine.score_scenario)
      - seismic    (seismic_risk.score_country_seismic_risk, if country is known)
      - flood      (flood_risk.flood_risk_for_country, if country is known)

    Weights redistribute proportionally when a signal is unavailable, so the
    verdict is never artificially depressed by a missing signal.
    """
    sector = req.sector
    employees = None

    if req.company:
        company = (
            db.query(models.Company)
            .filter(models.Company.name.ilike(req.company))
            .first()
        )
        if company:
            sector = sector or company.sector
            employees = sum(b.employees or 0 for b in company.branches)

    if req.investment_amount <= 0:
        raise HTTPException(status_code=400, detail={"error": "investment_amount must be > 0"})
    if req.time_horizon_years <= 0:
        raise HTTPException(status_code=400, detail={"error": "time_horizon_years must be > 0"})

    # --- Signal 1: Financial (always available) ---
    financial = score_scenario(
        sector=sector,
        investment_amount=req.investment_amount,
        time_horizon_years=req.time_horizon_years,
        employees=employees,
    )
    financial_score = financial["risk_score"]

    # --- Signal 2: Seismic ---
    seismic_score: Optional[float] = None
    seismic_reason: Optional[str] = None
    seismic_available = False

    if req.country:
        try:
            s = seismic_risk.score_country_seismic_risk(db, req.country, days=180)
            event_count = s.get("event_count", 0)
            if event_count > 0:
                seismic_score = s.get("risk_score")
                seismic_level = s.get("risk_level")
                seismic_reason = (
                    f"Seismic activity in {req.country}: {seismic_level} "
                    f"({event_count} event(s), max magnitude "
                    f"{s.get('max_magnitude') or '—'}) in last 180 days."
                )
            else:
                seismic_score = 0.0
                seismic_reason = (
                    f"No seismic events recorded in {req.country} in the last 180 days."
                )
            seismic_available = True
        except Exception:
            pass

    # --- Signal 3: Flood ---
    flood_score: Optional[float] = None
    flood_reasons: List[str] = []
    flood_available = False

    if req.country:
        try:
            f = flood_risk.flood_risk_for_country(req.country)
            if f.get("available") and f.get("risk_score") is not None:
                flood_score = f["risk_score"]
                flood_reasons = f.get("reasons", []) or []
                flood_available = True
        except Exception:
            pass

    # --- Weighted aggregation with proportional redistribution ---
    base_weights = {"financial": 0.50, "seismic": 0.20, "flood": 0.30}
    availability = {
        "financial": True,
        "seismic": seismic_available,
        "flood": flood_available,
    }
    available_sum = sum(w for k, w in base_weights.items() if availability[k]) or 1.0
    norm = {
        k: (base_weights[k] / available_sum if availability[k] else 0.0)
        for k in base_weights
    }

    combined = (
        financial_score * norm["financial"]
        + (seismic_score or 0.0) * norm["seismic"]
        + (flood_score or 0.0) * norm["flood"]
    )
    combined = round(combined, 1)

    if combined < 35:
        level = "Low"
    elif combined < 65:
        level = "Medium"
    else:
        level = "High"

    # --- Reasons: financial, then seismic, then flood ---
    reasons: List[str] = list(financial["reasons"])
    if seismic_reason:
        reasons.append(seismic_reason)
    reasons.extend(flood_reasons)

    # --- Suggestion derived from the COMPOSITE level ---
    from .risk_engine import _suggestion_for
    suggestion = _suggestion_for(level)

    if seismic_available and seismic_score is not None and seismic_score >= 35:
        suggestion += (
            " Seismic exposure is non-trivial — consider monitoring and contingency "
            "planning for assets in this region."
        )
    if flood_available and flood_score is not None and flood_score >= 40:
        suggestion += (
            " Historical flood recurrence is significant in this country — "
            "review physical asset placement and supply-chain contingencies."
        )

    signals = [
        schemas.SignalContribution(
            signal="financial",
            score=financial_score,
            weight=round(norm["financial"], 2),
            available=True,
        ),
        schemas.SignalContribution(
            signal="seismic",
            score=seismic_score,
            weight=round(norm["seismic"], 2),
            available=seismic_available,
        ),
        schemas.SignalContribution(
            signal="flood",
            score=flood_score,
            weight=round(norm["flood"], 2),
            available=flood_available,
        ),
    ]

    return schemas.AnalyzeResponse(
        risk_score=combined,
        risk_level=level,
        reasons=reasons,
        suggestion=suggestion,
        signals=signals,
    )


@app.get("/flood-risk", summary="Pre-computed global flood TDA risk analysis")
def flood_risk_endpoint(api_key: Optional[str] = Depends(get_api_key)):
    """
    Serves the pre-computed results of the flood TDA batch analysis
    (app/flood_analysis.py). This does NOT run the analysis live —
    persistent homology + pairwise Wasserstein distances across ~140
    countries take too long to compute per-request.

    To refresh the data, re-run:
        python app/flood_analysis.py
    from the backend root. No server restart needed — this endpoint
    reads the JSON file fresh on every request.
    """
    if not os.path.exists(FLOOD_RESULTS_PATH):
        raise HTTPException(
            status_code=404,
            detail={"error": "Flood analysis results not found. Run app/flood_analysis.py first."},
        )
    with open(FLOOD_RESULTS_PATH, "r") as f:
        return json.load(f)


@app.get("/seismic-risk/{country}", response_model=schemas.CountrySeismicRiskOut)
def seismic_risk_for_country(
    country: str,
    days: int = 90,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key),
):
    result = seismic_risk.score_country_seismic_risk(db, country, days=days)
    return schemas.CountrySeismicRiskOut(**result)


@app.get("/seismic-risk", response_model=List[schemas.CountrySeismicRiskOut])
def seismic_risk_overview(
    days: int = 90,
    min_events: int = 1,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key),
):
    results = seismic_risk.score_all_countries(db, days=days, min_events=min_events)
    return [schemas.CountrySeismicRiskOut(**r) for r in results]


@app.get("/seismic-risk/branches/exposure", response_model=List[schemas.BranchSeismicExposureOut])
def seismic_risk_branch_exposure(
    days: int = 90,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key),
):
    branches = db.query(models.Branch).all()
    results = [seismic_risk.score_branch_exposure(db, b, days=days) for b in branches]
    return [schemas.BranchSeismicExposureOut(**r) for r in results]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)