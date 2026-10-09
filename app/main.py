import os
import random
import threading
import time
import json
from collections import deque
from typing import Optional, List
from . import company_search
from . import decision

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import get_db
from . import models, schemas

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


@app.get("/regions", response_model=List[schemas.RegionOut], summary="List all ISO 3166-1 countries")
def list_regions(api_key: Optional[str] = Depends(get_api_key)):
    """
    Full country list from pycountry. Cached at module load — this
    endpoint is fast and stable.

    flag_url points at flagcdn.com (free, no key). Renders on all
    platforms including Windows, unlike emoji flags.
    """
    return _REGIONS


# Cached at import time; pycountry's country list doesn't change at runtime.
import pycountry as _pycountry  # noqa: E402


def _build_regions() -> List[dict]:
    out = []
    for c in sorted(_pycountry.countries, key=lambda x: x.name):
        iso = getattr(c, "alpha_2", None)
        if not iso:
            continue
        out.append({
            "iso_code": iso,
            "name": c.name,
            "flag_url": f"https://flagcdn.com/w40/{iso.lower()}.png",
        })
    return out


_REGIONS = _build_regions()


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
    """Score a single scenario. Core logic lives in app/decision.py."""
    if req.investment_amount <= 0:
        raise HTTPException(status_code=400, detail={"error": "investment_amount must be > 0"})
    if req.time_horizon_years <= 0:
        raise HTTPException(status_code=400, detail={"error": "time_horizon_years must be > 0"})

    result = decision.analyze_one(
        {
            "company": req.company,
            "sector": req.sector,
            "country": req.country,
            "investment_amount": req.investment_amount,
            "time_horizon_years": req.time_horizon_years,
        },
        db,
    )
    return schemas.AnalyzeResponse(**result)


@app.post("/compare", response_model=schemas.CompareResponse)
def compare_scenarios(
    req: schemas.CompareRequest,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key),
):
    """
    Run the same scenario against two regions and return both verdicts
    plus a summary of the divergence.
    """
    if req.investment_amount <= 0:
        raise HTTPException(status_code=400, detail={"error": "investment_amount must be > 0"})
    if req.time_horizon_years <= 0:
        raise HTTPException(status_code=400, detail={"error": "time_horizon_years must be > 0"})
    if not req.region_a or not req.region_b:
        raise HTTPException(status_code=400, detail={"error": "region_a and region_b are required"})

    base = {
        "company": req.company,
        "sector": req.sector,
        "investment_amount": req.investment_amount,
        "time_horizon_years": req.time_horizon_years,
    }

    result_a = decision.analyze_one({**base, "country": req.region_a}, db)
    result_b = decision.analyze_one({**base, "country": req.region_b}, db)

    score_a = result_a["risk_score"]
    score_b = result_b["risk_score"]
    diff = round(abs(score_a - score_b), 1)

    if score_a > score_b:
        riskier = "a"
    elif score_b > score_a:
        riskier = "b"
    else:
        riskier = "equal"

    level_change = f"{result_a['risk_level'].upper()} → {result_b['risk_level'].upper()}"

    sigs_a = {s["signal"]: s["score"] or 0.0 for s in result_a["signals"]}
    sigs_b = {s["signal"]: s["score"] or 0.0 for s in result_b["signals"]}
    deltas = {
        k: abs(sigs_a.get(k, 0.0) - sigs_b.get(k, 0.0))
        for k in set(sigs_a) | set(sigs_b)
    }
    top_signal = max(deltas, key=deltas.get) if deltas else "financial"
    top_delta = round(deltas.get(top_signal, 0.0), 1)

    if riskier == "a":
        summary_line = (
            f"{req.region_a} is riskier by {diff} points — "
            f"divergence driven mainly by {top_signal}."
        )
    elif riskier == "b":
        summary_line = (
            f"{req.region_b} is riskier by {diff} points — "
            f"divergence driven mainly by {top_signal}."
        )
    else:
        summary_line = "Both regions score identically for this scenario."

    summary = schemas.ComparisonSummary(
        riskier_region=riskier,
        score_difference=diff,
        level_change=level_change,
        top_diverging_signal=top_signal,
        diverging_signal_delta=top_delta,
        summary_line=summary_line,
    )

    return schemas.CompareResponse(
        company=req.company,
        region_a=req.region_a,
        region_b=req.region_b,
        result_a=schemas.AnalyzeResponse(**result_a),
        result_b=schemas.AnalyzeResponse(**result_b),
        summary=summary,
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
    from . import seismic_risk
    result = seismic_risk.score_country_seismic_risk(db, country, days=days)
    return schemas.CountrySeismicRiskOut(**result)


@app.get("/seismic-risk", response_model=List[schemas.CountrySeismicRiskOut])
def seismic_risk_overview(
    days: int = 90,
    min_events: int = 1,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key),
):
    from . import seismic_risk
    results = seismic_risk.score_all_countries(db, days=days, min_events=min_events)
    return [schemas.CountrySeismicRiskOut(**r) for r in results]


@app.get("/seismic-risk/branches/exposure", response_model=List[schemas.BranchSeismicExposureOut])
def seismic_risk_branch_exposure(
    days: int = 90,
    db: Session = Depends(get_db),
    api_key: Optional[str] = Depends(get_api_key),
):
    from . import seismic_risk
    branches = db.query(models.Branch).all()
    results = [seismic_risk.score_branch_exposure(db, b, days=days) for b in branches]
    return [schemas.BranchSeismicExposureOut(**r) for r in results]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)