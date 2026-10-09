from typing import Optional, List, Dict
from pydantic import BaseModel, ConfigDict


class BranchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    name: Optional[str] = None
    company: str
    continent: str
    country: str
    state: str
    city: Optional[str] = None
    sector: str
    subsector: str
    employees: int


class CompanyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    code: str
    sector: str
    subsector: str
    description: Optional[str] = None
    branch_count: int


class CompanyDetailOut(CompanyOut):
    branches: List[BranchOut]


class SampleManyOut(BaseModel):
    samples: List[BranchOut]
    distribution: Dict[str, int]


class SignalContribution(BaseModel):
    signal: str
    score: Optional[float] = None
    weight: float
    available: bool


class AnalyzeRequest(BaseModel):
    company: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    investment_amount: float
    time_horizon_years: float


class AnalyzeResponse(BaseModel):
    risk_score: float
    risk_level: str
    reasons: List[str]
    suggestion: str
    signals: List[SignalContribution] = []


class CountrySeismicRiskOut(BaseModel):
    country: str
    window_days: int
    risk_score: float
    risk_level: str
    event_count: int
    max_magnitude: Optional[float] = None
    avg_magnitude: Optional[float] = None
    reasons: List[str]


class BranchSeismicExposureOut(BaseModel):
    branch_code: str
    branch_name: Optional[str] = None
    company: Optional[str] = None
    country: Optional[str] = None
    window_days: Optional[int] = None
    risk_score: Optional[float] = None
    risk_level: str
    event_count: Optional[int] = None
    max_magnitude: Optional[float] = None
    avg_magnitude: Optional[float] = None
    reasons: List[str]


    # --------------------------------------------------------------------
# Region lookup (for /regions endpoint and frontend dropdowns)
# --------------------------------------------------------------------
class RegionOut(BaseModel):
    iso_code: str
    name: str
    flag_url: Optional[str] = None


# --------------------------------------------------------------------
# Compare mode: two scenarios, one company, different regions
# --------------------------------------------------------------------
class CompareRequest(BaseModel):
    company: Optional[str] = None
    sector: Optional[str] = None
    investment_amount: float
    time_horizon_years: float
    region_a: str  # ISO-2
    region_b: str  # ISO-2


class ComparisonSummary(BaseModel):
    riskier_region: str          # "a" | "b" | "equal"
    score_difference: float      # |score_a - score_b|, rounded
    level_change: str            # e.g. "MEDIUM → LOW", "LOW → LOW"
    top_diverging_signal: str    # "financial" | "seismic" | "flood"
    diverging_signal_delta: float
    summary_line: str            # one-line human-readable suggestion


class CompareResponse(BaseModel):
    company: Optional[str]
    region_a: str
    region_b: str
    result_a: AnalyzeResponse
    result_b: AnalyzeResponse
    summary: ComparisonSummary

    