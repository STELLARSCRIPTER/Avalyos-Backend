"""
Pydantic schemas: shape data going in/out of the API.
"""

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


class AnalyzeRequest(BaseModel):
    company: Optional[str] = None
    sector: Optional[str] = None
    investment_amount: float
    time_horizon_years: float


class AnalyzeResponse(BaseModel):
    risk_score: float
    risk_level: str
    reasons: List[str]
    suggestion: str
