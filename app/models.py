"""
SQLAlchemy ORM models. Postgres is now the single source of truth
(MongoDB has been removed).
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    code = Column(String, index=True)
    sector = Column(String)
    subsector = Column(String)
    description = Column(String)

    branches = relationship(
        "Branch", back_populates="company", cascade="all, delete-orphan"
    )


class Branch(Base):
    __tablename__ = "branches"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)

    continent = Column(String)
    country = Column(String)
    state = Column(String)
    city = Column(String)
    employees = Column(Integer, default=0)
    sector = Column(String)
    subsector = Column(String)
    description = Column(String)

    company = relationship("Company", back_populates="branches")


class UserAction(Base):
    """Optional: log user interactions (search, select, simulate, etc.)."""

    __tablename__ = "user_actions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String)
    action_type = Column(String)
    target = Column(String)
    details = Column(JSON)


class SimulationResult(Base):
    """Optional: store quantum/Monte Carlo simulation results."""

    __tablename__ = "simulation_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    simulation_type = Column(String)
    company = Column(String)
    branch = Column(String)
    n_samples = Column(Integer)
    results = Column(JSON)
    extra_metadata = Column(JSON)
