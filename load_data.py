"""
One-time (or repeatable) loader: creates Postgres tables and loads
companies.json into the companies/branches tables.

Usage:
    python load_data.py [path/to/companies.json]

Run generate_database.py first if you don't have companies.json yet.
"""

import json
import sys
import os

from app.database import Base, engine, SessionLocal
from app import models


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables created (or already existed).")


def load_companies(json_path: str):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        print("Run generate_database.py first to create companies.json.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    companies_block = data.get("companies", data)

    db = SessionLocal()
    try:
        created_companies = 0
        created_branches = 0

        for company_name, company_info in companies_block.items():
            existing = (
                db.query(models.Company)
                .filter(models.Company.name == company_name)
                .first()
            )
            if existing:
                company_row = existing
            else:
                company_row = models.Company(
                    name=company_name,
                    code=company_info.get("code", ""),
                    sector=company_info.get("sector", "unknown"),
                    subsector=company_info.get("subsector", "unknown"),
                    description=company_info.get("description", ""),
                )
                db.add(company_row)
                db.flush()  # get company_row.id
                created_companies += 1

            branches = company_info.get("branches", {})
            for branch_code, branch_info in branches.items():
                existing_branch = (
                    db.query(models.Branch)
                    .filter(models.Branch.code == branch_code)
                    .first()
                )
                if existing_branch:
                    continue

                branch_row = models.Branch(
                    code=branch_code,
                    name=branch_info.get("name", ""),
                    company_id=company_row.id,
                    continent=branch_info.get("continent", "unknown"),
                    country=branch_info.get("country", "unknown"),
                    state=branch_info.get("state", "unknown"),
                    city=branch_info.get("city", "unknown"),
                    employees=int(branch_info.get("employees", 0) or 0),
                    sector=branch_info.get("sector", "unknown"),
                    subsector=branch_info.get("subsector", "unknown"),
                    description=branch_info.get("description", ""),
                )
                db.add(branch_row)
                created_branches += 1

        db.commit()
        print(f"Loaded {created_companies} new companies, {created_branches} new branches.")
    finally:
        db.close()


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "companies.json"
    create_tables()
    load_companies(json_path)
