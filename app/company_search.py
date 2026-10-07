"""
app/company_search.py

Wraps the GLEIF API for company lookup. GLEIF covers ~2.5M legal entities
across 230+ jurisdictions, is free, keyless, and CC0-licensed.

Used by the frontend Scenario Builder to populate company autocomplete.
"""
from typing import Optional
from pygleif import GleifClient

_client = GleifClient()


def _extract_company(record) -> Optional[dict]:
    """
    Normalize a GLEIF record into a flat dict for our API response.
    Returns None if the record is malformed.
    """
    try:
        entity = record.attributes.entity
        legal_name = entity.legal_name.name if entity.legal_name else None
        if not legal_name:
            return None
        return {
            "lei": record.attributes.lei,
            "name": legal_name,
            "status": entity.status or "UNKNOWN",
            "jurisdiction": entity.jurisdiction or None,
            "country": entity.legal_address.country if entity.legal_address else None,
            "city": entity.legal_address.city if entity.legal_address else None,
        }
    except Exception:
        return None


def search_companies(query: str, limit: int = 20) -> list[dict]:
    """
    Full-text search for companies by name.

    GLEIF matching is loose (searching 'Apple' returns thousands of results),
    so we cap the result set and let the UI show country/status for
    disambiguation.
    """
    if not query or len(query.strip()) < 3:
        return []

    try:
        response = _client.search_fulltext(query.strip())
    except Exception:
        return []

    results = []
    for record in getattr(response, "data", []) or []:
        company = _extract_company(record)
        if company:
            results.append(company)
        if len(results) >= limit:
            break

    return results