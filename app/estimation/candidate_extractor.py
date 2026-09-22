"""
Candidate Vessel Extractor — discovers potential comparable vessels from
retrieved webpage contents and search result snippets.
"""

import json
import re
import structlog
from typing import Optional

from app.llm.client import LLMClient
from app.llm.retry import parse_json_response
from app.extraction.prompts import (
    PROMPT_C_SYSTEM,
    PROMPT_C_USER_TEMPLATE,
    format_spec_for_prompt,
)
from app.extraction.schemas import CandidateVessel, RetrievedPage, SearchResult, VesselProfile

logger = structlog.get_logger(__name__)


def discover_candidate_vessels(
    retrieved_pages: list[RetrievedPage],
    search_results: list[SearchResult],
    profile: VesselProfile,
    llm_client: Optional[LLMClient] = None,
    max_pages_to_analyze: int = 10,
) -> list[CandidateVessel]:
    """
    Extract structured CandidateVessel models from retrieved webpage contents
    and search result snippets.

    Args:
        retrieved_pages: List of fetched and cleaned webpage contents.
        search_results: List of web search results.
        profile: Target vessel profile.
        llm_client: Optional LLM client.
        max_pages_to_analyze: Max pages to send to LLM for deep analysis.

    Returns:
        List of deduplicated CandidateVessel objects.
    """
    candidates: list[CandidateVessel] = []
    seen_names: set[str] = set()

    # 1. Analyze top accessible retrieved pages
    accessible_pages = [p for p in retrieved_pages if p.is_accessible and p.content][:max_pages_to_analyze]

    if llm_client is None:
        try:
            llm_client = LLMClient()
        except Exception:
            llm_client = None

    for page in accessible_pages:
        page_candidates = _extract_candidates_from_page(page, profile, llm_client)
        for c in page_candidates:
            name_key = c.vessel_name.strip().lower()
            if name_key and name_key not in seen_names:
                seen_names.add(name_key)
                candidates.append(c)

    # 2. Heuristic extraction from search result snippets for additional candidate discovery
    snippet_candidates = _extract_candidates_from_snippets(search_results, profile)
    for c in snippet_candidates:
        name_key = c.vessel_name.strip().lower()
        if name_key and name_key not in seen_names:
            seen_names.add(name_key)
            candidates.append(c)

    logger.info(
        "candidate_discovery_complete",
        run_id=profile.run_id,
        total_candidates=len(candidates),
        category_a=len([c for c in candidates if c.data_category == "A"]),
        category_b=len([c for c in candidates if c.data_category == "B"]),
    )

    return candidates


def _extract_candidates_from_page(
    page: RetrievedPage,
    profile: VesselProfile,
    llm_client: Optional[LLMClient],
) -> list[CandidateVessel]:
    """Use LLM (with regex fallback) to extract candidates from a single page."""
    candidates: list[CandidateVessel] = []

    # Truncate page text for LLM prompt
    content_excerpt = page.content[:12_000]

    if llm_client is not None:
        try:
            messages = [
                {"role": "system", "content": PROMPT_C_SYSTEM},
                {
                    "role": "user",
                    "content": PROMPT_C_USER_TEMPLATE.format(
                        url=page.url,
                        source_tier=page.source_tier,
                        content=content_excerpt,
                    ),
                },
            ]
            raw = llm_client.complete_json(messages=messages, temperature=0.1, max_tokens=2048)
            parsed = parse_json_response(raw)

            candidate_dicts = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            for cd in candidate_dicts:
                if isinstance(cd, dict) and cd.get("vessel_name"):
                    c = CandidateVessel(
                        vessel_name=str(cd["vessel_name"]),
                        vessel_type=cd.get("vessel_type"),
                        shipbuilder=cd.get("shipbuilder"),
                        operator=cd.get("operator"),
                        country=cd.get("country"),
                        delivery_year=cd.get("delivery_year"),
                        length_m=_to_float(cd.get("length_m")),
                        beam_m=_to_float(cd.get("beam_m")),
                        draft_m=_to_float(cd.get("draft_m")),
                        displacement_t=_to_float(cd.get("displacement_t")),
                        speed_knots=_to_float(cd.get("speed_knots")),
                        mission=cd.get("mission"),
                        contract_value=_to_float(cd.get("contract_value")),
                        contract_currency=cd.get("contract_currency") or "USD",
                        contract_year=cd.get("contract_year"),
                        vessel_count_in_contract=cd.get("vessel_count_in_contract") or 1,
                        program_includes_support=cd.get("program_includes_support"),
                        source_urls=[page.url],
                        source_tier=page.source_tier,
                        data_category="A" if cd.get("contract_value") else "B",
                        raw_snippets=[content_excerpt[:300]],
                    )
                    candidates.append(c)
        except Exception as e:
            logger.debug("llm_page_candidate_extraction_failed", url=page.url, error=str(e))

    # Regex heuristic fallback if LLM returned nothing or was unavailable
    if not candidates:
        heuristic_c = _heuristic_candidate_from_text(page.content, page.url, page.source_tier)
        if heuristic_c:
            candidates.append(heuristic_c)

    return candidates


def _extract_candidates_from_snippets(
    results: list[SearchResult],
    profile: VesselProfile,
) -> list[CandidateVessel]:
    """Extract candidate vessels from search result snippets using regex heuristics."""
    candidates: list[CandidateVessel] = []

    for r in results:
        text = f"{r.title} {r.snippet}"
        c = _heuristic_candidate_from_text(text, r.url, r.source_tier)
        if c:
            candidates.append(c)

    return candidates


def _heuristic_candidate_from_text(text: str, url: str, source_tier: int) -> Optional[CandidateVessel]:
    """Regex-based candidate extractor for financial & vessel specifications."""
    # Look for vessel class or vessel name patterns (e.g., "Inshore Patrol Vessel", "Hamilton-class", "P-15B")
    # Extract candidate vessel name
    name_match = re.search(
        r"\b([A-Z0-9][a-zA-Z0-9\-\s]{1,30}?(?:Class|Patrol Vessel|Corvette|Frigate|OPV|Ship|Vessel)s?)\b",
        text,
        re.I,
    )
    if not name_match:
        return None

    vessel_name = name_match.group(1).strip()

    # Extract contract cost ($XX million, €XX million, ₹XX crore)
    cost_val = None
    currency = "INR"
    cost_match = re.search(r"(\$|€|£|₹|USD|EUR|GBP|INR)\s?([\d,]+(?:\.\d+)?)\s*(million|billion|crore|m|b)?", text, re.I)
    if cost_match:
        curr_str = cost_match.group(1)
        raw_num = float(cost_match.group(2).replace(",", ""))
        unit_multiplier = (cost_match.group(3) or "").lower()

        if curr_str in ("€", "EUR"):
            currency = "EUR"
        elif curr_str in ("£", "GBP"):
            currency = "GBP"
        elif curr_str in ("₹", "INR"):
            currency = "INR"

        multiplier = 1.0
        if unit_multiplier in ("million", "m"):
            multiplier = 1_000_000.0
        elif unit_multiplier in ("billion", "b"):
            multiplier = 1_000_000_000.0
        elif unit_multiplier == "crore":
            multiplier = 10_000_000.0  # 1 Crore = 10 Million

        cost_val = raw_num * multiplier

    # Extract vessel count in contract ("contract for 2 vessels", "2 Hamilton-class Offshore Patrol Vessels")
    count_match = re.search(r"\b(\d+)\s+(?:[a-zA-Z0-9\-]+\s+)*(?:vessels|ships|opvs|patrol vessels|units|boats)\b", text, re.I)
    vessel_count = int(count_match.group(1)) if count_match else 1

    # Extract length or displacement
    disp_match = re.search(r"([\d,]+)\s*(?:tonne|tonnes|tons|t)\b", text, re.I)
    displacement = float(disp_match.group(1).replace(",", "")) if disp_match else None

    len_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|metres)\b", text, re.I)
    length = float(len_match.group(1)) if len_match else None

    # Extract delivery year
    year_match = re.search(r"\b(20[12]\d)\b", text)
    year = int(year_match.group(1)) if year_match else None

    category = "A" if cost_val else "B"

    return CandidateVessel(
        vessel_name=vessel_name,
        vessel_type=vessel_name,
        delivery_year=year,
        length_m=length,
        displacement_t=displacement,
        contract_value=cost_val,
        contract_currency=currency,
        contract_year=year or 2020,
        vessel_count_in_contract=vessel_count,
        source_urls=[url],
        source_tier=source_tier,
        data_category=category,
        raw_snippets=[text[:200]],
    )


def _to_float(val) -> Optional[float]:
    """Helper to safely coerce values to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
