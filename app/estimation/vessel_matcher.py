"""
Vessel Matcher — calculates quantitative similarity scores between candidate
vessels and the target vessel specification, ranking and selecting top comparables.
"""

import math
import structlog
from dataclasses import dataclass
from typing import Optional

from app.extraction.schemas import CandidateVessel, VesselProfile

logger = structlog.get_logger(__name__)


@dataclass
class RankedComparable:
    """A candidate vessel scored and ranked against the target specification."""
    candidate: CandidateVessel
    similarity_score: float         # 0.0 to 1.0 (1.0 = perfect match)
    type_score: float               # 0.0 to 1.0
    displacement_score: float       # 0.0 to 1.0
    length_score: float             # 0.0 to 1.0
    recency_score: float            # 0.0 to 1.0
    source_quality_score: float     # 0.0 to 1.0
    match_reasons: list[str]


def rank_and_select_comparables(
    profile: VesselProfile,
    candidates: list[CandidateVessel],
    max_selected: int = 5,
    min_score_threshold: float = 0.30,
) -> list[RankedComparable]:
    """
    Score and rank candidate vessels based on multi-parameter similarity
    to the target vessel specification.

    Weights:
      - Vessel Type & Mission Match: 35%
      - Displacement Match Ratio:   25%
      - Length Match Ratio:         15%
      - Delivery Year Recency:       15%
      - Source Quality & Tier:       10%

    Args:
        profile: Target vessel specification profile.
        candidates: Discovered candidate vessels.
        max_selected: Maximum comparables to return.
        min_score_threshold: Minimum score threshold (0.0 - 1.0).

    Returns:
        List of RankedComparable objects sorted by similarity_score descending.
    """
    target_spec = profile.specification
    ranked_list: list[RankedComparable] = []

    target_type = _get_spec_value(target_spec.vessel_type)
    target_disp = _get_numeric_value(target_spec.displacement_t)
    target_len = _get_numeric_value(target_spec.length_m)

    for c in candidates:
        reasons: list[str] = []

        # 1. Type & Mission Match (Weight: 0.35)
        type_score = _calculate_type_score(target_type, c.vessel_type or c.vessel_name)
        if type_score > 0.7:
            reasons.append(f"Matching vessel type: '{c.vessel_type or c.vessel_name}'")

        # 2. Displacement Match (Weight: 0.25)
        disp_score = 0.5  # Neutral default if missing
        if target_disp and c.displacement_t and target_disp > 0:
            ratio = abs(c.displacement_t - target_disp) / target_disp
            disp_score = max(0.0, 1.0 - ratio)
            if disp_score > 0.7:
                reasons.append(f"Close displacement match ({c.displacement_t:.0f}t vs target {target_disp:.0f}t)")

        # 3. Length Match (Weight: 0.15)
        len_score = 0.5  # Neutral default if missing
        if target_len and c.length_m and target_len > 0:
            ratio = abs(c.length_m - target_len) / target_len
            len_score = max(0.0, 1.0 - ratio)
            if len_score > 0.7:
                reasons.append(f"Close length match ({c.length_m:.1f}m vs target {target_len:.1f}m)")

        # 4. Recency Score (Weight: 0.15)
        # Prefers vessels delivered in last 5–10 years (2015-2026)
        recency_score = 0.5
        if c.delivery_year or c.contract_year:
            year = c.delivery_year or c.contract_year
            years_diff = abs(2026 - year)
            recency_score = max(0.0, 1.0 - (years_diff / 15.0))
            if recency_score > 0.7:
                reasons.append(f"Recent delivery/contract year: {year}")

        # 5. Source Quality & Tier (Weight: 0.10)
        source_quality_score = max(0.2, (5.0 - c.source_tier) / 4.0)

        # Total Weighted Similarity Score
        total_score = (
            (type_score * 0.35)
            + (disp_score * 0.25)
            + (len_score * 0.15)
            + (recency_score * 0.15)
            + (source_quality_score * 0.10)
        )

        # Extra bonus if contract cost is available (Category A)
        if c.contract_value and c.contract_value > 0:
            total_score = min(1.0, total_score + 0.05)
            reasons.append("Verified contract cost data available")

        if total_score >= min_score_threshold:
            ranked_list.append(
                RankedComparable(
                    candidate=c,
                    similarity_score=round(total_score, 3),
                    type_score=round(type_score, 2),
                    displacement_score=round(disp_score, 2),
                    length_score=round(len_score, 2),
                    recency_score=round(recency_score, 2),
                    source_quality_score=round(source_quality_score, 2),
                    match_reasons=reasons,
                )
            )

    # Sort descending by similarity score
    ranked_list.sort(key=lambda r: r.similarity_score, reverse=True)

    # Filter top comparables (preferring those with cost data)
    selected = ranked_list[:max_selected]

    logger.info(
        "vessel_matching_complete",
        run_id=profile.run_id,
        total_candidates=len(candidates),
        ranked_count=len(ranked_list),
        selected_count=len(selected),
        top_score=selected[0].similarity_score if selected else 0.0,
    )

    return selected


def _calculate_type_score(target_type: Optional[str], candidate_type: Optional[str]) -> float:
    """Calculate matching score between vessel types."""
    if not target_type or not candidate_type:
        return 0.5

    t1 = target_type.lower()
    t2 = candidate_type.lower()

    if t1 in t2 or t2 in t1:
        return 1.0

    # Common synonyms / categories
    keywords = ["opv", "patrol", "corvette", "frigate", "research", "seismic", "auxiliary", "tanker", "lpd"]
    for kw in keywords:
        if kw in t1 and kw in t2:
            return 0.9

    return 0.3


def _get_spec_value(extracted) -> Optional[str]:
    """Safely get string value from ExtractedValue."""
    if extracted is None:
        return None
    if isinstance(extracted, dict):
        val = extracted.get("value")
        return str(val) if val is not None else None
    if hasattr(extracted, "value"):
        val = getattr(extracted, "value")
        return str(val) if val is not None else None
    return str(extracted)


def _get_numeric_value(extracted) -> Optional[float]:
    """Safely get numeric float value from ExtractedValue."""
    if extracted is None:
        return None
    val = None
    if isinstance(extracted, dict):
        val = extracted.get("value")
    elif hasattr(extracted, "value"):
        val = getattr(extracted, "value")
    else:
        val = extracted

    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
