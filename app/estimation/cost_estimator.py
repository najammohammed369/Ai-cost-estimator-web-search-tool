"""
Cost Estimator — normalizes comparable vessel costs, adjusts for inflation
and contract scope, and produces a structured CostEstimate range.
"""

import math
import structlog
from typing import Optional

from app.extraction.schemas import CostEstimate, VesselProfile
from app.estimation.vessel_matcher import RankedComparable

logger = structlog.get_logger(__name__)

# Currency conversion to USD (approximate reference rates)
EXCHANGE_RATES_TO_USD = {
    "USD": 1.0,
    "EUR": 1.10,
    "GBP": 1.28,
    "INR": 0.012,     # ~83 INR = 1 USD (1 Crore INR ~ $120,000 USD)
    "SGD": 0.75,
    "AUD": 0.67,
    "CAD": 0.74,
}

# Annual naval shipbuilding inflation rate assumption (~3.0% per annum)
ANNUAL_INFLATION_RATE = 0.030


def estimate_vessel_cost(
    profile: VesselProfile,
    selected_comparables: list[RankedComparable],
    target_year: int = 2026,
) -> CostEstimate:
    """
    Calculate total vessel cost estimate range (Low, Central/P50, High)
    by normalizing comparable vessel costs.

    Args:
        profile: Target vessel profile.
        selected_comparables: Ranked comparable vessels with cost data.
        target_year: Cost estimation target year (e.g., 2026).

    Returns:
        CostEstimate model with low_usd, central_usd, high_usd, confidence, methodology, and warnings.
    """
    warnings: list[str] = []

    # 1. Filter comparables with valid contract values
    valid_comparables = [
        rc for rc in selected_comparables
        if rc.candidate.contract_value and rc.candidate.contract_value > 0
    ]

    if not valid_comparables:
        # Fallback parametric estimate if no empirical cost data was found
        return _generate_parametric_fallback(profile, target_year, selected_comparables)

    # 2. Normalize costs to target year USD unit cost
    normalized_unit_costs_usd: list[float] = []

    for rc in valid_comparables:
        cand = rc.candidate
        raw_cost = cand.contract_value
        curr = (cand.contract_currency or "USD").upper()
        rate = EXCHANGE_RATES_TO_USD.get(curr, 1.0)
        cost_usd = raw_cost * rate

        # Unit cost = Total Contract / Vessel Count
        vessel_count = max(1, cand.vessel_count_in_contract or 1)
        unit_cost_usd = cost_usd / vessel_count

        # Inflation adjustment to target_year
        contract_year = cand.contract_year or cand.delivery_year or 2020
        years_elapsed = max(0, target_year - contract_year)
        inflation_factor = math.pow(1 + ANNUAL_INFLATION_RATE, years_elapsed)

        adjusted_unit_cost_usd = unit_cost_usd * inflation_factor
        normalized_unit_costs_usd.append(adjusted_unit_cost_usd)

        logger.info(
            "normalized_comparable_cost",
            vessel_name=cand.vessel_name,
            raw_cost=raw_cost,
            currency=curr,
            vessel_count=vessel_count,
            contract_year=contract_year,
            adjusted_unit_cost_usd=round(adjusted_unit_cost_usd, 2),
        )

    # 3. Calculate Low, Central (P50 / Median), and High Estimates
    normalized_unit_costs_usd.sort()
    count = len(normalized_unit_costs_usd)

    low_usd = normalized_unit_costs_usd[0]
    high_usd = normalized_unit_costs_usd[-1]

    # Central estimate = Median or weighted average by similarity score
    if count == 1:
        central_usd = normalized_unit_costs_usd[0]
        low_usd = central_usd * 0.85
        high_usd = central_usd * 1.15
        warnings.append("Estimate based on single comparable vessel; wider error margin applies (+/- 15%).")
    elif count % 2 == 1:
        central_usd = normalized_unit_costs_usd[count // 2]
    else:
        mid1 = normalized_unit_costs_usd[(count // 2) - 1]
        mid2 = normalized_unit_costs_usd[count // 2]
        central_usd = (mid1 + mid2) / 2.0

    # Ensure range bounds
    if low_usd >= central_usd:
        low_usd = central_usd * 0.90
    if high_usd <= central_usd:
        high_usd = central_usd * 1.10

    # Determine confidence level
    confidence = "high" if count >= 3 and (high_usd - low_usd) / central_usd < 0.40 else "medium"
    if count == 1:
        confidence = "low"

    vessel_type_str = profile.specification.vessel_type.value if profile.specification.vessel_type else "Vessel"
    methodology = (
        f"Vessel-level cost estimate for target '{vessel_type_str}' derived from {count} "
        f"comparable vessel contracts delivered/built in recent years. Costs were normalized for "
        f"batch count, currency exchange rates, and a {ANNUAL_INFLATION_RATE*100:.1f}% annual "
        f"naval shipbuilding inflation index to {target_year} USD."
    )

    logger.info(
        "cost_estimation_complete",
        run_id=profile.run_id,
        low_usd=round(low_usd, 2),
        central_usd=round(central_usd, 2),
        high_usd=round(high_usd, 2),
        confidence=confidence,
    )

    return CostEstimate(
        low_usd=round(low_usd, 2),
        central_usd=round(central_usd, 2),
        high_usd=round(high_usd, 2),
        currency="USD",
        estimate_year=target_year,
        confidence=confidence,
        comparable_count=count,
        methodology=methodology,
        warnings=warnings,
    )


def _generate_parametric_fallback(
    profile: VesselProfile,
    target_year: int,
    selected_comparables: list[RankedComparable],
) -> CostEstimate:
    """
    Parametric estimation fallback when no empirical contract price could be retrieved.
    Uses light displacement / gross tonnage naval cost factors (~$45,000 - $85,000 per displacement tonne).
    """
    warnings: list[str] = [
        "No verified comparable vessel contract values were retrieved from search results.",
        "Cost estimate generated using naval architectural parametric cost-per-tonne density factors.",
    ]

    spec = profile.specification
    disp = None
    if spec.displacement_t and spec.displacement_t.value:
        try:
            disp = float(spec.displacement_t.value)
        except (ValueError, TypeError):
            pass

    length = None
    if spec.length_m and spec.length_m.value:
        try:
            length = float(spec.length_m.value)
        except (ValueError, TypeError):
            pass

    # Default displacement estimate if missing
    if not disp:
        if length:
            disp = math.pow(length / 4.5, 3.0)  # Cubic length scaling rule of thumb
        else:
            disp = 2000.0  # Default fallback OPV displacement
            warnings.append("Displacement not provided; assumed 2,000 tonnes baseline.")

    # Naval OPV / Patrol Vessel parametric cost per tonne: ~$50k to ~$90k / tonne
    cost_per_tonne_low = 50_000.0
    cost_per_tonne_central = 70_000.0
    cost_per_tonne_high = 95_000.0

    low_usd = disp * cost_per_tonne_low
    central_usd = disp * cost_per_tonne_central
    high_usd = disp * cost_per_tonne_high

    methodology = (
        f"Parametric vessel-level estimate derived using naval cost-per-tonne density factors "
        f"(${cost_per_tonne_central:,.0f}/tonne) applied to estimated displacement of {disp:,.0f} tonnes."
    )

    return CostEstimate(
        low_usd=round(low_usd, 2),
        central_usd=round(central_usd, 2),
        high_usd=round(high_usd, 2),
        currency="USD",
        estimate_year=target_year,
        confidence="insufficient",
        comparable_count=len(selected_comparables),
        methodology=methodology,
        warnings=warnings,
    )
