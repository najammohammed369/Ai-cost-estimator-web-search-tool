"""
Pydantic Schemas for the Vessel Cost Estimator.

Every extracted value carries its source page, source text, and confidence
so the final estimate is fully auditable.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Atomic extracted value — carries provenance
# =============================================================================

class ExtractedValue(BaseModel):
    """
    A single extracted value with full provenance.

    Use this for every specification field so the estimate is auditable.
    """
    value: Any
    unit: Optional[str] = None
    source_page: Optional[int] = None
    source_text: Optional[str] = Field(
        default=None,
        description="The exact sentence or clause from the document that yielded this value.",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Extraction confidence: 'high', 'medium', or 'low'.",
    )


# =============================================================================
# Vessel Specification — structured output from the LLM extraction prompt
# =============================================================================

class VesselSpecification(BaseModel):
    """
    Complete structured vessel specification extracted from a document.

    Fields are all optional because many specifications will not cover every
    parameter. Do NOT hallucinate missing values — return null.
    """

    # --- Identity ---
    vessel_type: Optional[ExtractedValue] = Field(
        default=None, description="e.g. 'Offshore Patrol Vessel', 'Corvette', 'LPD'"
    )
    vessel_class: Optional[ExtractedValue] = Field(
        default=None, description="Named class if mentioned"
    )
    mission: Optional[ExtractedValue] = Field(
        default=None, description="Primary mission / role"
    )

    # --- Dimensions ---
    length_m: Optional[ExtractedValue] = Field(
        default=None, description="Length overall in metres"
    )
    beam_m: Optional[ExtractedValue] = Field(
        default=None, description="Beam in metres"
    )
    draft_m: Optional[ExtractedValue] = Field(
        default=None, description="Draft in metres"
    )

    # --- Tonnage ---
    displacement_t: Optional[ExtractedValue] = Field(
        default=None, description="Displacement in tonnes"
    )
    gross_tonnage: Optional[ExtractedValue] = Field(
        default=None, description="Gross tonnage (GT)"
    )
    deadweight_t: Optional[ExtractedValue] = Field(
        default=None, description="Deadweight tonnage (DWT)"
    )

    # --- Performance ---
    speed_knots: Optional[ExtractedValue] = Field(
        default=None, description="Maximum speed in knots"
    )
    range_nm: Optional[ExtractedValue] = Field(
        default=None, description="Range in nautical miles"
    )
    endurance_days: Optional[ExtractedValue] = Field(
        default=None, description="Endurance in days"
    )

    # --- Crew & Capacity ---
    crew: Optional[ExtractedValue] = Field(
        default=None, description="Total crew complement"
    )
    passenger_capacity: Optional[ExtractedValue] = Field(
        default=None, description="Passenger or troop capacity"
    )

    # --- Propulsion ---
    propulsion_type: Optional[ExtractedValue] = Field(
        default=None, description="e.g. 'CODAD', 'CODAG', 'diesel-electric', 'gas turbine'"
    )
    propulsion_power_mw: Optional[ExtractedValue] = Field(
        default=None, description="Total propulsion power in MW"
    )
    fuel_type: Optional[ExtractedValue] = Field(
        default=None, description="e.g. 'marine diesel', 'LNG', 'heavy fuel oil'"
    )

    # --- Aviation & Boats ---
    aviation_facility: Optional[ExtractedValue] = Field(
        default=None, description="Flight deck, hangar details"
    )
    helicopter_capability: Optional[ExtractedValue] = Field(
        default=None, description="Helicopter type/size accommodation"
    )
    boat_capability: Optional[ExtractedValue] = Field(
        default=None, description="RHIBs, workboats, etc."
    )

    # --- Sensors & Electronics ---
    radar: Optional[ExtractedValue] = Field(default=None, description="Radar systems")
    sensors: Optional[ExtractedValue] = Field(
        default=None, description="Sonar, EO/IR, other sensors"
    )

    # --- Armament ---
    armament: Optional[list[ExtractedValue]] = Field(
        default=None, description="Weapons systems"
    )

    # --- Mission Equipment ---
    mission_equipment: Optional[list[ExtractedValue]] = Field(
        default=None, description="Mission-specific equipment"
    )

    # --- Construction ---
    classification: Optional[ExtractedValue] = Field(
        default=None, description="Classification society and notation"
    )
    construction_material: Optional[ExtractedValue] = Field(
        default=None, description="e.g. 'steel', 'aluminium', 'GRP'"
    )

    # --- Programme ---
    customer_country: Optional[ExtractedValue] = Field(
        default=None, description="Country / navy ordering the vessel"
    )
    required_delivery_year: Optional[ExtractedValue] = Field(
        default=None, description="Target delivery year"
    )
    quantity: Optional[ExtractedValue] = Field(
        default=None, description="Number of vessels in the programme"
    )


# =============================================================================
# Manual override — user-provided values that take precedence over extraction
# =============================================================================

class ManualOverride(BaseModel):
    """A user-supplied value that overrides the LLM-extracted value."""
    field_name: str
    value: Any
    unit: Optional[str] = None


# =============================================================================
# Vessel Profile — merged specification with precedence tracking
# =============================================================================

class PrecedenceEntry(BaseModel):
    """Tracks where a final value came from."""
    field_name: str
    final_value: Any
    source: str  # "manual", "extracted", "default"
    extracted_value: Any = None
    manual_value: Any = None


class VesselProfile(BaseModel):
    """
    Final vessel profile combining extracted specification and manual overrides.

    The precedence rule is: manual override > LLM-extracted value.
    """
    document_id: str
    run_id: str
    file_name: str
    specification: VesselSpecification
    manual_overrides: list[ManualOverride] = Field(default_factory=list)
    precedence_log: list[PrecedenceEntry] = Field(default_factory=list)


# =============================================================================
# Search Query
# =============================================================================

class SearchQuery(BaseModel):
    """A generated web search query with category and rationale."""
    query: str
    category: str  # "vessel_type", "dimensions", "mission", "procurement", "vessel_specific", "official"
    rationale: str
    priority: int = Field(default=5, ge=1, le=10)  # 10 = highest priority


class SearchQuerySet(BaseModel):
    """A collection of generated search queries."""
    vessel_type: str
    queries: list[SearchQuery]
    total_queries: int


# =============================================================================
# Search Result
# =============================================================================

class SearchResult(BaseModel):
    """A single result from a web search."""
    query: str
    title: str
    url: str
    snippet: str
    source: str = ""
    published_date: Optional[str] = None
    position: int = 0
    source_tier: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Source quality tier (1=best: gov/official, 4=worst: blogs/forums)",
    )


# =============================================================================
# Retrieved Page
# =============================================================================

class RetrievedPage(BaseModel):
    """A fetched and cleaned webpage."""
    url: str
    title: str
    content: str           # Cleaned main text
    status_code: int
    is_paywalled: bool = False
    is_accessible: bool = True
    content_length: int = 0
    source_tier: int = 4
    error: Optional[str] = None


# =============================================================================
# Candidate Vessel
# =============================================================================

class CandidateVessel(BaseModel):
    """A potential comparable vessel discovered from search results."""
    vessel_name: str
    vessel_type: Optional[str] = None
    shipbuilder: Optional[str] = None
    operator: Optional[str] = None
    country: Optional[str] = None
    delivery_year: Optional[int] = None
    length_m: Optional[float] = None
    beam_m: Optional[float] = None
    draft_m: Optional[float] = None
    displacement_t: Optional[float] = None
    speed_knots: Optional[float] = None
    mission: Optional[str] = None
    contract_value: Optional[float] = None
    contract_currency: Optional[str] = None
    contract_year: Optional[int] = None
    vessel_count_in_contract: Optional[int] = None
    program_includes_support: Optional[bool] = None
    source_urls: list[str] = Field(default_factory=list)
    source_tier: int = 4
    data_category: str = "C"  # A=spec+cost, B=spec only, C=weak match
    raw_snippets: list[str] = Field(default_factory=list)


# =============================================================================
# Estimation Output
# =============================================================================

class CostEstimate(BaseModel):
    """Final vessel cost estimate with range."""
    low_usd: float
    central_usd: float
    high_usd: float
    currency: str = "USD"
    estimate_year: int
    confidence: str  # "high", "medium", "low", "insufficient"
    comparable_count: int
    methodology: str
    warnings: list[str] = Field(default_factory=list)
