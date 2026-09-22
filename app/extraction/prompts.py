"""
LLM Prompt Templates for the Vessel Cost Estimator.

Each prompt is a separate, maintainable template with:
- Anti-hallucination instructions
- Structured JSON output schema
- Clear task boundaries

Prompts:
  A — Document → Vessel Specification
  B — Vessel Specification → Search Queries
  C — Search Result → Candidate Vessel (Phase 3+)
  D — Candidate Vessel → Cost Evidence (Phase 4+)
  E — Target + Candidates → Similarity Analysis (Phase 4+)
  F — Comparable Costs → Final Estimate (Phase 6+)
"""

# =============================================================================
# PROMPT A — Document → Vessel Specification
# =============================================================================

PROMPT_A_SYSTEM = """You are a naval architecture and shipbuilding expert tasked with extracting \
vessel technical specifications from procurement and design documents.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. Extract ONLY information that is explicitly stated in the document.
2. Do NOT infer, guess, or hallucinate any value.
3. If a field is not mentioned in the document, return null for that field.
4. For every extracted value, you MUST cite the page number and the exact source sentence.
5. Return ONLY valid JSON — no preamble, no explanation, no markdown code blocks.
6. All numeric values must be numbers (not strings). Example: 125 not "125 m".
7. Units must be separate from values.

OUTPUT FORMAT:
Return a single JSON object matching this schema exactly. All fields are optional (null if not found):

{
  "vessel_type": {"value": "string", "unit": null, "source_page": 1, "source_text": "exact quote", "confidence": "high|medium|low"} | null,
  "vessel_class": {"value": "string", "unit": null, "source_page": 1, "source_text": "...", "confidence": "..."} | null,
  "mission": {"value": "string", "unit": null, "source_page": 1, "source_text": "...", "confidence": "..."} | null,
  "length_m": {"value": 125.0, "unit": "m", "source_page": 2, "source_text": "...", "confidence": "..."} | null,
  "beam_m": {"value": 16.5, "unit": "m", "source_page": 2, "source_text": "...", "confidence": "..."} | null,
  "draft_m": {"value": 4.5, "unit": "m", "source_page": 2, "source_text": "...", "confidence": "..."} | null,
  "displacement_t": {"value": 5000, "unit": "tonnes", "source_page": 2, "source_text": "...", "confidence": "..."} | null,
  "gross_tonnage": {"value": null, "unit": "GT", "source_page": null, "source_text": null, "confidence": null} | null,
  "deadweight_t": {"value": null, "unit": "DWT", "source_page": null, "source_text": null, "confidence": null} | null,
  "speed_knots": {"value": 22, "unit": "knots", "source_page": 3, "source_text": "...", "confidence": "..."} | null,
  "range_nm": {"value": 5000, "unit": "nm", "source_page": 3, "source_text": "...", "confidence": "..."} | null,
  "endurance_days": {"value": 30, "unit": "days", "source_page": 3, "source_text": "...", "confidence": "..."} | null,
  "crew": {"value": 120, "unit": "persons", "source_page": 4, "source_text": "...", "confidence": "..."} | null,
  "passenger_capacity": {"value": null, "unit": null, "source_page": null, "source_text": null, "confidence": null} | null,
  "propulsion_type": {"value": "string", "unit": null, "source_page": 4, "source_text": "...", "confidence": "..."} | null,
  "propulsion_power_mw": {"value": 20.0, "unit": "MW", "source_page": 4, "source_text": "...", "confidence": "..."} | null,
  "fuel_type": {"value": "string", "unit": null, "source_page": 4, "source_text": "...", "confidence": "..."} | null,
  "aviation_facility": {"value": "string", "unit": null, "source_page": 5, "source_text": "...", "confidence": "..."} | null,
  "helicopter_capability": {"value": "string", "unit": null, "source_page": 5, "source_text": "...", "confidence": "..."} | null,
  "boat_capability": {"value": "string", "unit": null, "source_page": 5, "source_text": "...", "confidence": "..."} | null,
  "radar": {"value": "string", "unit": null, "source_page": 5, "source_text": "...", "confidence": "..."} | null,
  "sensors": {"value": "string", "unit": null, "source_page": 5, "source_text": "...", "confidence": "..."} | null,
  "armament": [{"value": "string", "unit": null, "source_page": 5, "source_text": "...", "confidence": "..."}] | null,
  "mission_equipment": [{"value": "string", "unit": null, "source_page": 5, "source_text": "...", "confidence": "..."}] | null,
  "classification": {"value": "string", "unit": null, "source_page": 1, "source_text": "...", "confidence": "..."} | null,
  "construction_material": {"value": "string", "unit": null, "source_page": 1, "source_text": "...", "confidence": "..."} | null,
  "customer_country": {"value": "string", "unit": null, "source_page": 1, "source_text": "...", "confidence": "..."} | null,
  "required_delivery_year": {"value": 2028, "unit": "year", "source_page": 1, "source_text": "...", "confidence": "..."} | null,
  "quantity": {"value": 2, "unit": "vessels", "source_page": 1, "source_text": "...", "confidence": "..."} | null
}"""

PROMPT_A_USER_TEMPLATE = """Extract the vessel technical specifications from the following document.

DOCUMENT:
{document_text}

Remember:
- Return null for any field not found in the document.
- Every non-null field MUST have source_page and source_text.
- Do not hallucinate. Do not infer. Only extract what is explicitly stated.
- Return ONLY the JSON object, nothing else."""


# =============================================================================
# PROMPT B — Vessel Specification → Search Queries
# =============================================================================

PROMPT_B_SYSTEM = """You are an expert naval procurement analyst. Your task is to generate \
web search queries to find comparable vessels that have been publicly procured or delivered \
in approximately the last 5–10 years.

The goal is to find:
1. Comparable vessels with publicly available cost information (contract values, procurement prices)
2. Sources such as government announcements, navy press releases, shipbuilder statements, and maritime news

STRICT RULES:
1. Generate 5–15 diverse queries across different categories.
2. Each query must target vessel-LEVEL cost information — do NOT generate queries for individual components.
3. Focus on discriminative parameters (vessel type, displacement, length, mission).
4. Include queries targeting government/official sources.
5. Use appropriate search operators ("quotes", site:gov, site:mil).
6. Return ONLY valid JSON — no preamble, no explanation, no code blocks.

QUERY CATEGORIES (generate queries in each relevant category):
- vessel_type: Type + key specs + delivered + cost
- dimensions: Key dimensions + vessel type + delivered
- mission: Mission profile + capability + contract value
- procurement: Procurement/contract/shipbuilding + value
- official: Site-specific searches (gov, mil, navy, mod)

OUTPUT FORMAT:
{
  "vessel_type": "string (identified vessel type)",
  "queries": [
    {
      "query": "exact search query string",
      "category": "vessel_type|dimensions|mission|procurement|official",
      "rationale": "why this query is useful",
      "priority": 8
    }
  ],
  "total_queries": 10
}"""

PROMPT_B_USER_TEMPLATE = """Generate web search queries to find comparable vessels for cost estimation.

TARGET VESSEL SPECIFICATION:
{vessel_specification}

Generate queries that will find comparable real-world vessels delivered in the last 5–10 years \
with publicly available cost information. Focus on the most distinctive characteristics \
(vessel type, displacement, mission, key capabilities).

Return ONLY the JSON object."""


# =============================================================================
# PROMPT C — Search Snippet → Candidate Vessel Extraction
# =============================================================================

PROMPT_C_SYSTEM = """You are a naval intelligence analyst extracting structured vessel information \
from web search results and webpage content.

STRICT RULES:
1. Extract ONLY information explicitly stated in the provided text.
2. Do NOT hallucinate vessel names, costs, dates, or specifications.
3. If the text mentions a vessel but lacks key details, still record what is available.
4. Distinguish between VESSEL cost and PROGRAM/CONTRACT cost — they are NOT the same.
5. Return ONLY valid JSON.

A "candidate vessel" is any named vessel mentioned in the text that:
- Has a known vessel type
- Has at least some specifications or cost information
- Was or is being built/procured by a navy, coast guard, or government

OUTPUT FORMAT:
{
  "candidates": [
    {
      "vessel_name": "string",
      "vessel_type": "string or null",
      "shipbuilder": "string or null",
      "operator": "string or null",
      "country": "string or null",
      "delivery_year": integer or null,
      "length_m": float or null,
      "beam_m": float or null,
      "draft_m": float or null,
      "displacement_t": float or null,
      "speed_knots": float or null,
      "mission": "string or null",
      "contract_value": float or null,
      "contract_currency": "USD|EUR|GBP|INR|etc or null",
      "contract_year": integer or null,
      "vessel_count_in_contract": integer or null,
      "program_includes_support": true|false|null,
      "source_urls": ["url1"],
      "source_tier": 1|2|3|4,
      "data_category": "A|B|C",
      "raw_snippets": ["relevant text excerpts"]
    }
  ]
}

data_category:
  A = specifications AND cost available
  B = specifications available, cost unavailable
  C = weak match — minimal information"""

PROMPT_C_USER_TEMPLATE = """Extract candidate vessel information from the following search results.

SOURCE URL: {url}
SOURCE TIER: {source_tier}

CONTENT:
{content}

Extract any named vessels with naval/government procurement context. Return ONLY the JSON."""


# =============================================================================
# PROMPT D — Candidate → Cost Evidence (Phase 4+, stubbed)
# =============================================================================

PROMPT_D_SYSTEM = """You are a naval cost analyst. Extract precise cost evidence for a specific vessel \
from the provided text.

STRICT RULES:
1. Distinguish between: single vessel cost, multi-vessel contract total, program cost.
2. NEVER divide a program cost by vessel count without explicitly noting what the program includes.
3. Record currency, year, and whether it is a vessel-only cost.
4. Return null for any cost information not found.
5. Return ONLY valid JSON."""

PROMPT_D_USER_TEMPLATE = """Extract cost evidence for vessel: {vessel_name}

CONTENT:
{content}

Return structured cost evidence as JSON."""


# =============================================================================
# Helper: format vessel specification for prompts
# =============================================================================

def format_spec_for_prompt(specification: dict) -> str:
    """
    Format a VesselSpecification dict into a concise human-readable summary
    for use in LLM prompts.
    """
    lines: list[str] = []

    field_labels = {
        "vessel_type": "Vessel Type",
        "vessel_class": "Vessel Class",
        "mission": "Mission",
        "length_m": "Length",
        "beam_m": "Beam",
        "draft_m": "Draft",
        "displacement_t": "Displacement",
        "gross_tonnage": "Gross Tonnage",
        "speed_knots": "Max Speed",
        "range_nm": "Range",
        "endurance_days": "Endurance",
        "crew": "Crew",
        "passenger_capacity": "Passenger Capacity",
        "propulsion_type": "Propulsion",
        "propulsion_power_mw": "Propulsion Power",
        "fuel_type": "Fuel Type",
        "aviation_facility": "Aviation Facility",
        "helicopter_capability": "Helicopter Capability",
        "boat_capability": "Boat Capability",
        "radar": "Radar",
        "sensors": "Sensors",
        "armament": "Armament",
        "classification": "Classification",
        "construction_material": "Construction Material",
        "customer_country": "Customer Country",
        "required_delivery_year": "Required Delivery Year",
        "quantity": "Quantity",
    }

    for field_key, label in field_labels.items():
        field_data = specification.get(field_key)
        if field_data is None:
            continue

        if isinstance(field_data, list):
            # List field (armament, mission_equipment)
            values = [
                str(item.get("value", "")) if isinstance(item, dict) else str(item)
                for item in field_data
                if item
            ]
            if values:
                lines.append(f"{label}: {', '.join(values)}")
        elif isinstance(field_data, dict):
            value = field_data.get("value")
            unit = field_data.get("unit", "")
            if value is not None:
                unit_str = f" {unit}" if unit else ""
                lines.append(f"{label}: {value}{unit_str}")

    return "\n".join(lines) if lines else "No specifications extracted."
