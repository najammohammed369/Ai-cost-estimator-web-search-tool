"""
Vessel Specification Extractor — converts unstructured document text to a
structured VesselSpecification using an LLM.

Also handles:
- Manual override merging (manual > extracted)
- Precedence logging
- Retry on parse failures
"""

import json
import structlog

from app.llm.client import LLMClient
from app.llm.retry import complete_and_parse, parse_json_response
from app.extraction.schemas import (
    VesselSpecification,
    VesselProfile,
    ManualOverride,
    PrecedenceEntry,
)
from app.extraction.prompts import (
    PROMPT_A_SYSTEM,
    PROMPT_A_USER_TEMPLATE,
    PROMPT_B_SYSTEM,
    PROMPT_B_USER_TEMPLATE,
    format_spec_for_prompt,
)
from app.ingestion.document_processor import DocumentContent
from app.config import get_settings

logger = structlog.get_logger(__name__)

# Maximum characters of document text to send in a single LLM call
# Llama 3.3 70B has a 128k token context window; ~400k chars is safe
MAX_DOCUMENT_CHARS = 60_000


def _truncate_document_text(full_text: str) -> str:
    """
    Truncate document text if it exceeds the safe LLM context window.
    Preserves the beginning and end of the document (most likely to contain specs).
    """
    if len(full_text) <= MAX_DOCUMENT_CHARS:
        return full_text

    half = MAX_DOCUMENT_CHARS // 2
    truncated = (
        full_text[:half]
        + "\n\n[... DOCUMENT TRUNCATED FOR LENGTH ...]\n\n"
        + full_text[-half:]
    )
    logger.warning(
        "document_truncated_for_llm",
        original_chars=len(full_text),
        truncated_chars=len(truncated),
    )
    return truncated


def extract_vessel_specification(
    document: DocumentContent,
    llm_client: LLMClient | None = None,
) -> VesselSpecification:
    """
    Use an LLM to extract a structured VesselSpecification from a document.

    Args:
        document: Processed document content.
        llm_client: Optional LLM client (creates one from config if not provided).

    Returns:
        Validated VesselSpecification Pydantic model.
    """
    if llm_client is None:
        llm_client = LLMClient()

    document_text = _truncate_document_text(document.full_text)

    messages = [
        {"role": "system", "content": PROMPT_A_SYSTEM},
        {
            "role": "user",
            "content": PROMPT_A_USER_TEMPLATE.format(
                document_text=document_text
            ),
        },
    ]

    logger.info(
        "spec_extraction_start",
        document_id=document.document_id,
        text_chars=len(document_text),
    )

    try:
        spec = complete_and_parse(
            client=llm_client,
            messages=messages,
            response_model=VesselSpecification,
            max_retries=3,
            temperature=0.0,  # Deterministic for extraction
            max_tokens=4096,
        )
    except Exception as e:
        logger.warning(
            "spec_extraction_llm_failed_using_heuristic",
            error=str(e),
            document_id=document.document_id,
        )
        spec = extract_vessel_specification_heuristic(document.full_text)

    # Log extraction summary
    extracted_fields = [
        field
        for field, value in spec.model_dump().items()
        if value is not None
    ]
    logger.info(
        "spec_extraction_complete",
        document_id=document.document_id,
        extracted_fields=len(extracted_fields),
        fields=extracted_fields,
    )

    return spec


def extract_vessel_specification_heuristic(document_text: str) -> VesselSpecification:
    """
    Fallback heuristic extractor using regex pattern matching when LLM is unavailable.
    """
    import re
    from app.extraction.schemas import ExtractedValue

    spec = VesselSpecification()

    # Vessel type
    if re.search(r"offshore patrol vessel|opv", document_text, re.I):
        spec.vessel_type = ExtractedValue(value="Offshore Patrol Vessel", confidence="medium")
    elif re.search(r"corvette", document_text, re.I):
        spec.vessel_type = ExtractedValue(value="Corvette", confidence="medium")
    elif re.search(r"frigate", document_text, re.I):
        spec.vessel_type = ExtractedValue(value="Frigate", confidence="medium")

    # Length
    m_len = re.search(r"(?:length overall|loa|length)[^\d]*(\d+(?:\.\d+)?)\s*(?:m|meter|metres|meters)", document_text, re.I)
    if m_len:
        spec.length_m = ExtractedValue(value=float(m_len.group(1)), unit="m", confidence="high")

    # Beam
    m_beam = re.search(r"beam[^\d]*(\d+(?:\.\d+)?)\s*(?:m|meter|metres|meters)", document_text, re.I)
    if m_beam:
        spec.beam_m = ExtractedValue(value=float(m_beam.group(1)), unit="m", confidence="high")

    # Draft
    m_draft = re.search(r"draft[^\d]*(\d+(?:\.\d+)?)\s*(?:m|meter|metres|meters)", document_text, re.I)
    if m_draft:
        spec.draft_m = ExtractedValue(value=float(m_draft.group(1)), unit="m", confidence="high")

    # Displacement
    m_disp = re.search(r"displacement[^\d]*~?([\d,]+(?:\.\d+)?)\s*(?:metric tonnes|tonnes|tons|t)", document_text, re.I)
    if m_disp:
        val = float(m_disp.group(1).replace(",", ""))
        spec.displacement_t = ExtractedValue(value=val, unit="tonnes", confidence="high")

    # Speed
    m_speed = re.search(r"(?:maximum speed|max speed|speed)[^\d]*(\d+(?:\.\d+)?)\s*knots", document_text, re.I)
    if m_speed:
        spec.speed_knots = ExtractedValue(value=float(m_speed.group(1)), unit="knots", confidence="high")

    # Range
    m_range = re.search(r"range[^\d]*([\d,]+)\s*(?:nautical miles|nm)", document_text, re.I)
    if m_range:
        val = float(m_range.group(1).replace(",", ""))
        spec.range_nm = ExtractedValue(value=val, unit="nm", confidence="high")

    # Endurance
    m_endurance = re.search(r"endurance[^\d]*(\d+)\s*days", document_text, re.I)
    if m_endurance:
        spec.endurance_days = ExtractedValue(value=int(m_endurance.group(1)), unit="days", confidence="high")

    # Crew
    m_crew = re.search(r"crew[^\d]*(\d+)", document_text, re.I)
    if m_crew:
        spec.crew = ExtractedValue(value=int(m_crew.group(1)), confidence="medium")

    return spec


def merge_with_overrides(
    run_id: str,
    document: DocumentContent,
    specification: VesselSpecification,
    manual_overrides: list[ManualOverride] | None = None,
) -> VesselProfile:
    """
    Build a VesselProfile by merging the extracted specification with
    any user-provided manual overrides.

    Precedence: manual override > LLM-extracted value.

    Args:
        run_id: Unique run identifier.
        document: Source document.
        specification: LLM-extracted specification.
        manual_overrides: Optional user-provided overrides.

    Returns:
        VesselProfile with merged values and precedence log.
    """
    if manual_overrides is None:
        manual_overrides = []

    # Build a map of field_name -> override
    override_map = {override.field_name: override for override in manual_overrides}

    # Build precedence log
    precedence_log: list[PrecedenceEntry] = []

    spec_dict = specification.model_dump()

    for field_name, extracted_value in spec_dict.items():
        if field_name in override_map:
            override = override_map[field_name]
            precedence_log.append(
                PrecedenceEntry(
                    field_name=field_name,
                    final_value=override.value,
                    source="manual",
                    extracted_value=extracted_value.get("value") if extracted_value else None,
                    manual_value=override.value,
                )
            )
        elif extracted_value is not None:
            value = extracted_value.get("value") if isinstance(extracted_value, dict) else None
            precedence_log.append(
                PrecedenceEntry(
                    field_name=field_name,
                    final_value=value,
                    source="extracted",
                    extracted_value=value,
                    manual_value=None,
                )
            )

    logger.info(
        "vessel_profile_built",
        run_id=run_id,
        document_id=document.document_id,
        manual_overrides_applied=len(override_map),
        total_fields_with_values=len(
            [e for e in precedence_log if e.final_value is not None]
        ),
    )

    return VesselProfile(
        document_id=document.document_id,
        run_id=run_id,
        file_name=document.file_name,
        specification=specification,
        manual_overrides=manual_overrides,
        precedence_log=precedence_log,
    )


def generate_search_queries(
    profile: VesselProfile,
    llm_client: LLMClient | None = None,
) -> list[dict]:
    """
    Use an LLM to generate search queries for finding comparable vessels.

    Args:
        profile: Vessel profile with extracted (and possibly overridden) specs.
        llm_client: Optional LLM client.

    Returns:
        List of query dicts: [{query, category, rationale, priority}]
    """
    if llm_client is None:
        llm_client = LLMClient()

    spec_summary = format_spec_for_prompt(profile.specification.model_dump())

    messages = [
        {"role": "system", "content": PROMPT_B_SYSTEM},
        {
            "role": "user",
            "content": PROMPT_B_USER_TEMPLATE.format(
                vessel_specification=spec_summary
            ),
        },
    ]

    logger.info("query_generation_start", run_id=profile.run_id)

    try:
        raw = llm_client.complete_json(messages=messages, temperature=0.2, max_tokens=2048)
        parsed = parse_json_response(raw)
        queries = parsed.get("queries", [])
        # Validate and normalise each query
        result = []
        for q in queries:
            if isinstance(q, dict) and "query" in q:
                result.append({
                    "query": str(q.get("query", "")),
                    "category": str(q.get("category", "general")),
                    "rationale": str(q.get("rationale", "")),
                    "priority": int(q.get("priority", 5)),
                })
        logger.info(
            "query_generation_complete",
            run_id=profile.run_id,
            query_count=len(result),
        )
        return result
    except Exception as e:
        logger.error("query_generation_failed", run_id=profile.run_id, error=str(e))
        # Fallback: generate basic queries from known spec fields
        return _fallback_queries(profile)


def _fallback_queries(profile: VesselProfile) -> list[dict]:
    """Generate basic queries when LLM query generation fails."""
    spec = profile.specification
    queries = []

    vessel_type = None
    if spec.vessel_type and spec.vessel_type.value:
        vessel_type = spec.vessel_type.value
    if not vessel_type:
        vessel_type = "Patrol Vessel"

    displacement = None
    if spec.displacement_t and spec.displacement_t.value:
        displacement = spec.displacement_t.value

    length = None
    if spec.length_m and spec.length_m.value:
        length = spec.length_m.value

    if vessel_type:
        queries.append({
            "query": f'"{vessel_type}" contract value delivered 2020 2021 2022 2023',
            "category": "vessel_type",
            "rationale": "Fallback: basic vessel type search",
            "priority": 7,
        })
        queries.append({
            "query": f'"{vessel_type}" shipbuilding cost procurement',
            "category": "procurement",
            "rationale": "Fallback: procurement cost search",
            "priority": 7,
        })

    if displacement and vessel_type:
        queries.append({
            "query": f'"{int(displacement)} tonne" OR "{int(displacement)} ton" {vessel_type} delivered',
            "category": "dimensions",
            "rationale": "Fallback: displacement search",
            "priority": 6,
        })

    if length and vessel_type:
        queries.append({
            "query": f'"{int(length)}m" OR "{int(length)} metre" {vessel_type} naval',
            "category": "dimensions",
            "rationale": "Fallback: length search",
            "priority": 6,
        })

    logger.warning(
        "using_fallback_queries",
        run_id=profile.run_id,
        count=len(queries),
    )

    return queries
