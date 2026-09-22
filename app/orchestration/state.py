"""
EstimationState — the central state object passed through every pipeline step.

Every stage reads from and writes to this object so the complete
history of a run is preserved and auditable.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, Any

from app.ingestion.document_processor import DocumentContent
from app.extraction.schemas import (
    VesselSpecification,
    VesselProfile,
    SearchResult,
    RetrievedPage,
    CandidateVessel,
    CostEstimate,
)


@dataclass
class StepResult:
    """Metadata about a completed pipeline step."""
    step_name: str
    status: str           # "success", "warning", "failed"
    message: str = ""
    duration_s: float = 0.0


@dataclass
class EstimationState:
    """
    Central state object for a single vessel cost estimation run.

    Every pipeline step receives this state, does its work,
    and updates the relevant fields before returning it.
    This ensures every output is traceable back to its inputs.
    """

    # -------------------------------------------------------------------------
    # Run metadata
    # -------------------------------------------------------------------------
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    document_id: str = ""

    # -------------------------------------------------------------------------
    # Phase 1: Document ingestion
    # -------------------------------------------------------------------------
    document: Optional[DocumentContent] = None

    # -------------------------------------------------------------------------
    # Phase 1: Specification extraction
    # -------------------------------------------------------------------------
    vessel_specification: Optional[VesselSpecification] = None
    vessel_profile: Optional[VesselProfile] = None

    # -------------------------------------------------------------------------
    # Phase 2: Query generation
    # -------------------------------------------------------------------------
    search_queries: list[dict] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Phase 3: Web search
    # -------------------------------------------------------------------------
    search_results: list[SearchResult] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Phase 3: Webpage retrieval
    # -------------------------------------------------------------------------
    retrieved_pages: list[RetrievedPage] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Phase 4: Candidate vessel discovery (stubbed for Phases 4+)
    # -------------------------------------------------------------------------
    candidate_vessels: list[CandidateVessel] = field(default_factory=list)
    selected_comparables: list[CandidateVessel] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Phase 5: Cost extraction and normalization (stubbed)
    # -------------------------------------------------------------------------
    cost_evidence: list[dict] = field(default_factory=list)
    normalized_costs: list[dict] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Phase 5: Country/ERV adjustment (stubbed)
    # -------------------------------------------------------------------------
    adjustments: list[dict] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Phase 6: Estimation (stubbed)
    # -------------------------------------------------------------------------
    estimate: Optional[CostEstimate] = None

    # -------------------------------------------------------------------------
    # Sources and citations
    # -------------------------------------------------------------------------
    sources: list[dict] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    steps_completed: list[StepResult] = field(default_factory=list)
    current_step: str = "initialized"

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def mark_step(self, step_name: str, status: str = "success",
                  message: str = "", duration_s: float = 0.0) -> None:
        self.steps_completed.append(StepResult(
            step_name=step_name,
            status=status,
            message=message,
            duration_s=duration_s,
        ))
        self.current_step = step_name

    def to_summary_dict(self) -> dict:
        """Return a concise summary dict for API responses and logging."""
        return {
            "run_id": self.run_id,
            "document_id": self.document_id,
            "current_step": self.current_step,
            "steps_completed": [
                {"step": s.step_name, "status": s.status, "duration_s": s.duration_s}
                for s in self.steps_completed
            ],
            "search_queries": len(self.search_queries),
            "search_results": len(self.search_results),
            "retrieved_pages": len(self.retrieved_pages),
            "candidate_vessels": len(self.candidate_vessels),
            "selected_comparables": len(self.selected_comparables),
            "has_estimate": self.estimate is not None,
            "warnings": self.warnings,
            "errors": self.errors,
        }
