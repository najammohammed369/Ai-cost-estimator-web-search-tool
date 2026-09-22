"""
Pipeline — fixed sequential orchestration for the vessel cost estimation workflow.

Phases 1–3 are fully implemented. Phases 4–8 are stubbed with clear extension points.

Steps:
  1. ingest_document         — PDF/DOCX/TXT → DocumentContent
  2. extract_specification   — DocumentContent → VesselSpecification (via LLM)
  3. generate_queries        — VesselProfile → list[SearchQuery] (via LLM)
  4. execute_search          — queries → list[SearchResult] (via SerpAPI)
  5. retrieve_pages          — SearchResult URLs → list[RetrievedPage]

  [Phases 4–8 — stubbed, to be implemented in subsequent phases]
  6. discover_candidates     — RetrievedPage → list[CandidateVessel]
  7. match_vessels           — CandidateVessel → ranked comparables
  8. extract_cost_evidence   — comparables → cost evidence
  9. normalize_costs         — cost evidence → normalized costs
  10. apply_adjustments      — normalized costs → adjusted costs
  11. estimate_cost          — adjusted costs → CostEstimate
  12. generate_report        — CostEstimate → HTML/PDF report
"""

import time
import structlog

from app.ingestion.document_processor import process_document, DocumentContent
from app.extraction.vessel_spec_extractor import (
    extract_vessel_specification,
    merge_with_overrides,
    generate_search_queries,
)
from app.extraction.schemas import ManualOverride
from app.search.result_collector import ResultCollector
from app.retrieval.webpage_retriever import retrieve_pages_batch
from app.database.cache import CacheManager
from app.llm.client import LLMClient
from app.orchestration.state import EstimationState
from app.estimation import (
    discover_candidate_vessels,
    rank_and_select_comparables,
    estimate_vessel_cost,
)

logger = structlog.get_logger(__name__)


class VesselCostPipeline:
    """
    Fixed sequential pipeline for vessel cost estimation.

    Each step takes EstimationState as input/output, making the entire
    history of a run auditable and resumable.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        cache: CacheManager | None = None,
        max_pages_to_retrieve: int = 25,
    ):
        self.llm = llm_client or LLMClient()
        self.cache = cache or CacheManager()
        self.max_pages = max_pages_to_retrieve
        self.collector = ResultCollector(cache=self.cache)

    # =========================================================================
    # Step 1 — Document Ingestion
    # =========================================================================

    def ingest_document(self, state: EstimationState, file_path: str) -> EstimationState:
        """Extract text from uploaded PDF/DOCX/TXT document."""
        logger.info("pipeline_step_start", step="ingest_document", run_id=state.run_id)
        t0 = time.time()

        try:
            document = process_document(file_path)
            state.document = document
            state.document_id = document.document_id

            for warning in document.warnings:
                state.add_warning(f"[ingestion] {warning}")

            if document.is_scanned:
                state.add_warning(
                    "Document appears to be scanned. Text extraction quality may be low. "
                    "Consider using an OCR-preprocessed version."
                )

            state.mark_step(
                "ingest_document", "success",
                f"Extracted {document.total_pages} pages, {len(document.full_text):,} chars",
                duration_s=round(time.time() - t0, 2),
            )
            logger.info(
                "pipeline_step_complete",
                step="ingest_document",
                pages=document.total_pages,
                chars=len(document.full_text),
            )

        except Exception as e:
            state.add_error(f"Document ingestion failed: {e}")
            state.mark_step("ingest_document", "failed", str(e))
            logger.error("pipeline_step_failed", step="ingest_document", error=str(e))
            raise

        return state

    # =========================================================================
    # Step 2 — Specification Extraction
    # =========================================================================

    def extract_specification(
        self,
        state: EstimationState,
        manual_overrides: list[ManualOverride] | None = None,
    ) -> EstimationState:
        """Use LLM to extract structured vessel spec from document text."""
        if state.document is None:
            raise ValueError("Cannot extract specification: document not ingested yet.")

        logger.info("pipeline_step_start", step="extract_specification", run_id=state.run_id)
        t0 = time.time()

        try:
            spec = extract_vessel_specification(state.document, llm_client=self.llm)
            state.vessel_specification = spec

            profile = merge_with_overrides(
                run_id=state.run_id,
                document=state.document,
                specification=spec,
                manual_overrides=manual_overrides,
            )
            state.vessel_profile = profile

            # Count non-null fields
            extracted_count = sum(
                1 for v in spec.model_dump().values() if v is not None
            )

            if extracted_count == 0:
                state.add_warning(
                    "No vessel specifications were extracted from the document. "
                    "Search queries will be generic and results may be poor."
                )

            state.mark_step(
                "extract_specification", "success",
                f"Extracted {extracted_count} specification fields",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Specification extraction failed: {e}")
            state.mark_step("extract_specification", "failed", str(e))
            logger.error("pipeline_step_failed", step="extract_specification", error=str(e))
            raise

        return state

    # =========================================================================
    # Step 3 — Query Generation
    # =========================================================================

    def generate_queries(self, state: EstimationState) -> EstimationState:
        """Use LLM to generate search queries from vessel profile."""
        if state.vessel_profile is None:
            raise ValueError("Cannot generate queries: vessel profile not built yet.")

        logger.info("pipeline_step_start", step="generate_queries", run_id=state.run_id)
        t0 = time.time()

        try:
            queries = generate_search_queries(state.vessel_profile, llm_client=self.llm)
            state.search_queries = queries

            if len(queries) < 3:
                state.add_warning(
                    f"Only {len(queries)} search queries were generated. "
                    "Results coverage may be limited."
                )

            state.mark_step(
                "generate_queries", "success",
                f"Generated {len(queries)} search queries",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Query generation failed: {e}")
            state.mark_step("generate_queries", "failed", str(e))
            logger.error("pipeline_step_failed", step="generate_queries", error=str(e))
            raise

        return state

    # =========================================================================
    # Step 4 — Web Search
    # =========================================================================

    def execute_search(self, state: EstimationState) -> EstimationState:
        """Execute all search queries and collect results."""
        if not state.search_queries:
            raise ValueError("Cannot execute search: no queries generated yet.")

        logger.info("pipeline_step_start", step="execute_search", run_id=state.run_id)
        t0 = time.time()

        try:
            results = self.collector.collect(
                queries=state.search_queries,
                run_id=state.run_id,
            )
            state.search_results = results

            tier1 = sum(1 for r in results if r.source_tier == 1)
            tier2 = sum(1 for r in results if r.source_tier == 2)

            if len(results) == 0:
                state.add_warning("No search results returned. Check SerpAPI key and quota.")

            state.mark_step(
                "execute_search", "success",
                f"Collected {len(results)} unique results "
                f"(Tier1={tier1}, Tier2={tier2})",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Web search failed: {e}")
            state.mark_step("execute_search", "failed", str(e))
            logger.error("pipeline_step_failed", step="execute_search", error=str(e))
            raise

        return state

    # =========================================================================
    # Step 5 — Webpage Retrieval
    # =========================================================================

    def retrieve_pages(self, state: EstimationState) -> EstimationState:
        """Fetch and clean webpage content from search result URLs."""
        if not state.search_results:
            state.add_warning("No search results to retrieve pages from.")
            state.mark_step("retrieve_pages", "warning", "No search results available")
            return state

        logger.info("pipeline_step_start", step="retrieve_pages", run_id=state.run_id)
        t0 = time.time()

        try:
            pages = retrieve_pages_batch(
                results=state.search_results,
                cache=self.cache,
                max_pages=self.max_pages,
            )
            state.retrieved_pages = pages

            accessible = sum(1 for p in pages if p.is_accessible)
            paywalled = sum(1 for p in pages if p.is_paywalled)

            if accessible == 0:
                state.add_warning(
                    "No accessible page content was retrieved. "
                    "Candidate discovery will rely on search snippets only."
                )
            if paywalled > 0:
                state.add_warning(
                    f"{paywalled} pages appeared to be paywalled and could not be fully read."
                )

            state.mark_step(
                "retrieve_pages", "success",
                f"Retrieved {len(pages)} pages ({accessible} accessible, {paywalled} paywalled)",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Page retrieval failed: {e}")
            state.mark_step("retrieve_pages", "failed", str(e))
            logger.error("pipeline_step_failed", step="retrieve_pages", error=str(e))
            raise

        return state

    # =========================================================================
    # Phases 4–8 — Stubs (to be implemented)
    # =========================================================================

    # =========================================================================
    # Phases 4–6 — Candidate Discovery, Matching & Cost Estimation
    # =========================================================================

    def discover_candidates(self, state: EstimationState) -> EstimationState:
        """[Phase 4] Extract candidate vessels from retrieved pages and search results."""
        if not state.vessel_profile:
            state.add_warning("Cannot discover candidates: vessel profile not built.")
            state.mark_step("discover_candidates", "warning", "Vessel profile missing")
            return state

        logger.info("pipeline_step_start", step="discover_candidates", run_id=state.run_id)
        t0 = time.time()

        try:
            candidates = discover_candidate_vessels(
                retrieved_pages=state.retrieved_pages,
                search_results=state.search_results,
                profile=state.vessel_profile,
                llm_client=self.llm,
            )
            state.candidate_vessels = candidates

            state.mark_step(
                "discover_candidates", "success",
                f"Discovered {len(candidates)} candidate vessels",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Candidate discovery failed: {e}")
            state.mark_step("discover_candidates", "failed", str(e))
            logger.error("pipeline_step_failed", step="discover_candidates", error=str(e))
            raise

        return state

    def match_vessels(self, state: EstimationState) -> EstimationState:
        """[Phase 4] Score and rank candidates against target vessel specification."""
        if not state.vessel_profile or not state.candidate_vessels:
            state.add_warning("No candidate vessels available for matching.")
            state.mark_step("match_vessels", "warning", "No candidates available")
            return state

        logger.info("pipeline_step_start", step="match_vessels", run_id=state.run_id)
        t0 = time.time()

        try:
            ranked_comparables = rank_and_select_comparables(
                profile=state.vessel_profile,
                candidates=state.candidate_vessels,
                max_selected=5,
            )
            # Store selected comparable candidate objects
            state.selected_comparables = [rc.candidate for rc in ranked_comparables]
            # Store full ranked list in state attribute for reporting
            state._ranked_comparables = ranked_comparables

            state.mark_step(
                "match_vessels", "success",
                f"Selected top {len(state.selected_comparables)} comparable vessels",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Vessel matching failed: {e}")
            state.mark_step("match_vessels", "failed", str(e))
            logger.error("pipeline_step_failed", step="match_vessels", error=str(e))
            raise

        return state

    def estimate_cost(self, state: EstimationState) -> EstimationState:
        """[Phase 6] Generate final vessel-level cost estimate range."""
        if not state.vessel_profile:
            state.add_warning("Cannot estimate cost: vessel profile missing.")
            state.mark_step("estimate_cost", "warning", "Vessel profile missing")
            return state

        logger.info("pipeline_step_start", step="estimate_cost", run_id=state.run_id)
        t0 = time.time()

        try:
            ranked_comparables = getattr(state, "_ranked_comparables", [])
            cost_est = estimate_vessel_cost(
                profile=state.vessel_profile,
                selected_comparables=ranked_comparables,
                target_year=2026,
            )
            state.estimate = cost_est

            for w in cost_est.warnings:
                state.add_warning(f"[estimation] {w}")

            state.mark_step(
                "estimate_cost", "success",
                f"Central estimate: ${cost_est.central_usd:,.0f} USD (Confidence: {cost_est.confidence})",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Cost estimation failed: {e}")
            state.mark_step("estimate_cost", "failed", str(e))
            logger.error("pipeline_step_failed", step="estimate_cost", error=str(e))
            raise

        return state

    def generate_report(self, state: EstimationState) -> EstimationState:
        """[Phase 7] Generate HTML/PDF report."""
        state.add_warning("[Phase 7] Report generation not yet implemented.")
        state.mark_step("generate_report", "warning", "Not yet implemented")
        return state

    # =========================================================================
    # Full Pipeline Run (Phases 1–3 + stubs)
    # =========================================================================

    def run(
        self,
        file_path: str,
        manual_overrides: list[ManualOverride] | None = None,
        run_through_phase: int = 3,
    ) -> EstimationState:
        """
        Execute the full pipeline up to the specified phase.

        Args:
            file_path: Path to the vessel specification document.
            manual_overrides: Optional user-provided parameter overrides.
            run_through_phase: Stop after this phase (1-8). Default: 3.

        Returns:
            EstimationState with all completed stage outputs.
        """
        state = EstimationState()
        logger.info(
            "pipeline_run_start",
            run_id=state.run_id,
            file=file_path,
            through_phase=run_through_phase,
        )

        # Phase 1
        state = self.ingest_document(state, file_path)
        if run_through_phase >= 1 and state.errors:
            return state

        state = self.extract_specification(state, manual_overrides)
        if run_through_phase >= 1 and state.errors:
            return state

        if run_through_phase < 2:
            return state

        # Phase 2
        state = self.generate_queries(state)
        if state.errors:
            return state

        if run_through_phase < 3:
            return state

        # Phase 3
        state = self.execute_search(state)
        state = self.retrieve_pages(state)

        if run_through_phase < 4:
            return state

        # Phase 4+ stubs
        state = self.discover_candidates(state)
        state = self.match_vessels(state)
        state = self.extract_cost_evidence(state)

        if run_through_phase < 5:
            return state

        state = self.normalize_costs(state)
        state = self.apply_adjustments(state)

        if run_through_phase < 6:
            return state

        state = self.estimate_cost(state)

        if run_through_phase < 7:
            return state

        state = self.generate_report(state)

        logger.info(
            "pipeline_run_complete",
            run_id=state.run_id,
            steps=len(state.steps_completed),
            warnings=len(state.warnings),
            errors=len(state.errors),
        )

        return state
