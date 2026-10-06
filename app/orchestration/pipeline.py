"""
Pipeline — fixed sequential orchestration for the vessel cost estimation workflow.

Phases 1–8 are fully implemented.

Steps:
  1. ingest_document         — PDF/DOCX/TXT → DocumentContent
  2. extract_specification   — DocumentContent → VesselSpecification (via LLM)
  3. generate_queries        — VesselProfile → list[SearchQuery] (via LLM)
  4. execute_search          — queries → list[SearchResult] (via SerpAPI)
  5. retrieve_pages          — SearchResult URLs → list[RetrievedPage]
  6. discover_candidates     — RetrievedPage → list[CandidateVessel]
  7. match_vessels           — CandidateVessel → ranked comparables
  8. extract_cost_evidence   — comparables → cost evidence
  9. normalize_costs         — cost evidence → normalized costs
  10. apply_adjustments      — normalized costs → adjusted costs
  11. estimate_cost          — adjusted costs → CostEstimate
  12. generate_report        — CostEstimate → HTML report
"""

from openpyxl.styles.colors import BLACK
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
                f"Central estimate: ${cost_est.central_usd_derived:,.0f} USD ≈ ₹{cost_est.central_inr_derived:,.0f} INR (Confidence: {cost_est.confidence})",
                duration_s=round(time.time() - t0, 2),
            )

        except Exception as e:
            state.add_error(f"Cost estimation failed: {e}")
            state.mark_step("estimate_cost", "failed", str(e))
            logger.error("pipeline_step_failed", step="estimate_cost", error=str(e))
            raise

        return state

    def generate_report(self, state: EstimationState) -> EstimationState:
        """[Phase 8] Generate a rich multi-sheet Excel cost estimate report."""
        import os
        from datetime import datetime
        from openpyxl import Workbook
        from openpyxl.styles import (
            Font, PatternFill, Alignment, Border, Side, GradientFill
        )
        from openpyxl.utils import get_column_letter
        from openpyxl.styles.numbers import FORMAT_NUMBER_COMMA_SEPARATED1

        logger.info("pipeline_step_start", step="generate_report", run_id=state.run_id)
        t0 = time.time()

        # ── Colour palette ────────────────────────────────────────────────────
        DARK_NAVY   = "0F172A"
        MID_NAVY    = "1E3A5F"
        LIGHT_NAVY  = "1E293B"
        STEEL       = "293548"
        SKY_BLUE    = "38BDF8"
        INDIGO      = "818CF8"
        GREEN       = "22C55E"
        AMBER       = "F59E0B"
        RED         = "EF4444"
        SLATE       = "94A3B8"
        WHITE       = "FFFFFF"
        BLACK       = "000000"
        LIGHT_GRAY  = "F1F5F9"
        DARK_TEXT   = "0F172A"

        # ── Helper style factories ────────────────────────────────────────────
        def _fill(hex_color: str) -> PatternFill:
            return PatternFill("solid", fgColor=hex_color)

        def _font(bold=False, color=BLACK, size=11, italic=False) -> Font:
            return Font(bold=bold, color=color, size=size, italic=italic,
                        name="Calibri")

        def _border(color="D1D5DB") -> Border:
            side = Side(style="thin", color=color)
            return Border(left=side, right=side, top=side, bottom=side)

        def _center() -> Alignment:
            return Alignment(horizontal="center", vertical="center",
                             wrap_text=True)

        def _left(wrap=True) -> Alignment:
            return Alignment(horizontal="left", vertical="center",
                             wrap_text=wrap)

        def _right() -> Alignment:
            return Alignment(horizontal="right", vertical="center")

        def _header_row(ws, row: int, cols: list[str],
                        bg: str = WHITE) -> None:
            """Write a styled header row."""
            for c, label in enumerate(cols, 1):
                cell = ws.cell(row=row, column=c, value=label)
                cell.font      = _font(bold=True, color=BLACK, size=10)
                cell.fill      = _fill(bg)
                cell.alignment = _center()
                cell.border    = _border(STEEL)

        def _data_row(ws, row: int, values: list,
                      bg: str = WHITE, text_color: str = DARK_TEXT,
                      bold: bool = False, num_fmt: str | None = None) -> None:
            for c, val in enumerate(values, 1):
                cell = ws.cell(row=row, column=c, value=val)
                cell.font      = _font(bold=bold, color=text_color, size=10)
                cell.fill      = _fill(bg)
                cell.alignment = _left()
                cell.border    = _border()
                if num_fmt and isinstance(val, (int, float)):
                    cell.number_format = num_fmt

        def _set_col_widths(ws, widths: list[int]) -> None:
            for i, w in enumerate(widths, 1):
                ws.column_dimensions[get_column_letter(i)].width = w

        def _title_block(ws, title: str, subtitle: str = "") -> None:
            """Write a branded title block in the first 3 rows."""
            ws.row_dimensions[1].height = 36
            ws.row_dimensions[2].height = 20
            ws.row_dimensions[3].height = 14

            # Row 1 — main title (merge A1:G1)
            ws.merge_cells("A1:G1")
            t = ws["A1"]
            t.value     = title
            t.font      = Font(bold=True, color=BLACK, size=16, name="Calibri")
            t.fill      = _fill(WHITE)
            t.alignment = _left(wrap=False)

            # Row 2 — subtitle
            ws.merge_cells("A2:G2")
            s = ws["A2"]
            s.value     = subtitle
            s.font      = _font(color=SLATE, size=10, italic=True)
            s.fill      = _fill(WHITE)
            s.alignment = _left(wrap=False)

            # Row 3 — spacer
            ws.merge_cells("A3:G3")
            ws["A3"].fill = _fill(WHITE)

        # ── Gather data ───────────────────────────────────────────────────────
        est     = state.estimate
        profile = state.vessel_profile
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        vessel_type = "Unknown Vessel"
        if profile and profile.specification.vessel_type:
            vessel_type = profile.specification.vessel_type.value or "Unknown Vessel"

        # ── Create workbook ───────────────────────────────────────────────────
        wb = Workbook()
        wb.remove(wb.active)          # remove default blank sheet

        # =====================================================================
        # SHEET 1 — Summary
        # =====================================================================
        ws1 = wb.create_sheet("Summary")
        ws1.sheet_view.showGridLines = False

        _title_block(
            ws1,
            f"Vessel Cost Estimation Report — {vessel_type}",
            f"Run ID: {state.run_id}  |  Generated: {now_str}  |  Document: {profile.file_name if profile else 'N/A'}"
        )
        _set_col_widths(ws1, [26, 30, 16, 16, 16, 16, 20])

        # Section header
        r = 4
        ws1.merge_cells(f"A{r}:G{r}")
        ws1[f"A{r}"].value     = "RUN SUMMARY"
        ws1[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws1[f"A{r}"].fill      = _fill(WHITE)
        ws1[f"A{r}"].alignment = _left()
        ws1.row_dimensions[r].height = 22
        r += 1

        summary_items = [
            ("Run ID",                  state.run_id),
            ("Document",                profile.file_name if profile else "N/A"),
            ("Vessel Type",             vessel_type),
            ("Generated",               now_str),
            ("Search Queries",          len(state.search_queries)),
            ("Search Results",          len(state.search_results)),
            ("Pages Retrieved",         len(state.retrieved_pages)),
            ("Candidate Vessels Found", len(state.candidate_vessels)),
            ("Comparables Selected",    len(state.selected_comparables)),
            ("Cost Evidence Items",     len(state.cost_evidence)),
            ("Normalised Costs",        len(state.normalized_costs)),
            ("Pipeline Steps Completed",len(state.steps_completed)),
            ("Errors",                  len(state.errors)),
            ("Warnings",                len(state.warnings)),
        ]

        alt = False
        for label, value in summary_items:
            bg = WHITE if alt else WHITE
            c1 = ws1.cell(row=r, column=1, value=label)
            c1.font = _font(bold=True, color=BLACK, size=10)
            c1.fill = _fill(bg)
            c1.alignment = _left()
            c1.border = _border()

            c2 = ws1.cell(row=r, column=2, value=value)
            c2.font = _font(color=BLACK, size=10)
            c2.fill = _fill(bg)
            c2.alignment = _left()
            c2.border = _border()
            alt = not alt
            r += 1

        # =====================================================================
        # SHEET 2 — Cost Estimate
        # =====================================================================
        ws2 = wb.create_sheet("Cost Estimate")
        ws2.sheet_view.showGridLines = False
        _set_col_widths(ws2, [28, 22, 22, 22, 28])

        _title_block(
            ws2,
            f"Cost Estimate — {vessel_type}",
            f"Target Year: {est.estimate_year if est else 2026}  |  Confidence: {est.confidence.upper() if est else 'N/A'}  |  Comparables: {est.comparable_count if est else 0}"
        )

        r = 4
        # Sub-header: Cost Range
        ws2.merge_cells(f"A{r}:E{r}")
        ws2[f"A{r}"].value     = "COST ESTIMATE RANGE"
        ws2[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws2[f"A{r}"].fill      = _fill(WHITE)
        ws2[f"A{r}"].alignment = _left()
        ws2.row_dimensions[r].height = 22
        r += 1

        # Column headers
        _header_row(ws2, r, ["Scenario", "Amount (USD)", "Amount (INR)", "Currency", "Notes"])
        r += 1

        if est:
            scenarios = [
                ("Low Estimate",           est.low_usd_derived,     est.low_inr_derived,     "USD / INR", "Conservative estimate",          GREEN),
                ("Central Estimate (P50)", est.central_usd_derived, est.central_inr_derived, "USD / INR", "Most likely estimate (median)",   SKY_BLUE),
                ("High Estimate",          est.high_usd_derived,    est.high_inr_derived,    "USD / INR", "Optimistic / upper-bound estimate",AMBER),
            ]
            for label, usd_val, inr_val, curr, note, color in scenarios:
                ws2.row_dimensions[r].height = 24
                for c_idx, (val, fmt) in enumerate([
                    (label,   None),
                    (usd_val, '"$"#,##0'),
                    (inr_val, '"₹"#,##0'),
                    (curr,    None),
                    (note,    None),
                ], 1):
                    cell = ws2.cell(row=r, column=c_idx, value=val)
                    cell.fill      = _fill(WHITE if c_idx > 1 else color + "33")
                    cell.alignment = _right() if c_idx in (2, 3) else _left()
                    cell.border    = _border()
                    cell.font      = Font(
                        bold=(c_idx == 1),
                        color=color if c_idx == 1 else BLACK,
                        size=11 if c_idx in (2, 3) else 10,
                        name="Calibri"
                    )
                    if fmt and isinstance(val, (int, float)):
                        cell.number_format = fmt
                r += 1
        else:
            ws2.cell(row=r, column=1, value="No estimate generated.")
            r += 1

        # Confidence & Methodology block
        r += 1
        ws2.merge_cells(f"A{r}:E{r}")
        ws2[f"A{r}"].value     = "CONFIDENCE & METHODOLOGY"
        ws2[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws2[f"A{r}"].fill      = _fill(WHITE)
        ws2[f"A{r}"].alignment = _left()
        ws2.row_dimensions[r].height = 22
        r += 1

        conf_label = est.confidence.upper() if est else "N/A"
        conf_color_map = {"HIGH": GREEN, "MEDIUM": AMBER, "LOW": RED, "INSUFFICIENT": SLATE}
        conf_color = conf_color_map.get(conf_label, SLATE)

        for label, value in [
            ("Confidence Level", conf_label),
            ("Comparables Used", est.comparable_count if est else 0),
            ("Estimate Year",    est.estimate_year if est else 2026),
            ("Primary Currency", est.currency if est else "N/A"),
            ("Methodology",      est.methodology if est else "N/A"),
        ]:
            c1 = ws2.cell(row=r, column=1, value=label)
            c1.font = _font(bold=True, color=BLACK, size=10)
            c1.fill = _fill(WHITE)
            c1.alignment = _left()
            c1.border = _border()

            ws2.merge_cells(f"B{r}:E{r}")
            c2 = ws2.cell(row=r, column=2, value=value)
            c2.font = Font(
                bold=(label == "Confidence Level"),
                color=conf_color if label == "Confidence Level" else DARK_TEXT,
                size=10, name="Calibri"
            )
            c2.fill      = _fill(WHITE)
            c2.alignment = _left()
            c2.border    = _border()
            ws2.row_dimensions[r].height = 18 if label != "Methodology" else 40
            r += 1

        # =====================================================================
        # SHEET 3 — Comparable Vessels
        # =====================================================================
        ws3 = wb.create_sheet("Comparable Vessels")
        ws3.sheet_view.showGridLines = False
        _set_col_widths(ws3, [32, 22, 16, 10, 16, 18, 18, 10, 12, 36])

        _title_block(
            ws3,
            "Comparable Vessel Dataset",
            f"All {len(state.candidate_vessels)} discovered candidates | Top {len(state.selected_comparables)} selected for estimation"
        )

        r = 4
        ws3.merge_cells(f"A{r}:J{r}")
        ws3[f"A{r}"].value     = "SELECTED COMPARABLES (Used in Estimation)"
        ws3[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws3[f"A{r}"].fill      = _fill(WHITE)
        ws3[f"A{r}"].alignment = _left()
        ws3.row_dimensions[r].height = 22
        r += 1

        _header_row(ws3, r, [
            "Vessel Name", "Vessel Type", "Country", "Year",
            "Displacement (t)", "Contract Value", "Currency",
            "Data Cat.", "Source Tier", "Source URLs"
        ])
        r += 1

        alt = False
        for c in state.selected_comparables:
            bg = "E8F5E9" if not alt else WHITE
            row_vals = [
                c.vessel_name,
                c.vessel_type or "",
                c.country or "",
                c.delivery_year or c.contract_year or "",
                c.displacement_t or "",
                c.contract_value or "",
                c.contract_currency or "",
                c.data_category,
                c.source_tier,
                "; ".join(c.source_urls[:2]) if c.source_urls else "",
            ]
            _data_row(ws3, r, row_vals, bg=bg, text_color=DARK_TEXT)
            # Currency format for contract value column (F)
            cv_cell = ws3.cell(row=r, column=6)
            if isinstance(cv_cell.value, (int, float)):
                cv_cell.number_format = "#,##0"
            r += 1
            alt = not alt

        # All candidates section
        r += 1
        ws3.merge_cells(f"A{r}:J{r}")
        ws3[f"A{r}"].value     = "ALL DISCOVERED CANDIDATE VESSELS"
        ws3[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws3[f"A{r}"].fill      = _fill(WHITE)
        ws3[f"A{r}"].alignment = _left()
        ws3.row_dimensions[r].height = 22
        r += 1

        _header_row(ws3, r, [
            "Vessel Name", "Vessel Type", "Country", "Year",
            "Displacement (t)", "Contract Value", "Currency",
            "Data Cat.", "Source Tier", "Mission"
        ])
        r += 1

        alt = False
        for c in state.candidate_vessels:
            bg = LIGHT_GRAY if alt else WHITE
            row_vals = [
                c.vessel_name,
                c.vessel_type or "",
                c.country or "",
                c.delivery_year or c.contract_year or "",
                c.displacement_t or "",
                c.contract_value or "",
                c.contract_currency or "",
                c.data_category,
                c.source_tier,
                c.mission or "",
            ]
            _data_row(ws3, r, row_vals, bg=bg, text_color=DARK_TEXT)
            cv_cell = ws3.cell(row=r, column=6)
            if isinstance(cv_cell.value, (int, float)):
                cv_cell.number_format = "#,##0"
            r += 1
            alt = not alt

        # =====================================================================
        # SHEET 4 — Vessel Specification
        # =====================================================================
        ws4 = wb.create_sheet("Vessel Specification")
        ws4.sheet_view.showGridLines = False
        _set_col_widths(ws4, [32, 36, 14, 16, 40])

        _title_block(
            ws4,
            f"Extracted Vessel Specification — {vessel_type}",
            f"Extracted by LLM from document: {profile.file_name if profile else 'N/A'}"
        )

        r = 4
        ws4.merge_cells(f"A{r}:E{r}")
        ws4[f"A{r}"].value     = "SPECIFICATION PARAMETERS"
        ws4[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws4[f"A{r}"].fill      = _fill(WHITE)
        ws4[f"A{r}"].alignment = _left()
        ws4.row_dimensions[r].height = 22
        r += 1

        _header_row(ws4, r, ["Parameter", "Value", "Unit", "Source Page", "Source Text"])
        r += 1

        if profile:
            spec_dict = profile.specification.model_dump()
            alt = False
            for field_name, val in spec_dict.items():
                if val is None:
                    continue
                bg = LIGHT_GRAY if alt else WHITE

                if isinstance(val, dict):
                    disp_val   = val.get("value", "")
                    unit       = val.get("unit") or ""
                    src_page   = val.get("source_page") or ""
                    src_text   = val.get("source_text") or ""
                    confidence = val.get("confidence") or ""
                elif isinstance(val, list):
                    combined = []
                    for item in val:
                        if isinstance(item, dict):
                            combined.append(str(item.get("value", item)))
                        else:
                            combined.append(str(item))
                    disp_val = ", ".join(combined)
                    unit = src_page = src_text = confidence = ""
                else:
                    disp_val = str(val)
                    unit = src_page = src_text = confidence = ""

                row_vals = [
                    field_name.replace("_", " ").title(),
                    disp_val,
                    unit,
                    src_page,
                    src_text,
                ]
                _data_row(ws4, r, row_vals, bg=bg, text_color=DARK_TEXT)
                r += 1
                alt = not alt

        # Manual overrides sub-section
        if profile and profile.manual_overrides:
            r += 1
            ws4.merge_cells(f"A{r}:E{r}")
            ws4[f"A{r}"].value     = "MANUAL OVERRIDES"
            ws4[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
            ws4[f"A{r}"].fill      = _fill(WHITE)
            ws4[f"A{r}"].alignment = _left()
            ws4.row_dimensions[r].height = 22
            r += 1
            _header_row(ws4, r, ["Field", "Overridden Value", "Unit", "", ""], bg=WHITE)
            r += 1
            for ov in profile.manual_overrides:
                _data_row(ws4, r, [ov.field_name, ov.value, ov.unit or "", "", ""],
                          bg="FFF8E1", text_color=DARK_TEXT)
                r += 1

        # =====================================================================
        # SHEET 5 — Pipeline Audit
        # =====================================================================
        ws5 = wb.create_sheet("Pipeline Audit")
        ws5.sheet_view.showGridLines = False
        _set_col_widths(ws5, [28, 14, 60, 14])

        _title_block(
            ws5,
            "Pipeline Execution Audit Log",
            f"Run ID: {state.run_id}  |  {len(state.steps_completed)} steps  |  Errors: {len(state.errors)}  |  Warnings: {len(state.warnings)}"
        )

        r = 4
        ws5.merge_cells(f"A{r}:D{r}")
        ws5[f"A{r}"].value     = "PIPELINE STEPS"
        ws5[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws5[f"A{r}"].fill      = _fill(WHITE)
        ws5[f"A{r}"].alignment = _left()
        ws5.row_dimensions[r].height = 22
        r += 1

        _header_row(ws5, r, ["Step Name", "Status", "Message", "Duration (s)"])
        r += 1

        STATUS_COLORS = {"success": "E8F5E9", "warning": "FFF8E1", "failed": "FFEBEE"}
        STATUS_TEXT   = {"success": "155724", "warning": "856404", "failed": "842029"}

        for step in state.steps_completed:
            bg       = STATUS_COLORS.get(step.status, WHITE)
            txt      = STATUS_TEXT.get(step.status, DARK_TEXT)
            icon_map = {"success": "✅", "warning": "⚠️", "failed": "❌"}
            icon     = icon_map.get(step.status, "ℹ️")

            row_vals = [
                f"{icon} {step.step_name}",
                step.status.upper(),
                step.message,
                step.duration_s,
            ]
            _data_row(ws5, r, row_vals, bg=bg, text_color=txt)
            ws5.cell(row=r, column=4).number_format = "0.00"
            ws5.row_dimensions[r].height = 18
            r += 1

        # Warnings
        r += 1
        ws5.merge_cells(f"A{r}:D{r}")
        ws5[f"A{r}"].value     = "WARNINGS"
        ws5[f"A{r}"].font      = _font(bold=True, color=BLACK, size=11)
        ws5[f"A{r}"].fill      = _fill(WHITE)
        ws5[f"A{r}"].alignment = _left()
        ws5.row_dimensions[r].height = 22
        r += 1

        if state.warnings:
            for i, w in enumerate(state.warnings, 1):
                ws5.merge_cells(f"A{r}:D{r}")
                cell = ws5.cell(row=r, column=1, value=f"{i}. {w}")
                cell.font      = _font(color=DARK_TEXT, size=10)
                cell.fill      = _fill("FFF8E1")
                cell.alignment = _left()
                cell.border    = _border()
                ws5.row_dimensions[r].height = 30
                r += 1
        else:
            ws5.merge_cells(f"A{r}:D{r}")
            cell = ws5.cell(row=r, column=1, value="No warnings.")
            cell.font      = _font(color=SLATE, size=10, italic=True)
            cell.fill      = _fill(WHITE)
            cell.alignment = _left()
            r += 1

        # Errors
        r += 1
        ws5.merge_cells(f"A{r}:D{r}")
        ws5[f"A{r}"].value     = "ERRORS"
        ws5[f"A{r}"].font      = _font(bold=True, color=RED, size=11)
        ws5[f"A{r}"].fill      = _fill(WHITE)
        ws5[f"A{r}"].alignment = _left()
        ws5.row_dimensions[r].height = 22
        r += 1

        if state.errors:
            for i, e in enumerate(state.errors, 1):
                ws5.merge_cells(f"A{r}:D{r}")
                cell = ws5.cell(row=r, column=1, value=f"{i}. {e}")
                cell.font      = _font(color="842029", size=10)
                cell.fill      = _fill("FFEBEE")
                cell.alignment = _left()
                cell.border    = _border()
                ws5.row_dimensions[r].height = 30
                r += 1
        else:
            ws5.merge_cells(f"A{r}:D{r}")
            cell = ws5.cell(row=r, column=1, value="No errors.")
            cell.font      = _font(color=SLATE, size=10, italic=True)
            cell.fill      = _fill(WHITE)
            cell.alignment = _left()

        # ── Save workbook ─────────────────────────────────────────────────────
        try:
            output_dir  = os.path.join(os.getcwd(), "reports")
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, f"estimate_{state.run_id}.xlsx")
            wb.save(report_path)

            state.mark_step(
                "generate_report", "success",
                f"Excel report saved \u2192 {report_path}",
                duration_s=round(time.time() - t0, 2),
            )
            logger.info("generate_report_complete", run_id=state.run_id, path=report_path)
            state._report_path = report_path

        except Exception as e:
            state.add_error(f"Report generation failed: {e}")
            state.mark_step("generate_report", "failed", str(e))
            logger.error("pipeline_step_failed", step="generate_report", error=str(e))

        return state

    # =========================================================================
    # Full Pipeline Run (Phases 1–3 + stubs)
    # =========================================================================

    def run(
        self,
        file_path: str,
        manual_overrides: list[ManualOverride] | None = None,
        run_through_phase: int = 8,
    ) -> EstimationState:
        """
        Execute the full pipeline up to the specified phase.

        Args:
            file_path: Path to the vessel specification document.
            manual_overrides: Optional user-provided parameter overrides.
            run_through_phase: Stop after this phase (1-8). Default: 8 (full run).

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

        # ── Phase 1: Ingest document + extract specification ──────────────────
        state = self.ingest_document(state, file_path)
        if state.errors:
            return state

        state = self.extract_specification(state, manual_overrides)
        if state.errors:
            return state

        if run_through_phase < 2:
            return state

        # ── Phase 2: Generate search queries ─────────────────────────────────
        state = self.generate_queries(state)
        if state.errors:
            return state

        if run_through_phase < 3:
            return state

        # ── Phase 3: Web search + retrieve pages ──────────────────────────────
        state = self.execute_search(state)
        state = self.retrieve_pages(state)

        if run_through_phase < 4:
            return state

        # ── Phase 4: Candidate discovery + vessel matching ────────────────────
        state = self.discover_candidates(state)
        state = self.match_vessels(state)

        if run_through_phase < 5:
            return state

        # ── Phase 5: Cost evidence extraction + normalization + adjustments ───
        state = self.extract_cost_evidence(state)
        state = self.normalize_costs(state)
        state = self.apply_adjustments(state)

        if run_through_phase < 6:
            return state

        # ── Phase 6: Final cost estimate ──────────────────────────────────────
        state = self.estimate_cost(state)

        if run_through_phase < 7:
            return state

        # ── Phase 7: Generate report ──────────────────────────────────────────
        state = self.generate_report(state)

        return state
    def extract_cost_evidence(self, state: EstimationState) -> EstimationState:
        """[Phase 4] Extract cost evidence from retrieved webpages.

        This implementation scans the cleaned text of each retrieved page for
        monetary amounts (e.g. $120 M, USD 150 000, INR 1,20,00,000) and stores
        them in ``state.cost_evidence`` as a list of dictionaries containing the
        detected amount, currency, source URL and a short snippet.
        The extraction is deliberately lightweight – it uses regular
        expressions and does not depend on heavy NLP models, keeping the pipeline
        fast and easy to run on a typical developer laptop.
        """
        import re, time
        logger.info("extract_cost_evidence start", run_id=state.run_id)
        t0 = time.time()

        # Regex to capture monetary values with optional currency symbols/code
        money_regex = re.compile(r"(?i)(?P<amount>\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?)(?:\s*(?P<currency>USD|INR|SGD|EUR|GBP|\$|\u00A3|\u20B9))?")

        evidence = []
        for page in state.retrieved_pages:
            for match in money_regex.finditer(page.content):
                raw_amount = match.group('amount')
                amount_clean = raw_amount.replace(',', '').replace(' ', '').replace('​', '')
                try:
                    amount_val = float(amount_clean)
                except ValueError:
                    continue
                currency = match.group('currency')
                if currency:
                    currency = currency.upper()
                    if currency == '$':
                        currency = 'USD'
                    elif currency == '\u00A3':
                        currency = 'GBP'
                    elif currency == '\u20B9':
                        currency = 'INR'
                else:
                    # Simple heuristic: large numbers likely INR (Indian projects)
                    currency = 'INR' if amount_val > 1e4 else 'USD'
                start, end = match.span()
                snippet_start = max(0, start - 40)
                snippet_end = min(len(page.content), end + 40)
                snippet = page.content[snippet_start:snippet_end].replace('\n', ' ')
                evidence.append({
                    "amount": amount_val,
                    "currency": currency,
                    "source_url": page.url,
                    "snippet": snippet.strip(),
                })
        state.cost_evidence = evidence
        logger.info(
            "extract_cost_evidence complete",
            run_id=state.run_id,
            found=len(evidence),
            duration_s=round(time.time() - t0, 2),
        )
        state.mark_step(
            "extract_cost_evidence",
            "success",
            f"Extracted {len(evidence)} cost items",
            duration_s=round(time.time() - t0, 2),
        )
        return state

    def normalize_costs(self, state: EstimationState) -> EstimationState:
        """
        [Phase 5a] Normalize raw cost evidence to a common base currency (INR, 2026).

        Reads state.cost_evidence (list of dicts from extract_cost_evidence),
        converts each monetary value to INR using reference exchange rates,
        applies a simple recency inflation adjustment, and stores the results
        in state.normalized_costs.
        """
        import math

        logger.info("pipeline_step_start", step="normalize_costs", run_id=state.run_id)
        t0 = time.time()

        # Reference exchange rates → INR
        FX_TO_INR: dict[str, float] = {
            "USD": 83.0,
            "EUR": 91.0,
            "GBP": 106.0,
            "INR": 1.0,
            "SGD": 62.0,
            "AUD": 55.0,
            "CAD": 62.0,
            "JPY": 0.56,
            "CNY": 11.5,
            "AED": 22.6,
        }
        ANNUAL_INFLATION = 0.030
        TARGET_YEAR = 2026

        normalized: list[dict] = []
        for ev in state.cost_evidence:
            amount = ev.get("amount", 0)
            currency = (ev.get("currency") or "INR").upper()
            if currency == "$":
                currency = "USD"
            fx = FX_TO_INR.get(currency, 1.0)
            amount_inr = amount * fx

            # Skip trivially small amounts (likely page numbers, years, etc.)
            if amount_inr < 1_000:
                continue

            # Assume costs found in text are current; apply 0 inflation by default
            # A future enhancement could extract the year from context.
            year_assumed = TARGET_YEAR
            inflation_factor = math.pow(1 + ANNUAL_INFLATION, max(0, TARGET_YEAR - year_assumed))
            amount_inr_adjusted = amount_inr * inflation_factor

            normalized.append({
                "amount_inr": round(amount_inr_adjusted, 2),
                "original_amount": amount,
                "original_currency": currency,
                "fx_rate": fx,
                "source_url": ev.get("source_url", ""),
                "snippet": ev.get("snippet", ""),
            })

        state.normalized_costs = normalized

        state.mark_step(
            "normalize_costs", "success",
            f"Normalized {len(normalized)} cost data points to INR 2026",
            duration_s=round(time.time() - t0, 2),
        )
        logger.info(
            "normalize_costs_complete",
            run_id=state.run_id,
            normalized_count=len(normalized),
        )
        return state

    def apply_adjustments(self, state: EstimationState) -> EstimationState:
        """
        [Phase 5b] Apply scope and risk adjustments to normalized costs.

        Adjustments applied:
          • Programme support factor (+5–15% if multi-vessel contract or
            spares/training are included)
          • Complexity uplift based on vessel type
          • Records the adjustment log in state.adjustments for auditability.
        """
        logger.info("pipeline_step_start", step="apply_adjustments", run_id=state.run_id)
        t0 = time.time()

        vessel_type_str = ""
        if state.vessel_profile and state.vessel_profile.specification.vessel_type:
            vessel_type_str = (state.vessel_profile.specification.vessel_type.value or "").lower()

        # Complexity multiplier by vessel type keyword
        COMPLEXITY: dict[str, float] = {
            "submarine": 1.30,
            "destroyer": 1.20,
            "frigate": 1.15,
            "corvette": 1.10,
            "patrol": 1.05,
            "opv": 1.05,
            "offshore patrol": 1.05,
            "research": 1.08,
            "survey": 1.08,
            "auxiliary": 1.02,
            "tanker": 1.02,
        }
        complexity_factor = 1.0
        for kw, factor in COMPLEXITY.items():
            if kw in vessel_type_str:
                complexity_factor = factor
                break

        adjustments: list[dict] = []

        # Apply adjustments to selected comparable costs stored in state
        for c in state.selected_comparables:
            adj_entry: dict = {
                "vessel_name": c.vessel_name,
                "complexity_factor": complexity_factor,
                "support_factor": 1.0,
                "total_factor": complexity_factor,
                "notes": [],
            }

            # Multi-vessel support factor
            if c.program_includes_support:
                support_factor = 1.10
                adj_entry["support_factor"] = support_factor
                adj_entry["total_factor"] = round(complexity_factor * support_factor, 3)
                adj_entry["notes"].append("Programme includes support; +10% uplift applied")

            if complexity_factor != 1.0:
                adj_entry["notes"].append(
                    f"Complexity uplift for '{vessel_type_str}' vessel: ×{complexity_factor:.2f}"
                )

            adjustments.append(adj_entry)

        state.adjustments = adjustments

        state.mark_step(
            "apply_adjustments", "success",
            f"Applied complexity factor ×{complexity_factor:.2f} to {len(adjustments)} vessel comparables",
            duration_s=round(time.time() - t0, 2),
        )
        logger.info(
            "apply_adjustments_complete",
            run_id=state.run_id,
            complexity_factor=complexity_factor,
            adjustments_count=len(adjustments),
        )
        return state
