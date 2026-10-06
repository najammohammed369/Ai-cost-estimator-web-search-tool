"""
CLI entry point for the Vessel Cost Estimator.

Usage:
    python run.py --file path/to/spec.pdf
    python run.py --file path/to/spec.pdf --phase 2
    python run.py --server
"""

import argparse
import sys
import json
from pathlib import Path

# Force stdout/stderr to UTF-8 with replacement for unencodable characters on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import structlog

# Configure structlog for pretty console output
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=False),
    ],
)

logger = structlog.get_logger()


def run_pipeline(file_path: str, phase: int = 3) -> None:
    """Run the estimation pipeline on a vessel specification document."""
    from app.orchestration.pipeline import VesselCostPipeline

    if not Path(file_path).exists():
        logger.error("file_not_found", path=file_path)
        sys.exit(1)

    logger.info("starting_pipeline", file=file_path, through_phase=phase)

    pipeline = VesselCostPipeline()
    state = pipeline.run(file_path=file_path, run_through_phase=phase)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RUN SUMMARY")
    print("=" * 60)
    summary = state.to_summary_dict()
    print(json.dumps(summary, indent=2))

    if state.vessel_specification:
        print("\n" + "=" * 60)
        print("EXTRACTED VESSEL SPECIFICATION")
        print("=" * 60)
        spec = state.vessel_specification.model_dump()
        for field, value in spec.items():
            if value is not None:
                if isinstance(value, dict):
                    print(f"  {field}: {value.get('value')} {value.get('unit', '') or ''}"
                          f"  [page {value.get('source_page', '?')}]")
                elif isinstance(value, list):
                    print(f"  {field}: {[v.get('value', v) for v in value if v]}")

    if state.search_queries:
        print("\n" + "=" * 60)
        print(f"SEARCH QUERIES ({len(state.search_queries)})")
        print("=" * 60)
        for q in state.search_queries:
            print(f"  [{q['category']}] {q['query']}")

    if state.search_results:
        print("\n" + "=" * 60)
        print(f"SEARCH RESULTS ({len(state.search_results)} unique URLs)")
        print("=" * 60)
        for r in state.search_results[:10]:
            print(f"  [Tier {r.source_tier}] {r.title[:60]} | {r.url[:70]}")

    if state.estimate:
        print("\n" + "=" * 60)
        print("VESSEL COST ESTIMATE (TARGET YEAR 2026)")
        print("=" * 60)
        est = state.estimate
        print(f"  Low Estimate:     ${est.low_usd_derived:>14,.0f} USD   \u20b9{est.low_inr_derived:>18,.0f} INR")
        print(f"  Central Estimate: ${est.central_usd_derived:>14,.0f} USD   \u20b9{est.central_inr_derived:>18,.0f} INR")
        print(f"  High Estimate:    ${est.high_usd_derived:>14,.0f} USD   \u20b9{est.high_inr_derived:>18,.0f} INR")
        print(f"  Confidence:       {est.confidence.upper()}")
        print(f"  Comparables used: {est.comparable_count}")
        print(f"  Methodology:      {est.methodology[:120]}..." if len(est.methodology) > 120 else f"  Methodology:      {est.methodology}")

    if state.candidate_vessels:
        print("\n" + "=" * 60)
        print(f"CANDIDATE VESSELS DISCOVERED ({len(state.candidate_vessels)})")
        print("=" * 60)
        for c in state.candidate_vessels[:10]:
            cost = f" | {c.contract_currency or ''}{c.contract_value:,.0f}" if c.contract_value else ""
            print(f"  [{c.data_category}] {c.vessel_name} ({c.vessel_type or '?'}) {c.country or ''}{cost}")

    report_path = getattr(state, "_report_path", None)
    if report_path:
        print("\n" + "=" * 60)
        print(f"EXCEL REPORT: {report_path}")
        print("=" * 60)

    if state.warnings:
        print("\nWARNINGS:")
        for w in state.warnings:
            print(f"  * {w}")


def main():
    parser = argparse.ArgumentParser(
        description="Vessel Cost Estimator — Evidence-driven vessel cost estimation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --file spec.pdf                   Run full pipeline (phases 1-6)
  python run.py --file spec.pdf --phase 1         Run phase 1 only (ingestion + extraction)
  python run.py --file spec.pdf --phase 3         Run phases 1-3 (ingestion, spec, queries & search)
  python run.py --file spec.pdf --phase 6         Run full pipeline (+ candidates, matching & cost estimation)
        """,
    )

    parser.add_argument(
        "--file", type=str, required=True,
        help="Path to vessel specification document (PDF, DOCX, or TXT)",
    )
    parser.add_argument(
        "--phase", type=int, default=8,
        help="Run through this phase (1=spec, 2=queries, 3=search, 4=candidates, 5=costs, 6=estimate, 7=report). Default: 8 (full run)",
    )

    args = parser.parse_args()

    run_pipeline(args.file, phase=args.phase)


if __name__ == "__main__":
    main()
