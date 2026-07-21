import asyncio
import time
from pathlib import Path
from typing import Optional

from result_companion._internal.analysis_helpers import (
    apply_concurrency_overrides,
    resolve_tags,
)
from result_companion.api import run_analysis
from result_companion.core.chunking.chunking import ChunkingStrategy
from result_companion.core.parsers.config import DefaultConfigModel, load_config
from result_companion.core.plugins.base import (
    ParsedResults,
    ParseOptions,
    ResultParserPlugin,
)
from result_companion.core.plugins.registry import get_plugin, validate_options
from result_companion.core.results.analysis_result import AnalysisResult
from result_companion.core.results.text_report import (
    render_json_report,
    render_text_report,
)
from result_companion.core.utils.llm_debug import LLMDebugLogger
from result_companion.core.utils.log_levels import LogLevels
from result_companion.core.utils.logging_config import logger, set_global_log_level


async def _main(
    output: Path,
    log_level: LogLevels,
    config: Optional[Path],
    report: Optional[str],
    include_passing: bool,
    test_case_concurrency: Optional[int] = None,
    chunk_concurrency: Optional[int] = None,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    dryrun: bool = False,
    html_report: bool = True,
    text_report: Optional[str] = None,
    json_report: Optional[str] = None,
    print_text_report: bool = False,
    summarize_failures: bool = False,
    quiet: bool = False,
    debug_log: Optional[Path] = None,
    result_format: Optional[str] = None,
) -> bool:
    resolved_log_level = "ERROR" if quiet else str(log_level)
    set_global_log_level(resolved_log_level)

    logger.info("Starting Result Companion!")
    start = time.time()
    parsed_config = load_config(config)
    if debug_log:
        parsed_config.debug_logger = LLMDebugLogger.from_path(debug_log)
    apply_concurrency_overrides(parsed_config, test_case_concurrency, chunk_concurrency)

    options = ParseOptions(
        include_tags=resolve_tags(include_tags, parsed_config.test_filter.include_tags),
        exclude_tags=resolve_tags(exclude_tags, parsed_config.test_filter.exclude_tags),
        exclude_fields=parsed_config.rendering.exclude_fields or None,
        exclude_passing=not include_passing
        and not parsed_config.test_filter.include_passing,
    )
    plugin = get_plugin(result_format, output)
    validate_options(plugin, options)
    results = plugin.parse(output, options)
    if not results.has_chunking:
        strategy = ChunkingStrategy(
            tokenizer_config=parsed_config.tokenizer,
            system_prompt=parsed_config.llm_config.question_prompt,
        )
        results.set_chunking(strategy)

    logger.info(
        f"Total tests: {results.total_test_count}, filtered: {len(results.test_names)}"
    )
    logger.debug(f"Using model: {parsed_config.llm_factory.model}")

    analysis_result = await run_analysis(
        config=parsed_config,
        results=results,
        summarize_failures=summarize_failures,
        dryrun=dryrun,
        quiet=quiet,
    )

    _emit_reports(
        output=output,
        analysis_result=analysis_result,
        config=parsed_config,
        results=results,
        plugin=plugin,
        report=report,
        html_report=html_report,
        text_report=text_report,
        json_report=json_report,
        print_text_report=print_text_report,
    )

    stop = time.time()
    logger.debug(f"Execution time: {stop - start}")
    return True


def _emit_reports(
    output: Path,
    analysis_result: AnalysisResult,
    config: DefaultConfigModel,
    results: ParsedResults,
    plugin: ResultParserPlugin,
    report: Optional[str],
    html_report: bool,
    text_report: Optional[str],
    json_report: Optional[str],
    print_text_report: bool,
) -> None:
    """Writes HTML/text/JSON reports from analysis results."""
    report_path = Path(report if report else "rc_log.html")
    if analysis_result.llm_results and html_report:
        render_html_report = getattr(plugin, "render_html_report", None)
        if not callable(render_html_report):
            if report:
                raise ValueError(
                    f"Format '{plugin.name}' does not support HTML reports. "
                    "Use --no-html-report, or use --text-report, --json-report, "
                    "or --print-text-report."
                )
            logger.warning(
                f"Format '{plugin.name}' does not support HTML reports. "
                "Skipping HTML report. Use --no-html-report to hide this warning, "
                "or use --text-report, --json-report, --print-text-report, "
                "AnalysisResult.text_report, or AnalysisResult.llm_results."
            )
        else:
            render_html_report(
                output,
                report_path,
                analysis_result.llm_results,
                {"model": config.llm_factory.model},
                analysis_result.summary,
            )
            logger.info(f"Report created: {report_path.resolve()}")

    should_emit_text = bool(text_report) or print_text_report
    if should_emit_text:
        text_output = render_text_report(
            llm_results=analysis_result.llm_results,
            analyzed_test_names=analysis_result.test_names,
            overall_summary=analysis_result.summary,
        )
        if text_report:
            Path(text_report).write_text(text_output)
            logger.info(f"Text report created: {Path(text_report).resolve()}")
        if print_text_report:
            print(text_output)

    if json_report:
        json_output = render_json_report(
            llm_results=analysis_result.llm_results,
            analyzed_test_names=analysis_result.test_names,
            overall_summary=analysis_result.summary,
            model=config.llm_factory.model,
            source_file=str(output),
            total_test_count=results.total_test_count,
            source_hash=results.source_hash,
        )
        Path(json_report).write_text(json_output)
        logger.info(f"JSON report created: {Path(json_report).resolve()}")


def run_rc(
    output: Path,
    log_level: LogLevels,
    config: Optional[Path],
    report: Optional[str],
    include_passing: bool,
    test_case_concurrency: Optional[int] = None,
    chunk_concurrency: Optional[int] = None,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    dryrun: bool = False,
    html_report: bool = True,
    text_report: Optional[str] = None,
    json_report: Optional[str] = None,
    print_text_report: bool = False,
    summarize_failures: bool = False,
    quiet: bool = False,
    debug_log: Optional[Path] = None,
    result_format: Optional[str] = None,
) -> bool:
    """Runs the Result Companion analysis.

    Args:
        output: Path to a test result artifact.
        log_level: Logging verbosity level.
        config: Optional path to user config file.
        report: Optional HTML report output path.
        include_passing: Whether to include passing tests.
        test_case_concurrency: Number of test cases to process in parallel.
        chunk_concurrency: Number of chunks to process in parallel.
        include_tags: RF tag patterns to include.
        exclude_tags: RF tag patterns to exclude.
        dryrun: If True, skip LLM calls.
        html_report: Whether to generate HTML report.
        text_report: Optional text summary output path.
        json_report: Optional JSON report output path.
        print_text_report: Whether to print text report to stdout.
        summarize_failures: Whether to ask LLM for overall failure summary.
        quiet: Whether to suppress logs and progress output.
        debug_log: Optional path to write all LLM prompts and responses to.
        result_format: Optional parser plugin name. Auto-detects when omitted.

    Returns:
        True if analysis completed successfully.
    """
    try:
        return asyncio.run(
            _main(
                output=output,
                result_format=result_format,
                log_level=log_level,
                config=config,
                report=report,
                html_report=html_report,
                text_report=text_report,
                json_report=json_report,
                print_text_report=print_text_report,
                summarize_failures=summarize_failures,
                quiet=quiet,
                include_passing=include_passing,
                test_case_concurrency=test_case_concurrency,
                chunk_concurrency=chunk_concurrency,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                dryrun=dryrun,
                debug_log=debug_log,
            )
        )
    except Exception:
        logger.critical("Unhandled exception", exc_info=True)
        raise
