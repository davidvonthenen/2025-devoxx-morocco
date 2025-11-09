"""Helpers for detecting and formatting finance-related MCP tool usage."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

RESULTS_TOOL_SUFFIX = "get_recent_quarters_results"
IMPROVEMENT_TOOL_SUFFIX = "get_recent_quarters_improvement"

__all__ = [
    "detect_financial_tool_request",
    "extract_candidate_symbol",
    "format_improvement_summary",
    "format_results_summary",
    "IMPROVEMENT_TOOL_SUFFIX",
    "RESULTS_TOOL_SUFFIX",
]

_FINANCIAL_STOPWORDS = {
    "AND",
    "ASK",
    "CAGR",
    "COMPANY",
    "COMPARED",
    "DID",
    "DOES",
    "EARNINGS",
    "HOW",
    "IMPROVEMENT",
    "LAST",
    "MOST",
    "PLEASE",
    "PREVIOUS",
    "QUARTER",
    "RECENT",
    "RESULT",
    "RESULTS",
    "SHOW",
    "STOCK",
    "TELL",
    "WHAT",
    "WHEN",
    "WHERE",
    "WHICH",
    "WHO",
    "WHY",
}

_RESULT_KEY_PHRASES = (
    "last quarter results",
    "last quarter's results",
    "earnings last quarter",
    "results last quarter",
    "most recent quarter",
    "latest quarter results",
    "recent quarter earnings",
)

_RESULT_KEYWORDS = (
    "earnings",
    "results",
    "report",
    "performance",
    "financials",
    "quarterly",
)

_IMPROVEMENT_CONTEXT_PHRASES = (
    "last quarter",
    "previous quarter",
    "prior quarter",
)

_IMPROVEMENT_KEYWORDS = (
    "improv",
    "compared",
    "compare",
    "versus",
    "vs",
    "change",
    "difference",
    "growth",
    "decline",
    "increase",
    "decrease",
    "better",
    "worse",
)


def extract_candidate_symbol(question: str) -> str | None:
    """Return the first plausible ticker symbol mentioned in ``question``.

    The helper is intentionally conservative; it ignores common English words
    that may also appear as stock tickers to reduce false positives.
    """

    if not question:
        return None

    dollar_matches = re.findall(r"\$([A-Za-z]{1,5})\b", question)
    for match in dollar_matches:
        candidate = match.upper()
        if candidate and candidate not in _FINANCIAL_STOPWORDS:
            return candidate

    uppercase_matches = re.findall(r"\b[A-Z]{1,5}\b", question)
    for match in uppercase_matches:
        candidate = match.upper()
        if candidate and candidate not in _FINANCIAL_STOPWORDS:
            return candidate
    return None


def detect_financial_tool_request(question: str) -> tuple[str, str] | None:
    """Identify whether ``question`` should trigger a financial MCP tool.

    Returns:
        ``(tool_suffix, symbol)`` if a matching tool should be invoked, or
        ``None`` when the question should be answered directly by the LLM.
    """

    symbol = extract_candidate_symbol(question)
    if not symbol:
        return None

    lowered = question.lower()
    mentions_last_quarter = any(phrase in lowered for phrase in _IMPROVEMENT_CONTEXT_PHRASES) or (
        "most recent quarter" in lowered
    )
    mentions_results = any(keyword in lowered for keyword in _RESULT_KEYWORDS)
    mentions_stock_performance = "stock performance" in lowered or (
        "performance" in lowered and "stock" in lowered
    )

    if mentions_last_quarter and any(keyword in lowered for keyword in _IMPROVEMENT_KEYWORDS):
        return IMPROVEMENT_TOOL_SUFFIX, symbol

    if any(phrase in lowered for phrase in _RESULT_KEY_PHRASES):
        return RESULTS_TOOL_SUFFIX, symbol

    if mentions_last_quarter and (mentions_results or mentions_stock_performance):
        return RESULTS_TOOL_SUFFIX, symbol

    return None


def _format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if abs(numeric) >= 100:
        return f"{numeric:,.2f}"
    return f"{numeric:.2f}"


def _format_percent(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric:+.2f}%"


def format_results_summary(symbol: str, payload: Mapping[str, Any] | None) -> str:
    """Render a readable summary of quarterly results for ``symbol``."""

    if not isinstance(payload, Mapping):
        return f"Unexpected response format while fetching quarterly results for {symbol}."
    quarters = payload.get("quarters")
    if not isinstance(quarters, Sequence) or not quarters:
        return f"No quarterly results available for {symbol}."

    lines = [f"Most recent quarterly earnings for {symbol.upper()}:"]
    for entry in quarters:
        if not isinstance(entry, Mapping):
            continue
        fiscal = entry.get("fiscal_date") or "Unknown fiscal date"
        reported_eps = _format_number(entry.get("reported_eps"))
        estimate = entry.get("estimated_eps")
        estimate_str = _format_number(estimate) if estimate not in (None, "") else "n/a"
        surprise = entry.get("surprise")
        surprise_pct = entry.get("surprise_percentage")
        details = [f"reported EPS {reported_eps}"]
        if estimate_str != "n/a":
            details.append(f"estimate {estimate_str}")
        if surprise not in (None, ""):
            surprise_value = _format_number(surprise)
            surprise_pct_value = _format_percent(surprise_pct)
            details.append(f"surprise {surprise_value} ({surprise_pct_value})")
        report_time = entry.get("report_time")
        suffix = f" [{report_time}]" if report_time else ""
        lines.append(f"- {fiscal}: {', '.join(details)}{suffix}")
    return "\n".join(lines)


def format_improvement_summary(symbol: str, payload: Mapping[str, Any] | None) -> str:
    """Render quarter-over-quarter EPS change details for ``symbol``."""

    if not isinstance(payload, Mapping):
        return (
            f"Unexpected response format while evaluating quarterly improvements for {symbol}."
        )
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, Sequence) or not comparisons:
        return f"No quarterly comparison data available for {symbol}."

    lines = [f"Quarter-over-quarter EPS change for {symbol.upper()}:"]
    for entry in comparisons:
        if not isinstance(entry, Mapping):
            continue
        current_date = entry.get("fiscal_date") or "current quarter"
        previous_date = entry.get("previous_fiscal_date") or "previous quarter"
        previous_eps = _format_number(entry.get("previous_reported_eps"))
        current_eps = _format_number(entry.get("reported_eps"))
        change_pct = _format_percent(entry.get("change_percentage"))
        absolute_change = entry.get("absolute_change")
        absolute_change_str = (
            _format_number(absolute_change) if absolute_change not in (None, "") else "n/a"
        )
        lines.append(
            "- "
            f"{previous_date} -> {current_date}: {change_pct} (EPS {previous_eps} -> {current_eps}, "
            f"Δ {absolute_change_str})"
        )
    return "\n".join(lines)

