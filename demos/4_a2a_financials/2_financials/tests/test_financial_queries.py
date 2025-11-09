"""Unit tests for finance-specific helper functions."""
from __future__ import annotations

from src.agent.financial_queries import (
    IMPROVEMENT_TOOL_SUFFIX,
    RESULTS_TOOL_SUFFIX,
    detect_financial_tool_request,
    format_improvement_summary,
    format_results_summary,
)


def test_detects_results_for_stock_performance_question() -> None:
    question = "How was Apple's (ticker: AAPL) stock performance in the last quarter?"
    detection = detect_financial_tool_request(question)
    assert detection == (RESULTS_TOOL_SUFFIX, "AAPL")


def test_detects_improvement_question() -> None:
    question = "Did MSFT improve versus the previous quarter?"
    detection = detect_financial_tool_request(question)
    assert detection == (IMPROVEMENT_TOOL_SUFFIX, "MSFT")


def test_formats_results_summary_with_single_entry() -> None:
    payload = {
        "quarters": [
            {
                "fiscal_date": "2024-06-30",
                "reported_eps": 1.5,
                "estimated_eps": 1.4,
                "surprise": 0.1,
                "surprise_percentage": 7.14,
                "report_time": "amc",
            }
        ]
    }

    summary = format_results_summary("AAPL", payload)
    assert "Most recent quarterly earnings for AAPL" in summary
    assert "reported EPS 1.50" in summary


def test_formats_improvement_summary() -> None:
    payload = {
        "comparisons": [
            {
                "fiscal_date": "2024-06-30",
                "reported_eps": 1.5,
                "previous_fiscal_date": "2024-03-31",
                "previous_reported_eps": 1.3,
                "absolute_change": 0.2,
                "change_percentage": 15.38,
            }
        ]
    }

    summary = format_improvement_summary("AAPL", payload)
    assert "Quarter-over-quarter EPS change for AAPL" in summary
    assert "15.38%" in summary
