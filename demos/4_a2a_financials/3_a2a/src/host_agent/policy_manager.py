"""Policy utilities used by the host agent LangGraph (News/Financial routing)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyClassification:
    """
    Structured result describing how the host should react to a message.

    route: "news" or "finance"
    tickers: extracted stock symbols (empty if none found)
    is_financial: True if the utterance expresses financial intent
    note: short human-readable policy note for auditability
    """
    route: str
    tickers: list[str]
    is_financial: bool
    note: Optional[str]


class NewsFinancePolicyManager:
    """
    Policy:
      * If the question is financial AND includes at least one recognizable stock
        ticker, route to the Financial Agent.
      * Otherwise, route to the News Agent.
    """

    FINANCE_KEYWORDS = (
        "earnings",
        "revenue",
        "guidance",
        "eps",
        "dividend",
        "split",
        "stock",
        "share price",
        "price target",
        "valuation",
        "market cap",
        "cash flow",
        "balance sheet",
        "income statement",
        "analyst",
        "buyback",
        "quarter",
        "q1", "q2", "q3", "q4",
        "10-k", "10q", "10-q",
        "sec filing",
        "financial results",
        "results",
        "outlook",
    )

    # Recognize ticker variants:
    #   $AAPL
    #   NASDAQ:GOOG, NYSE:IBM, TSX:SHOP, LSE:BP
    #   "ticker: MSFT", "ticker symbol MSFT"
    _PATTERNS = (
        re.compile(r"\$(?P<ticker>[A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b"),
        re.compile(
            r"\b(?:NASDAQ|NYSE|AMEX|ASX|TSX|LSE|NSE|BSE)\s*[:\-]\s*(?P<ticker>[A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:ticker(?:\s+symbol)?|symbol)\s*[:=]?\s*(?P<ticker>[A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\((?:\s*(?:ticker(?:\s+symbol)?|symbol)\s*[:=]?\s*(?P<ticker>[A-Z]{1,5}(?:\.[A-Z]{1,3})?)\s*)\)",
            re.IGNORECASE,
        ),
    )

    def classify_request(self, message: str) -> PolicyClassification:
        lowered = message.lower().strip()

        is_financial = any(k in lowered for k in self.FINANCE_KEYWORDS)
        tickers = self._extract_tickers(message)

        if is_financial and tickers:
            route = "finance"
            note = f"Policy: financial intent with ticker(s): {', '.join(tickers)}."
        else:
            route = "news"
            note = "Policy: route to News (non-financial or no ticker detected)."

        return PolicyClassification(
            route=route, tickers=tickers, is_financial=is_financial, note=note
        )

    def _extract_tickers(self, message: str) -> list[str]:
        seen: set[str] = set()
        for pattern in self._PATTERNS:
            for m in pattern.finditer(message):
                t = m.group("ticker").upper()
                if 1 <= len(t) <= 6:
                    seen.add(t)
        return sorted(seen)
