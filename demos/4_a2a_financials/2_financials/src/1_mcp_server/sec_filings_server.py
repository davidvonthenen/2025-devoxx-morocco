"""MCP server exposing SEC filing and earnings data via Finnhub and Alpha Vantage."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import queue
import sys
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Mapping

import requests


LOGGER = logging.getLogger("sec-filings-mcp-server")

FINNHUB_API_BASE = "https://finnhub.io/api/v1"
_DEFAULT_EARNINGS_LOOKAHEAD_DAYS = 30
_DEFAULT_QUARTERS_LIMIT = 4

TOOLS = [
    {
        "name": "get_earnings_calendar",
        "description": (
            "Fetch upcoming earnings calendar entries for a symbol using Finnhub. "
            "Dates default to the next 30 days if omitted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol for the company (for example: AAPL).",
                },
                "from_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Defaults to today.",
                },
                "to_date": {
                    "type": "string",
                    "description": (
                        "End date (YYYY-MM-DD). Defaults to 30 days after from_date."
                    ),
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_recent_quarters_results",
        "description": (
            "Return the most recent quarterly earnings, including reported and estimated EPS, "
            "using Alpha Vantage's EARNINGS endpoint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol for the company (for example: MSFT).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of quarters to return (1-8). Defaults to 4.",
                    "minimum": 1,
                    "maximum": 8,
                    "default": _DEFAULT_QUARTERS_LIMIT,
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_recent_quarters_improvement",
        "description": (
            "Calculate quarter-over-quarter EPS changes based on Alpha Vantage earnings data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol for the company (for example: TSLA).",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Number of recent quarters to analyse (2-8). Defaults to 4 quarters."
                    ),
                    "minimum": 2,
                    "maximum": 8,
                    "default": _DEFAULT_QUARTERS_LIMIT,
                },
            },
            "required": ["symbol"],
        },
    },
]


class ServerError(RuntimeError):
    """Raised when the MCP server cannot satisfy a tool request."""


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ServerError(
            f"Environment variable {name} must be set to query external market data APIs."
        )
    return value.strip()


def _parse_date(value: str | None, *, default: dt.date | None = None) -> dt.date:
    if value:
        try:
            return dt.datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ServerError(f"Invalid date format '{value}'. Expected YYYY-MM-DD.") from exc
    if default is None:
        return dt.date.today()
    return default


def _request_json(url: str, params: Mapping[str, object]) -> Dict[str, object]:
    try:
        LOGGER.debug("Requesting %s with params=%s", url, params)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise ServerError(f"Network error while contacting {url}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ServerError(f"Failed to decode JSON payload from {url}") from exc

    return payload


def _fetch_finnhub_calendar(symbol: str, from_date: dt.date, to_date: dt.date) -> Dict[str, object]:
    token = _require_env("FINNHUB_API_KEY")
    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": token,
    }
    payload = _request_json(f"{FINNHUB_API_BASE}/calendar/earnings", params)
    entries: List[Dict[str, object]] = []
    for item in payload.get("earningsCalendar", []) or []:
        entry = {
            "symbol": item.get("symbol"),
            "date": item.get("date"),
            "eps_actual": item.get("epsActual"),
            "eps_estimate": item.get("epsEstimate"),
            "revenue_actual": item.get("revenueActual"),
            "revenue_estimate": item.get("revenueEstimate"),
            "updated_time": item.get("hour"),
        }
        entries.append(entry)

    return {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "earnings": entries,
    }


def _safe_float(value: object | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_finnhub_company_earnings(symbol: str, *, limit: int | None = None) -> list[dict]:
    """
    Finnhub 'company earnings surprises' for a symbol (most-recent first).
    Docs: https://finnhub.io/docs/api/company-earnings  (REST: /stock/earnings?symbol=...)
    """
    token = _require_env("FINNHUB_API_KEY")
    params: dict[str, object] = {"symbol": symbol.upper(), "token": token}
    if isinstance(limit, int) and limit > 0:
        # Finnhub supports a 'limit' query param to cap returned periods.
        params["limit"] = limit  # leave unset to get full history
    payload = _request_json(f"{FINNHUB_API_BASE}/stock/earnings", params)

    _check_finnhub_errors(payload)

    if not isinstance(payload, list):
        raise ServerError("Unexpected Finnhub payload: expected a list for /stock/earnings.")
    if not payload:
        raise ServerError(f"No quarterly earnings data available for {symbol.upper()}.")

    # Deterministic ordering (Finnhub is typically newest-first, but sort defensively).
    def _parse_period(v: str | None):
        from datetime import datetime, date
        try:
            return datetime.strptime(v or "", "%Y-%m-%d").date()
        except Exception:
            return date.min

    payload.sort(key=lambda e: _parse_period(e.get("period")), reverse=True)
    return payload


def _check_finnhub_errors(payload: object) -> None:
    # Finnhub errors often appear as {"error": "..."} on 200 responses,
    # and proper HTTP error codes for 401/403/429. Let _request_json handle HTTP.
    if isinstance(payload, dict) and "error" in payload:
        raise ServerError(f"Finnhub error: {payload.get('error')}")

def _get_recent_quarters_results(symbol: str, *, limit: int = 4) -> dict[str, object]:
    """
    Return most recent quarterly earnings information for `symbol` using Finnhub.
    Maintains the AlphaVantage-shaped output keys for downstream compatibility.
    - fiscal_date            <- Finnhub 'period' (YYYY-MM-DD)
    - reported_date          <- None (not available from free company-earnings endpoint)
    - reported_eps           <- Finnhub 'actual'
    - estimated_eps          <- Finnhub 'estimate'
    - surprise               <- actual - estimate (computed)
    - surprise_percentage    <- (actual - estimate)/abs(estimate)*100 (computed)
    - report_time            <- None (not available from this endpoint)
    """
    if limit < 1:
        raise ServerError("limit must be >= 1")

    earnings = _fetch_finnhub_company_earnings(symbol, limit=limit)

    results: list[dict[str, object]] = []
    for entry in earnings[:limit]:
        actual = _safe_float(entry.get("actual"))
        estimate = _safe_float(entry.get("estimate"))

        surprise = None
        surprise_pct = None
        if actual is not None and estimate is not None:
            surprise = actual - estimate
            if estimate != 0:
                surprise_pct = (surprise / abs(estimate)) * 100

        results.append(
            {
                "fiscal_date": entry.get("period"),
                "reported_date": None,         # Not available from this free endpoint
                "reported_eps": actual,
                "estimated_eps": estimate,
                "surprise": surprise,
                "surprise_percentage": surprise_pct,
                "report_time": None,           # Not available from this free endpoint
            }
        )

    return {
        "symbol": symbol.upper(),
        "quarters": results,
    }


def _get_recent_quarters_improvement(symbol: str, *, limit: int = 4) -> dict[str, object]:
    """
    Quarter-over-quarter EPS changes using Finnhub 'actual' EPS.
    Returns `limit-1` comparison rows, newest vs. previous.
    """
    if limit < 2:
        raise ServerError("limit must be >= 2 to compute quarter-over-quarter changes")

    earnings = _fetch_finnhub_company_earnings(symbol, limit=limit)
    if len(earnings) < 2:
        raise ServerError(f"Insufficient quarterly data available for {symbol.upper()}.")

    comparisons: list[dict[str, object]] = []
    max_index = min(limit, len(earnings))
    for idx in range(max_index - 1):
        current = earnings[idx]
        previous = earnings[idx + 1]
        current_eps = _safe_float(current.get("actual"))
        previous_eps = _safe_float(previous.get("actual"))

        absolute_change = None
        change_percentage = None
        if current_eps is not None and previous_eps is not None:
            absolute_change = current_eps - previous_eps
            if previous_eps != 0:
                change_percentage = (absolute_change / abs(previous_eps)) * 100

        comparisons.append(
            {
                "fiscal_date": current.get("period"),
                "reported_eps": current_eps,
                "previous_fiscal_date": previous.get("period"),
                "previous_reported_eps": previous_eps,
                "absolute_change": absolute_change,
                "change_percentage": change_percentage,
            }
        )

    return {
        "symbol": symbol.upper(),
        "comparisons": comparisons,
    }


def _build_handshake(session_id: str, *, include_url: bool = False) -> Dict[str, object]:
    handshake: Dict[str, object] = {
        "type": "ready",
        "session_id": session_id,
        "alias": "sec-filings",
        "tools": TOOLS,
    }
    if include_url:
        handshake["invoke_url"] = "/mcp/invoke"
    return handshake


def _handle_invoke(tool: str, arguments: Mapping[str, object]) -> Dict[str, object]:
    try:
        if tool == "get_earnings_calendar":
            symbol = str(arguments.get("symbol", "") or arguments.get("ticker", "")).strip()
            if not symbol:
                raise ServerError("'symbol' is required")
            from_value = arguments.get("from_date")
            to_value = arguments.get("to_date")
            from_date = _parse_date(str(from_value)) if from_value else dt.date.today()
            default_to = from_date + dt.timedelta(days=_DEFAULT_EARNINGS_LOOKAHEAD_DAYS)
            to_date = _parse_date(str(to_value), default=default_to) if to_value else default_to
            result = _fetch_finnhub_calendar(symbol, from_date, to_date)

        elif tool == "get_recent_quarters_results":
            symbol = str(arguments.get("symbol", "") or arguments.get("ticker", "")).strip()
            if not symbol:
                raise ServerError("'symbol' is required")
            limit = arguments.get("limit", _DEFAULT_QUARTERS_LIMIT)
            try:
                limit_int = int(limit)
            except (TypeError, ValueError) as exc:
                raise ServerError("'limit' must be an integer") from exc
            limit_int = max(1, min(limit_int, 8))
            result = _get_recent_quarters_results(symbol, limit=limit_int)

        elif tool == "get_recent_quarters_improvement":
            symbol = str(arguments.get("symbol", "") or arguments.get("ticker", "")).strip()
            if not symbol:
                raise ServerError("'symbol' is required")
            limit = arguments.get("limit", _DEFAULT_QUARTERS_LIMIT)
            try:
                limit_int = int(limit)
            except (TypeError, ValueError) as exc:
                raise ServerError("'limit' must be an integer") from exc
            limit_int = max(2, min(limit_int, 8))
            result = _get_recent_quarters_improvement(symbol, limit=limit_int)

        else:
            LOGGER.warning("Unknown tool requested: %s", tool)
            return {
                "ok": False,
                "error": {
                    "type": "UnknownTool",
                    "message": f"Tool '{tool}' is not available.",
                },
            }
    except ServerError as exc:
        LOGGER.exception("Tool '%s' failed: %s", tool, exc)
        return {
            "ok": False,
            "error": {
                "type": "ToolExecutionError",
                "message": str(exc),
            },
        }

    return {"ok": True, "content": result}


def run_stdio_server() -> None:
    session_id = str(uuid.uuid4())
    handshake = _build_handshake(session_id)
    print(json.dumps(handshake), flush=True)
    LOGGER.info("SEC filings MCP stdio server ready (session=%s)", session_id)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError:
            LOGGER.warning("Received non-JSON payload: %s", raw_line)
            continue

        if message.get("type") == "shutdown":
            LOGGER.info("Shutdown requested; exiting.")
            break

        if message.get("type") != "invoke":
            LOGGER.warning("Unsupported message type: %s", message.get("type"))
            continue

        call_id = message.get("id", str(uuid.uuid4()))
        tool = message.get("tool", "")
        arguments = message.get("arguments", {}) if isinstance(message, dict) else {}
        LOGGER.info(
            "STDIO invocation received (call_id=%s, tool=%s, arguments=%s)",
            call_id,
            tool,
            arguments,
        )
        result = _handle_invoke(tool, arguments if isinstance(arguments, Mapping) else {})
        response = {
            "type": "result",
            "id": call_id,
            **result,
        }
        LOGGER.info("STDIO response (call_id=%s): %s", call_id, response)
        print(json.dumps(response), flush=True)


@dataclass
class _ClientInfo:
    session_id: str
    queue: "queue.Queue[Dict[str, object]]"


class _MCPHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.clients: Dict[str, _ClientInfo] = {}
        self._lock = threading.Lock()

    def register_client(self, session_id: str) -> queue.Queue:
        q: "queue.Queue[Dict[str, object]]" = queue.Queue()
        with self._lock:
            self.clients[session_id] = _ClientInfo(session_id, q)
        return q

    def push_event(self, session_id: str, payload: Dict[str, object]) -> bool:
        with self._lock:
            info = self.clients.get(session_id)
        if not info:
            return False
        info.queue.put(payload)
        return True

    def remove_client(self, session_id: str) -> None:
        with self._lock:
            self.clients.pop(session_id, None)


class MCPRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - inherited signature
        LOGGER.info("%s - %s", self.client_address[0], format % args)

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - HTTP verb name
        if self.path != "/mcp":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        session_id = str(uuid.uuid4())
        q = self.server.register_client(session_id)  # type: ignore[attr-defined]

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        handshake = _build_handshake(session_id, include_url=True)
        payload = json.dumps(handshake)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()
        LOGGER.info("SSE client connected (session=%s)", session_id)

        try:
            while True:
                try:
                    message = q.get(timeout=1)
                except queue.Empty:
                    continue
                if message is None:
                    break
                self.wfile.write(("data: " + json.dumps(message) + "\n\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("SSE client disconnected (session=%s)", session_id)
        finally:
            self.server.remove_client(session_id)  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802 - HTTP verb name
        if self.path != "/mcp/invoke":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            return

        session_id = payload.get("session_id")
        if not isinstance(session_id, str):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing session_id"})
            return

        call_id = payload.get("id", str(uuid.uuid4()))
        tool = payload.get("tool", "")
        arguments = payload.get("arguments", {}) if isinstance(payload.get("arguments"), Mapping) else {}
        LOGGER.info(
            "SSE invocation received (session=%s, call_id=%s, tool=%s, arguments=%s)",
            session_id,
            call_id,
            tool,
            arguments,
        )
        result = _handle_invoke(tool, arguments)
        event = {
            "type": "result",
            "id": call_id,
            **result,
        }
        pushed = self.server.push_event(session_id, event)  # type: ignore[attr-defined]
        if not pushed:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown session"})
            return
        LOGGER.info("SSE response dispatched (session=%s, call_id=%s)", session_id, call_id)
        self._send_json(HTTPStatus.OK, {"status": "ok"})


def run_sse_server(port: int) -> None:
    server = _MCPHTTPServer(("0.0.0.0", port), MCPRequestHandler)
    LOGGER.info("SEC filings MCP SSE server listening on port %s", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        LOGGER.info("SSE server interrupted; shutting down")
    finally:
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server for SEC filings and earnings data")
    parser.add_argument("--stdio", action="store_true", help="Run using stdio transport")
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run using SSE/HTTP transport (default)",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for SSE mode")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.stdio and args.sse:
        parser.error("Cannot enable both stdio and SSE modes")

    if args.stdio:
        run_stdio_server()
    else:
        run_sse_server(args.port)


if __name__ == "__main__":
    main()
