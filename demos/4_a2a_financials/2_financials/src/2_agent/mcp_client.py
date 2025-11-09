"""Lightweight MCP client for integrating external tool servers."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency hint
    import modelcontextprotocol  # type: ignore  # noqa: F401
except Exception:  # noqa: BLE001 - optional runtime dependency
    modelcontextprotocol = None

LOGGER = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    """Base error for MCP client failures."""


class MCPConnectionError(MCPClientError):
    """Raised when a connection to an MCP server fails."""


class MCPInvocationError(MCPClientError):
    """Raised when invoking an MCP tool fails."""


def _json_dumps(data: Mapping[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")


def _json_loads(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def _parse_targets(raw_targets: str | Sequence[str]) -> List[Tuple[str, str, str]]:
    """Parse target strings into (scheme, value, label)."""

    if isinstance(raw_targets, str):
        parts = [p.strip() for p in raw_targets.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in raw_targets if p.strip()]

    parsed: List[Tuple[str, str, str]] = []
    for idx, part in enumerate(parts):
        if ":" not in part:
            raise ValueError(f"Invalid MCP target format: {part}")
        scheme, value = part.split(":", 1)
        label = f"{scheme}-{idx}"
        parsed.append((scheme.lower(), value.strip(), label))
    return parsed


def _normalise_tool(tool: Mapping[str, Any], *, prefix: str | None = None) -> Tuple[str, Dict[str, Any]]:
    name = str(tool.get("name"))
    if prefix:
        full_name = f"{prefix}.{name}" if not name.startswith(f"{prefix}.") else name
    else:
        full_name = name

    schema = {
        "type": "function",
        "function": {
            "name": full_name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        },
    }
    return full_name, schema


@dataclass
class _ToolRegistration:
    tool_name: str
    session_alias: str
    raw_tool_name: str


class _BaseSession:
    """Base class for protocol sessions."""

    def __init__(self, alias: str, *, connect_timeout: float, invocation_timeout: float) -> None:
        self.alias = alias
        self.connect_timeout = connect_timeout
        self.invocation_timeout = invocation_timeout

    async def start(self) -> Dict[str, Any]:  # pragma: no cover - abstract method
        raise NotImplementedError

    async def invoke(self, tool: str, arguments: Mapping[str, Any]) -> Any:  # pragma: no cover - abstract method
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - optional override
        return None


class _StdioSession(_BaseSession):
    """Session using a stdio-based MCP server."""

    def __init__(self, alias: str, command: str, *, connect_timeout: float, invocation_timeout: float) -> None:
        super().__init__(alias, connect_timeout=connect_timeout, invocation_timeout=invocation_timeout)
        self.command = command
        self.process: Optional[asyncio.subprocess.Process] = None
        self._stdout_lock = asyncio.Lock()
        self._stdin_lock = asyncio.Lock()
        self._handshake: Dict[str, Any] | None = None

    async def start(self) -> Dict[str, Any]:
        args = shlex.split(self.command)
        if not args:
            raise MCPConnectionError(f"Empty command for MCP target {self.alias}")

        LOGGER.info("Starting MCP stdio target '%s': %s", self.alias, args)
        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert self.process.stdout is not None  # for type-checkers
        try:
            raw = await asyncio.wait_for(self.process.stdout.readline(), timeout=self.connect_timeout)
        except asyncio.TimeoutError as exc:  # pragma: no cover - defensive
            raise MCPConnectionError(f"Timed out waiting for handshake from {self.alias}") from exc

        if not raw:
            raise MCPConnectionError(f"MCP target {self.alias} terminated before handshake")

        try:
            handshake = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise MCPConnectionError(f"Invalid handshake from {self.alias}: {raw!r}") from exc

        self._handshake = handshake
        LOGGER.info(
            "MCP target '%s' ready (session=%s, tools=%s)",
            self.alias,
            handshake.get("session_id"),
            [tool.get("name") for tool in handshake.get("tools", [])],
        )
        return handshake

    async def invoke(self, tool: str, arguments: Mapping[str, Any]) -> Any:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise MCPInvocationError("Session is not connected")

        request_id = str(uuid.uuid4())
        payload = {
            "type": "invoke",
            "id": request_id,
            "tool": tool,
            "arguments": arguments,
        }

        async with self._stdin_lock:
            self.process.stdin.write(_json_dumps(payload))
            await self.process.stdin.drain()

        async with self._stdout_lock:
            deadline = time.time() + self.invocation_timeout
            while True:
                timeout = max(0.1, deadline - time.time())
                try:
                    raw = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise MCPInvocationError(f"Timeout waiting for result from {self.alias}") from exc

                if not raw:
                    raise MCPInvocationError(f"MCP target {self.alias} closed stdout unexpectedly")

                try:
                    message = _json_loads(raw)
                except json.JSONDecodeError:
                    LOGGER.warning("Ignoring non-JSON MCP output from %s: %r", self.alias, raw)
                    continue

                if message.get("type") == "log":
                    LOGGER.info("[MCP %s] %s", self.alias, message.get("message"))
                    continue

                if message.get("type") == "result" and message.get("id") == request_id:
                    if message.get("ok", True):
                        return message.get("content")
                    error = message.get("error") or {"message": "unknown error"}
                    raise MCPInvocationError(
                        f"Tool {tool} failed on {self.alias}: {error.get('message')}"
                    )

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            try:
                self.process.stdin.write(_json_dumps({"type": "shutdown"}))
                await self.process.stdin.drain()
            except Exception:  # noqa: BLE001 - best effort
                pass
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2)
            except asyncio.TimeoutError:  # pragma: no cover - defensive
                self.process.kill()
        self.process = None


class _SSESession(_BaseSession):
    """Session backed by a simple HTTP(S) server using SSE for results."""

    def __init__(self, alias: str, endpoint: str, *, connect_timeout: float, invocation_timeout: float) -> None:
        super().__init__(alias, connect_timeout=connect_timeout, invocation_timeout=invocation_timeout)
        from urllib.parse import urlparse

        parsed = urlparse(endpoint.rstrip("/"))
        if parsed.scheme not in {"http", "https"}:
            raise MCPConnectionError(f"Unsupported SSE scheme for target {endpoint}")
        if parsed.scheme == "https":  # pragma: no cover - environment limitation
            raise MCPConnectionError("HTTPS SSE targets are not supported in this environment")

        self.endpoint = endpoint.rstrip("/")
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._path = parsed.path or "/"
        if parsed.query:
            self._path += f"?{parsed.query}"
        self._base_url = f"http://{self._host}:{self._port}"

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._event_queue: asyncio.Queue[Dict[str, Any]] | None = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._session_id: Optional[str] = None
        self._invoke_url: Optional[str] = None

    async def _open_stream(self) -> None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self._host, self._port), timeout=self.connect_timeout
        )
        request = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            "Accept: text/event-stream\r\n"
            "Connection: keep-alive\r\n\r\n"
        )
        writer.write(request.encode("utf-8"))
        await writer.drain()

        header_block = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=self.connect_timeout)
        status_line = header_block.split(b"\r\n", 1)[0]
        parts = status_line.split()
        if len(parts) < 2 or parts[1] != b"200":
            raise MCPConnectionError(f"Unexpected SSE response status: {status_line.decode('utf-8', 'ignore')}")

        self._reader = reader
        self._writer = writer

    async def _read_event(self, timeout: Optional[float]) -> Optional[str]:
        if not self._reader:
            return None

        data_parts: List[str] = []
        while True:
            try:
                raw_line = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                raise
            if raw_line == b"":
                return None
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_parts:
                    return "\n".join(data_parts)
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_parts.append(line[5:].lstrip())

    async def start(self) -> Dict[str, Any]:
        from urllib.parse import urljoin

        await self._open_stream()
        LOGGER.info("Connecting to MCP SSE endpoint '%s'", self.endpoint)

        raw_event = await self._read_event(self.connect_timeout)
        if raw_event is None:
            raise MCPConnectionError("Failed to receive handshake from SSE server")
        try:
            handshake = json.loads(raw_event)
        except json.JSONDecodeError as exc:
            raise MCPConnectionError(f"Invalid handshake payload: {raw_event}") from exc

        if handshake.get("type") != "ready":
            raise MCPConnectionError(f"Unexpected handshake message: {handshake}")

        self._session_id = handshake.get("session_id")
        self._invoke_url = handshake.get("invoke_url")
        if self._invoke_url:
            if not self._invoke_url.startswith("http"):
                self._invoke_url = urljoin(self.endpoint + "/", self._invoke_url)
        else:
            self._invoke_url = f"{self.endpoint}/invoke"

        self._event_queue = asyncio.Queue()

        async def _reader_loop() -> None:
            assert self._event_queue is not None
            try:
                while True:
                    data = await self._read_event(None)
                    if data is None:
                        break
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        message = json.loads(data)
                    except json.JSONDecodeError:
                        LOGGER.warning("Ignoring invalid SSE payload from %s: %s", self.alias, data)
                        continue
                    await self._event_queue.put(message)
            except asyncio.CancelledError:  # pragma: no cover - cleanup
                raise

        self._reader_task = asyncio.create_task(_reader_loop())

        LOGGER.info(
            "MCP SSE target '%s' ready (session=%s)", self.alias, self._session_id
        )
        return handshake

    async def _post_json(self, url: str, payload: Mapping[str, Any]) -> None:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme != "http":
            raise MCPInvocationError("Only HTTP POST endpoints are supported for SSE tools")

        host = parsed.hostname or self._host
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("utf-8") + body

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=self.invocation_timeout
        )
        writer.write(request)
        await writer.drain()

        header_block = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=self.invocation_timeout)
        status_line = header_block.split(b"\r\n", 1)[0]
        parts = status_line.split()
        status = int(parts[1]) if len(parts) > 1 else 0

        content_length = 0
        for line in header_block.decode("utf-8").split("\r\n")[1:]:
            if not line:
                continue
            name, _, value = line.partition(":")
            if name.lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError:  # pragma: no cover - defensive
                    content_length = 0
                break

        if content_length:
            await asyncio.wait_for(reader.readexactly(content_length), timeout=self.invocation_timeout)

        writer.close()
        await writer.wait_closed()

        if status >= 400:
            raise MCPInvocationError(f"SSE invocation returned HTTP {status}")

    async def invoke(self, tool: str, arguments: Mapping[str, Any]) -> Any:
        if not self._event_queue or not self._invoke_url:
            raise MCPInvocationError("SSE session not connected")

        request_id = str(uuid.uuid4())
        payload = {
            "id": request_id,
            "session_id": self._session_id,
            "tool": tool,
            "arguments": arguments,
        }
        await self._post_json(self._invoke_url, payload)

        deadline = time.time() + self.invocation_timeout
        while True:
            timeout = max(0.1, deadline - time.time())
            try:
                message = await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise MCPInvocationError("Timeout waiting for SSE tool response") from exc

            if message.get("type") == "log":
                LOGGER.info("[MCP %s] %s", self.alias, message.get("message"))
                continue

            if message.get("type") == "result" and message.get("id") == request_id:
                if message.get("ok", True):
                    return message.get("content")
                error = message.get("error") or {"message": "unknown error"}
                raise MCPInvocationError(
                    f"Tool {tool} failed on {self.alias}: {error.get('message')}"
                )

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(Exception):  # pragma: no cover - cleanup
                await self._reader_task
        if self._writer:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        self._reader_task = None
        self._reader = None
        self._writer = None
        self._event_queue = None


class MCPClient:
    """Coordinator that manages tool discovery and invocation across MCP sessions."""

    def __init__(
        self,
        *,
        connect_timeout: float = 5.0,
        invocation_timeout: float = 15.0,
    ) -> None:
        self.connect_timeout = connect_timeout
        self.invocation_timeout = invocation_timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-client-loop")
        self._thread.start()
        self._sessions: Dict[str, _BaseSession] = {}
        self._tool_registry: Dict[str, _ToolRegistration] = {}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        async def _shutdown() -> None:
            for session in list(self._sessions.values()):
                try:
                    await session.close()
                except Exception:  # noqa: BLE001 - best effort
                    LOGGER.exception("Failed to close MCP session %s", session.alias)

        future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        try:
            future.result(timeout=5)
        except Exception:  # noqa: BLE001 - best effort
            LOGGER.exception("Error while shutting down MCP client")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1)
            self._loop.close()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover(self, targets: str | Sequence[str]) -> None:
        parsed_targets = _parse_targets(targets)

        async def _connect() -> None:
            for scheme, value, label in parsed_targets:
                if scheme == "stdio":
                    session: _BaseSession = _StdioSession(
                        label,
                        value,
                        connect_timeout=self.connect_timeout,
                        invocation_timeout=self.invocation_timeout,
                    )
                elif scheme in {"sse", "http", "https"}:
                    endpoint = value if scheme == "sse" else f"{scheme}://{value}"
                    session = _SSESession(
                        label,
                        endpoint,
                        connect_timeout=self.connect_timeout,
                        invocation_timeout=self.invocation_timeout,
                    )
                else:
                    LOGGER.warning("Skipping unsupported MCP target scheme '%s'", scheme)
                    continue

                try:
                    handshake = await session.start()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Failed to connect to MCP target %s", label)
                    continue

                prefix = handshake.get("alias") or label
                for tool in handshake.get("tools", []):
                    unique_name, schema = _normalise_tool(tool, prefix=prefix)
                    if unique_name in self._tool_registry:
                        LOGGER.warning(
                            "Tool name collision for '%s'; skipping registration", unique_name
                        )
                        continue
                    self._tool_registry[unique_name] = _ToolRegistration(
                        tool_name=unique_name,
                        session_alias=label,
                        raw_tool_name=tool.get("name", unique_name),
                    )
                    self._tool_schemas[unique_name] = schema

                self._sessions[label] = session

        future = asyncio.run_coroutine_threadsafe(_connect(), self._loop)
        future.result()
        LOGGER.info("MCP discovery complete. Registered %s tools.", len(self._tool_registry))

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------
    def invoke(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        if tool_name not in self._tool_registry:
            raise MCPInvocationError(f"Unknown tool: {tool_name}")
        registration = self._tool_registry[tool_name]
        session = self._sessions.get(registration.session_alias)
        if not session:
            raise MCPInvocationError(f"Session '{registration.session_alias}' not found for tool {tool_name}")

        async def _invoke() -> Any:
            return await session.invoke(registration.raw_tool_name, arguments)

        start = time.perf_counter()
        LOGGER.info(
            "Dispatching MCP tool '%s' on session '%s' with arguments: %s",
            tool_name,
            registration.session_alias,
            json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, Mapping) else arguments,
        )
        future = asyncio.run_coroutine_threadsafe(_invoke(), self._loop)
        try:
            result = future.result(timeout=self.invocation_timeout + self.connect_timeout)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool invocation failed for %s", tool_name)
            raise MCPInvocationError(str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - start
            LOGGER.info("Tool %s completed in %.3fs", tool_name, elapsed)
        return result

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    @property
    def tools(self) -> List[Dict[str, Any]]:
        return list(self._tool_schemas.values())

    @property
    def tool_names(self) -> List[str]:
        return list(self._tool_registry.keys())

    def has_tools(self) -> bool:
        return bool(self._tool_registry)


__all__ = ["MCPClient", "MCPClientError", "MCPInvocationError", "MCPConnectionError"]

