"""Minimal fallback implementation of the LangGraph API used in tests.

This stub mirrors just enough of :mod:`langgraph.graph` for our unit tests. It
supports registering synchronous or asynchronous node callables, sequential
edges, and conditional routing. The real LangGraph library exposes a richer set
of features; this fallback merely executes nodes in a simple loop so that the
host routing agent remains testable when the dependency cannot be installed
(e.g. in offline CI environments).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


# The compiled LangGraph uses a sentinel named ``END`` to indicate that the
# workflow has completed. We model that with a unique object to avoid string
# collisions.
class _EndSentinel:
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "<LangGraph END>"


END = _EndSentinel()


class _CompiledGraph:
    """Execute nodes registered on a :class:`StateGraph`."""

    def __init__(
        self,
        nodes: dict[str, Callable[[dict[str, Any]], Any]],
        edges: dict[str, str | _EndSentinel],
        conditionals: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]],
        entry_point: str,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditionals = conditionals
        self._entry_point = entry_point

    async def ainvoke(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Run the graph starting from ``initial_state``."""

        state = dict(initial_state)
        current = self._entry_point

        while current is not END:
            node = self._nodes[current]
            result = node(state)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            if result is not None:
                state = result

            if current in self._conditionals:
                router, mapping = self._conditionals[current]
                route = router(state)
                current = mapping.get(route, END)
                continue

            next_node = self._edges.get(current, END)
            current = next_node

        return state


class StateGraph:
    """Lightweight container matching the subset of LangGraph that we use."""

    def __init__(self, _state_type: type[Any]) -> None:
        self._nodes: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._edges: dict[str, str | _EndSentinel] = {}
        self._conditionals: dict[
            str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]
        ] = {}
        self._entry_point: str | None = None

    def add_node(self, name: str, func: Callable[[dict[str, Any]], Any]) -> None:
        self._nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        self._entry_point = name

    def add_edge(self, start: str, end: str | _EndSentinel) -> None:
        self._edges[start] = end

    def add_conditional_edges(
        self,
        start: str,
        router: Callable[[dict[str, Any]], str],
        mapping: dict[str, str],
    ) -> None:
        self._conditionals[start] = (router, mapping)

    def compile(self) -> _CompiledGraph:
        if self._entry_point is None:
            raise RuntimeError("StateGraph requires an entry point before compile().")
        return _CompiledGraph(self._nodes, self._edges, self._conditionals, self._entry_point)


__all__ = ["END", "StateGraph"]
