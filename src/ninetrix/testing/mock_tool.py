"""
ninetrix.testing.mock_tool — MockTool: fake tool for testing.

Layer: L9 (testing) — may import L8 (agent) and all below.

MockTool registers a fake tool in the ToolRegistry so the dispatcher
can find and call it during agent runs inside an AgentSandbox.

Usage::

    # Simple return value
    mock = MockTool("get_weather", return_value={"temp": 72, "unit": "F"})

    # Side effect callable
    mock = MockTool("get_weather", side_effect=lambda city: {"city": city, "temp": 72})

    # Raise an exception
    from ninetrix import ToolError
    mock = MockTool("get_weather", side_effect=ToolError("API down"))

    # Assertions
    mock.assert_called()
    mock.assert_called_with(city="London")
    mock.assert_call_count(1)
"""

from __future__ import annotations

import json
from typing import Any, Callable


class MockTool:
    """A fake tool implementation for testing.

    Registers itself as a synthetic :class:`~ninetrix.registry.ToolDef` so the
    dispatcher can route calls to it.  All calls are tracked for later assertion.

    Args:
        name:         Tool name (must match the name the LLM will call).
        return_value: Value returned on every call (serialised to JSON string
                      when needed).  Ignored if *side_effect* is set.
        side_effect:  Callable or exception instance/class.

                      - If callable: called with the same kwargs as the tool,
                        and its return value is used.
                      - If an exception instance or class: raised on every call.

        schema:       Optional JSON schema ``object`` dict.  When omitted a
                      minimal ``{"type": "object", "properties": {}}`` is used.
                      Pass a real schema to make the tool visible in dry-run
                      checks or provider payloads.

    Example::

        mock = MockTool("search", return_value=["result1", "result2"])
        mock.assert_called()            # raises AssertionError if never called
        mock.assert_called_with(q="py") # asserts last call kwargs
        mock.assert_call_count(3)       # asserts exactly 3 calls
    """

    def __init__(
        self,
        name: str,
        *,
        return_value: Any = None,
        side_effect: Any = None,
        schema: dict | None = None,
    ) -> None:
        self.name = name
        self._return_value = return_value
        self._side_effect = side_effect
        self._schema = schema or {"type": "object", "properties": {}}

        # Call tracking
        self._calls: list[dict[str, Any]] = []

        # Build and register the synthetic ToolDef
        self._tool_def = self._build_tool_def()

    # ------------------------------------------------------------------
    # Internal: build ToolDef and register
    # ------------------------------------------------------------------

    def _build_tool_def(self) -> Any:
        """Build a ToolDef backed by this mock and register it."""
        from ninetrix.registry import ToolDef, _registry

        mock_self = self

        async def _mock_fn(**kwargs: Any) -> Any:
            return await mock_self._dispatch(**kwargs)

        _mock_fn.__name__ = self.name
        _mock_fn.__doc__ = f"Mock tool: {self.name}"

        td = ToolDef(
            name=self.name,
            description=f"Mock tool: {self.name}",
            parameters=self._schema,
            fn=_mock_fn,
        )
        # Register — silently replace if already present (test teardown may leave stale)
        _registry._tools[self.name] = td
        return td

    # ------------------------------------------------------------------
    # Internal: dispatching logic
    # ------------------------------------------------------------------

    async def _dispatch(self, **kwargs: Any) -> Any:
        """Called by the ToolDef's fn when the dispatcher routes a call here."""
        self._calls.append(dict(kwargs))

        se = self._side_effect
        if se is not None:
            # Exception instance
            if isinstance(se, BaseException):
                raise se
            # Exception class
            if isinstance(se, type) and issubclass(se, BaseException):
                raise se()
            # Callable
            if callable(se):
                result = se(**kwargs)
                # Support async side_effect
                import inspect
                if inspect.isawaitable(result):
                    result = await result
                return result

        return self._return_value

    # ------------------------------------------------------------------
    # Call tracking
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Number of times this mock has been called."""
        return len(self._calls)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """List of kwargs dicts for each call, in order."""
        return list(self._calls)

    def reset(self) -> None:
        """Clear all recorded calls.  Does not affect registration."""
        self._calls.clear()

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_called(self) -> None:
        """Raise :exc:`AssertionError` if the tool was never called."""
        if not self._calls:
            raise AssertionError(
                f"MockTool '{self.name}' was never called.\n"
                "  Why: the agent run did not invoke this tool.\n"
                "  Fix: check that the mock's name matches the tool_calls in your script."
            )

    def assert_called_with(self, **kwargs: Any) -> None:
        """Assert the most recent call was made with the given kwargs.

        Args:
            **kwargs: Expected keyword arguments of the last call.

        Raises:
            AssertionError: If never called or if last call kwargs do not match.
        """
        self.assert_called()
        last = self._calls[-1]
        for key, expected in kwargs.items():
            if key not in last:
                raise AssertionError(
                    f"MockTool '{self.name}': expected kwarg '{key}' not found "
                    f"in last call.\n"
                    f"  Last call kwargs: {last}"
                )
            actual = last[key]
            if actual != expected:
                raise AssertionError(
                    f"MockTool '{self.name}': kwarg '{key}' mismatch.\n"
                    f"  Expected: {expected!r}\n"
                    f"  Actual:   {actual!r}"
                )

    def assert_call_count(self, n: int) -> None:
        """Assert that the tool was called exactly *n* times.

        Args:
            n: Expected call count.

        Raises:
            AssertionError: If call count does not equal *n*.
        """
        if self.call_count != n:
            raise AssertionError(
                f"MockTool '{self.name}': expected {n} call(s), "
                f"got {self.call_count}.\n"
                f"  Calls: {self._calls}"
            )

    def assert_not_called(self) -> None:
        """Raise :exc:`AssertionError` if the tool was called at all."""
        if self._calls:
            raise AssertionError(
                f"MockTool '{self.name}' was called {self.call_count} time(s) "
                "but was expected not to be called.\n"
                f"  Calls: {self._calls}"
            )

    def __repr__(self) -> str:
        return (
            f"MockTool(name={self.name!r}, calls={self.call_count}, "
            f"return_value={self._return_value!r})"
        )
