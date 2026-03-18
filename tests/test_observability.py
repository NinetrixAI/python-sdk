"""
Tests for observability debug module (PR 33).

Covers:
- attach_debug_listener: pretty-prints lifecycle events to stderr
- enable_debug(agent=...): attaches listener to agent._event_bus
- OTEL auto-attach: _otel_configured=True causes attach_otel_to_bus call
- NINETRIX_DEBUG=1 env var: sets root logger to DEBUG
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ninetrix.observability.events import AgentEvent, EventBus
from ninetrix.observability.debug import attach_debug_listener, _debug_handler
from ninetrix.observability.logger import _ensure_handler, _ROOT_LOGGER_NAME


# =============================================================================
# Helpers
# =============================================================================


def _make_event(
    etype: str,
    agent_name: str = "test-agent",
    thread_id: str = "abc123def456",
    data: dict | None = None,
) -> AgentEvent:
    return AgentEvent(
        type=etype,
        thread_id=thread_id,
        agent_name=agent_name,
        data=data or {},
    )


# =============================================================================
# attach_debug_listener — EventBus integration
# =============================================================================


@pytest.mark.asyncio
async def test_attach_debug_listener_subscribes(capsys):
    """attach_debug_listener must subscribe a handler to the bus."""
    bus = EventBus()
    assert bus.subscriber_count() == 0
    attach_debug_listener(bus)
    # Subscribed under the "*" global wildcard
    assert bus.subscriber_count("*") == 1


@pytest.mark.asyncio
async def test_debug_listener_run_start(capsys):
    """run.start event → output contains agent name and 'run started'."""
    bus = EventBus()
    attach_debug_listener(bus)

    await bus.emit(_make_event(
        "run.start",
        agent_name="analyst",
        thread_id="a3f9c2xxxxxxxx",
        data={"model": "claude-sonnet-4-6"},
    ))

    captured = capsys.readouterr()
    assert "analyst" in captured.err
    assert "run started" in captured.err
    assert "claude-sonnet-4-6" in captured.err


@pytest.mark.asyncio
async def test_debug_listener_turn_start(capsys):
    """turn.start event → output contains turn number."""
    bus = EventBus()
    attach_debug_listener(bus)

    await bus.emit(_make_event(
        "turn.start",
        agent_name="agent1",
        data={"turn": 0},
    ))

    captured = capsys.readouterr()
    assert "turn 1" in captured.err
    assert "agent1" in captured.err


@pytest.mark.asyncio
async def test_debug_listener_tool_call(capsys):
    """tool.call event → output contains tool name and args."""
    bus = EventBus()
    attach_debug_listener(bus)

    await bus.emit(_make_event(
        "tool.call",
        agent_name="searcher",
        data={"tool_name": "web_search", "arguments": {"q": "ESG trends"}},
    ))

    captured = capsys.readouterr()
    assert "web_search" in captured.err
    assert "searcher" in captured.err
    assert "->" in captured.err


@pytest.mark.asyncio
async def test_debug_listener_tool_result(capsys):
    """tool.result event → output contains tool name and result preview."""
    bus = EventBus()
    attach_debug_listener(bus)

    # First emit tool.call to seed the timer
    await bus.emit(_make_event(
        "tool.call",
        thread_id="tid001",
        agent_name="bot",
        data={"tool_name": "lookup", "arguments": {}},
    ))
    # Then emit tool.result
    await bus.emit(_make_event(
        "tool.result",
        thread_id="tid001",
        agent_name="bot",
        data={"tool_name": "lookup", "result": "ESG stands for Environmental"},
    ))

    captured = capsys.readouterr()
    assert "lookup" in captured.err
    assert "ESG stands for Environmental" in captured.err
    assert "<-" in captured.err


@pytest.mark.asyncio
async def test_debug_listener_turn_end(capsys):
    """turn.end event → output contains turn done and token counts."""
    bus = EventBus()
    attach_debug_listener(bus)

    await bus.emit(_make_event(
        "turn.end",
        agent_name="analyst",
        data={"turn": 0, "input_tokens": 420, "output_tokens": 110},
    ))

    captured = capsys.readouterr()
    assert "turn 1 done" in captured.err
    assert "420" in captured.err
    assert "110" in captured.err


@pytest.mark.asyncio
async def test_debug_listener_run_end(capsys):
    """run.end event → output contains 'run done', turns, tokens, cost."""
    bus = EventBus()
    attach_debug_listener(bus)

    # Seed a run start so elapsed can be computed
    await bus.emit(_make_event(
        "run.start",
        thread_id="run_end_test",
        agent_name="analyst",
        data={"model": "claude-sonnet-4-6"},
    ))
    await bus.emit(_make_event(
        "run.end",
        thread_id="run_end_test",
        agent_name="analyst",
        data={"steps": 2, "tokens_used": 1240, "cost_usd": 0.0031},
    ))

    captured = capsys.readouterr()
    assert "run done" in captured.err
    assert "turns=2" in captured.err
    assert "1240" in captured.err
    assert "0.0031" in captured.err
    assert "elapsed=" in captured.err


@pytest.mark.asyncio
async def test_debug_listener_error(capsys):
    """error event → output contains 'error' and message."""
    bus = EventBus()
    attach_debug_listener(bus)

    await bus.emit(_make_event(
        "error",
        agent_name="bot",
        data={"message": "something went wrong"},
    ))

    captured = capsys.readouterr()
    assert "error" in captured.err
    assert "something went wrong" in captured.err


@pytest.mark.asyncio
async def test_debug_listener_unknown_event(capsys):
    """Unknown event types print a generic line without crashing."""
    bus = EventBus()
    attach_debug_listener(bus)

    await bus.emit(_make_event(
        "checkpoint.saved",
        agent_name="bot",
        data={"thread_id": "t1"},
    ))

    captured = capsys.readouterr()
    assert "checkpoint.saved" in captured.err


# =============================================================================
# enable_debug with agent kwarg
# =============================================================================


def test_enable_debug_with_agent_attaches_listener():
    """enable_debug(agent=...) must attach the debug listener to agent._event_bus."""
    from ninetrix.observability.logger import enable_debug

    mock_bus = MagicMock()
    mock_agent = MagicMock()
    mock_agent._event_bus = mock_bus

    enable_debug(agent=mock_agent)

    # subscribe should have been called with "*" and the _debug_handler
    mock_bus.subscribe.assert_called_once_with("*", _debug_handler)


def test_enable_debug_no_agent_sets_log_level():
    """enable_debug() without agent still sets root logger to DEBUG."""
    from ninetrix.observability.logger import enable_debug

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    original_level = root.level

    try:
        enable_debug()
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(original_level)


def test_enable_debug_with_agent_sets_log_level():
    """enable_debug(agent=...) also sets root logger to DEBUG."""
    from ninetrix.observability.logger import enable_debug

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    original_level = root.level

    mock_bus = MagicMock()
    mock_agent = MagicMock()
    mock_agent._event_bus = mock_bus

    try:
        enable_debug(agent=mock_agent)
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(original_level)


# =============================================================================
# OTEL auto-attach in Agent._build_runner()
# =============================================================================


@pytest.mark.asyncio
async def test_otel_auto_attach_when_configured():
    """When _otel_configured=True, _build_runner() calls attach_otel_to_bus."""
    import ninetrix.observability.otel as otel_mod

    original = otel_mod._otel_configured
    try:
        otel_mod._otel_configured = True

        with patch(
            "ninetrix.observability.otel.attach_otel_to_bus"
        ) as mock_attach:
            # We need to test _build_runner — stub out the heavy dependencies
            with (
                patch("ninetrix._internals.auth.CredentialStore") as mock_creds_cls,
                patch("ninetrix._internals.config.NinetrixConfig.load") as mock_cfg,
                patch("ninetrix.agent.agent.Agent._build_provider") as mock_bp,
                patch("ninetrix.agent.agent.Agent._build_checkpointer") as mock_bc,
                patch("ninetrix.runtime.dispatcher.ToolDispatcher.initialize", new_callable=lambda: lambda self: asyncio.coroutine(lambda: None)()),
            ):
                # Set up the mock chain so _build_runner() doesn't crash
                mock_cfg.return_value = MagicMock(
                    workspace_id="",
                    mcp_gateway_token="",
                    mcp_gateway_url="",
                )
                mock_creds_instance = MagicMock()
                mock_creds_instance.resolve.return_value = "fake-key"
                mock_creds_instance.resolve_workspace_token.return_value = ""
                mock_creds_cls.return_value = mock_creds_instance
                mock_bp.return_value = MagicMock()
                mock_bc.return_value = MagicMock()

                # Patch ToolDispatcher.initialize to be a proper async no-op
                async def _noop_init(self):
                    pass

                with patch(
                    "ninetrix.runtime.dispatcher.ToolDispatcher.initialize",
                    _noop_init,
                ):
                    from ninetrix.agent.agent import Agent
                    agent = Agent(
                        provider="anthropic",
                        model="claude-sonnet-4-6",
                        api_key="fake-key",
                    )
                    await agent._build_runner()

                mock_attach.assert_called_once_with(agent._event_bus)
    finally:
        otel_mod._otel_configured = original


@pytest.mark.asyncio
async def test_otel_not_attached_when_not_configured():
    """When _otel_configured=False, attach_otel_to_bus is NOT called."""
    import ninetrix.observability.otel as otel_mod

    original = otel_mod._otel_configured
    try:
        otel_mod._otel_configured = False

        with patch(
            "ninetrix.observability.otel.attach_otel_to_bus"
        ) as mock_attach:
            with (
                patch("ninetrix._internals.auth.CredentialStore") as mock_creds_cls,
                patch("ninetrix._internals.config.NinetrixConfig.load") as mock_cfg,
                patch("ninetrix.agent.agent.Agent._build_provider") as mock_bp,
                patch("ninetrix.agent.agent.Agent._build_checkpointer") as mock_bc,
            ):
                mock_cfg.return_value = MagicMock(
                    workspace_id="",
                    mcp_gateway_token="",
                    mcp_gateway_url="",
                )
                mock_creds_instance = MagicMock()
                mock_creds_instance.resolve.return_value = "fake-key"
                mock_creds_instance.resolve_workspace_token.return_value = ""
                mock_creds_cls.return_value = mock_creds_instance
                mock_bp.return_value = MagicMock()
                mock_bc.return_value = MagicMock()

                async def _noop_init(self):
                    pass

                with patch(
                    "ninetrix.runtime.dispatcher.ToolDispatcher.initialize",
                    _noop_init,
                ):
                    from ninetrix.agent.agent import Agent
                    agent = Agent(
                        provider="anthropic",
                        model="claude-sonnet-4-6",
                        api_key="fake-key",
                    )
                    await agent._build_runner()

                mock_attach.assert_not_called()
    finally:
        otel_mod._otel_configured = original


# =============================================================================
# NINETRIX_DEBUG=1 env var
# =============================================================================


def test_ninetrix_debug_env_var_sets_debug_level(monkeypatch):
    """NINETRIX_DEBUG=1 must set root logger to DEBUG during _ensure_handler()."""
    import ninetrix.observability.logger as logger_mod

    monkeypatch.setenv("NINETRIX_DEBUG", "1")

    # Reset handler state so _ensure_handler runs again
    original_installed = logger_mod._handler_installed
    logger_mod._handler_installed = False

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    original_level = root.level
    # Remove existing handlers temporarily to avoid duplicate output
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        logger_mod._ensure_handler()
        assert root.level == logging.DEBUG
    finally:
        logger_mod._handler_installed = original_installed
        root.setLevel(original_level)
        root.handlers.clear()
        root.handlers.extend(original_handlers)


def test_ninetrix_debug_env_var_true_string(monkeypatch):
    """NINETRIX_DEBUG=true also triggers DEBUG level."""
    import ninetrix.observability.logger as logger_mod

    monkeypatch.setenv("NINETRIX_DEBUG", "true")

    original_installed = logger_mod._handler_installed
    logger_mod._handler_installed = False

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    original_level = root.level
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        logger_mod._ensure_handler()
        assert root.level == logging.DEBUG
    finally:
        logger_mod._handler_installed = original_installed
        root.setLevel(original_level)
        root.handlers.clear()
        root.handlers.extend(original_handlers)


def test_ninetrix_debug_env_var_absent(monkeypatch):
    """When NINETRIX_DEBUG is not set, level stays at WARNING (default)."""
    import ninetrix.observability.logger as logger_mod

    monkeypatch.delenv("NINETRIX_DEBUG", raising=False)
    monkeypatch.delenv("NINETRIX_LOG_LEVEL", raising=False)

    original_installed = logger_mod._handler_installed
    logger_mod._handler_installed = False

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    original_level = root.level
    original_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        logger_mod._ensure_handler()
        assert root.level == logging.WARNING
    finally:
        logger_mod._handler_installed = original_installed
        root.setLevel(original_level)
        root.handlers.clear()
        root.handlers.extend(original_handlers)
