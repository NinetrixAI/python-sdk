"""
ninetrix.testing — testing utilities for Ninetrix SDK agents.

Layer: L9 (testing) — may import L8 (agent) and all below.

Provides:
  MockTool     — fake tool implementation with call tracking and assertions.
  AgentSandbox — isolated test environment for running agents without real LLM calls.

Quick start::

    from ninetrix.testing import MockTool, AgentSandbox

    async with AgentSandbox(agent, script=["The answer is 42."]) as sandbox:
        sandbox.add_mock(MockTool("get_data", return_value={"value": 42}))
        result = await sandbox.run("What is the value?")
        assert result.success
        sandbox.assert_tool_called("get_data")
"""

from ninetrix.testing.mock_tool import MockTool as MockTool
from ninetrix.testing.sandbox import AgentSandbox as AgentSandbox

__all__ = [
    "MockTool",
    "AgentSandbox",
]
