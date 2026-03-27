"""
Example 5 — Toolkit grouping + streaming output.

Demonstrates:
  - Toolkit: bundle related @Tool functions into a named group
  - Passing a Toolkit directly to Agent(tools=[...])
  - Agent.stream() for real-time token-by-token output
  - StreamEvent types: token, tool_call, tool_result, done
  - YAML round-trip: agent.to_yaml() / Agent.from_yaml()

Use case: a code-review agent that can read files, run linting, and
          stream its feedback token-by-token to the terminal.

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from ninetrix import Agent, Toolkit, Tool


# ---------------------------------------------------------------------------
# 1. Define tools, grouped by concern
# ---------------------------------------------------------------------------

code_tools = Toolkit("code-review", description="Tools for reading and linting code files.")


@code_tools.tool
def read_file(path: str) -> str:
    """Read a text file from disk and return its contents.

    Args:
        path: Relative or absolute path to the file.
    """
    return Path(path).read_text(encoding="utf-8")


@code_tools.tool
def list_files(directory: str, extension: str = ".py") -> list[str]:
    """List all files with a given extension in a directory.

    Args:
        directory: Directory to scan.
        extension: File extension filter, e.g. ".py" or ".ts".
    """
    return [str(p) for p in Path(directory).rglob(f"*{extension}")]


@code_tools.tool
def run_lint(path: str) -> dict:
    """Run ruff linter on a Python file and return diagnostics.

    Args:
        path: Path to the Python file to lint.
    """
    result = subprocess.run(
        ["ruff", "check", "--output-format=json", path],
        capture_output=True,
        text=True,
    )
    import json
    try:
        diagnostics = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError:
        diagnostics = []
    return {"path": path, "issues": diagnostics, "exit_code": result.returncode}


# ---------------------------------------------------------------------------
# 2. Build the agent using the Toolkit
# ---------------------------------------------------------------------------

review_agent = Agent(
    name="code-reviewer",
    provider="anthropic",
    model="claude-sonnet-4-6",
    role=(
        "You are a senior software engineer performing a code review. "
        "Use your tools to read files, check for linting issues, and provide "
        "actionable feedback. Be specific: reference line numbers and suggest fixes."
    ),
    tools=[code_tools],   # pass the whole Toolkit — all three tools are registered
)


# ---------------------------------------------------------------------------
# 3. Inspect the agent
# ---------------------------------------------------------------------------

def show_info() -> None:
    info = review_agent.info()
    print(f"Agent  : {info.name}")
    print(f"Toolkit: {info.local_tools}")   # shows individual tool names from the kit

    # YAML round-trip
    yaml_str = review_agent.to_yaml()
    print(f"\n--- agent.to_yaml() ---\n{yaml_str}")
    reloaded = Agent.from_yaml(yaml_str)
    assert reloaded.config.name == review_agent.config.name
    print("YAML round-trip: OK\n")


# ---------------------------------------------------------------------------
# 4. Stream the review to the terminal
# ---------------------------------------------------------------------------

async def stream_review(target: str) -> None:
    print(f"Streaming code review for: {target}\n")
    print("=" * 60)

    async for event in review_agent.stream(
        f"Review the Python code in {target!r}. "
        "List the top 3 issues with specific line references and suggested fixes."
    ):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "tool_call":
            print(f"\n[tool] {event.tool_name}({event.tool_input})", flush=True)
        elif event.type == "tool_result":
            # Don't print full tool results — just a status line
            print(f"[tool result — {len(str(event.content))} chars]\n", flush=True)
        elif event.type == "done":
            print(f"\n{'=' * 60}")
            print(f"Tokens: {event.result.tokens_used}  |  Cost: ${event.result.cost_usd:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    show_info()

    # Review the examples folder itself (or pass any path you like)
    target_dir = "examples"
    asyncio.run(stream_review(target_dir))
