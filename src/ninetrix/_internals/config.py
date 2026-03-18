"""
ninetrix._internals.config
==========================
L1 kernel — stdlib only, no ninetrix imports.

NinetrixConfig — immutable SDK configuration resolved in layered priority order:

  1. Explicit kwargs passed to NinetrixConfig.load()   (highest priority)
  2. NINETRIX_* environment variables
  3. Project-local .ninetrix.toml  (cwd or nearest parent)
  4. User-global ~/.ninetrix/config.toml
  5. Built-in defaults                                  (lowest priority)

All fields are read-only after construction (frozen=True).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# TOML loading — stdlib tomllib (3.11+) with no fallback needed
# ---------------------------------------------------------------------------

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — we require 3.11+
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file. Returns empty dict if missing or parse error."""
    if not path.exists():
        return {}
    if tomllib is None:  # pragma: no cover
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _find_project_toml() -> Path | None:
    """
    Walk up from cwd looking for .ninetrix.toml.
    Returns the first match, or None if not found.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".ninetrix.toml"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# NinetrixConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NinetrixConfig:
    """
    Immutable SDK-wide configuration.

    Use NinetrixConfig.load() to construct — never instantiate directly.

    Fields
    ------
    default_provider:   LLM provider used when Agent(provider=) is not set.
    default_model:      Model used when Agent(model=) is not set.
    default_temperature: Sampling temperature (0.0–2.0).
    api_url:            Ninetrix Cloud / local API base URL for telemetry + runner events.
    runner_token:       Bearer token sent with runner events (machine secret or workspace token).
    mcp_gateway_url:    MCP gateway base URL (e.g. http://localhost:8080).
    workspace_id:       Workspace identifier (used by TenantContext auto-init).
    api_key:            Ninetrix API key (nxt_...) for cloud features and RemoteAgent.
    telemetry_enabled:  Whether anonymous SDK usage telemetry is sent.
    debug:              Enable verbose debug logging.
    log_level:          Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
    """

    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-6"
    default_temperature: float = 0.0

    api_url: str = ""
    runner_token: str = ""
    mcp_gateway_url: str = ""
    workspace_id: str = ""
    api_key: str = ""

    telemetry_enabled: bool = True
    debug: bool = False
    log_level: str = "WARNING"

    @classmethod
    def load(
        cls,
        *,
        default_provider: str | None = None,
        default_model: str | None = None,
        default_temperature: float | None = None,
        api_url: str | None = None,
        runner_token: str | None = None,
        mcp_gateway_url: str | None = None,
        workspace_id: str | None = None,
        api_key: str | None = None,
        telemetry_enabled: bool | None = None,
        debug: bool | None = None,
        log_level: str | None = None,
        _project_toml: Path | None = ...,   # type: ignore[assignment]  — sentinel
        _user_toml: Path | None = ...,      # type: ignore[assignment]  — sentinel
    ) -> "NinetrixConfig":
        """
        Build a NinetrixConfig by merging all layers, highest priority first.

        The _project_toml and _user_toml parameters exist for testing only —
        pass explicit Path objects to override the default discovery logic.
        """
        # --- Layer 4: user global toml ---
        user_toml_path = (
            _user_toml
            if _user_toml is not ...  # type: ignore[comparison-overlap]
            else Path.home() / ".ninetrix" / "config.toml"
        )
        user_toml = _load_toml(user_toml_path) if user_toml_path else {}

        # --- Layer 3: project local toml ---
        project_toml_path = (
            _project_toml
            if _project_toml is not ...  # type: ignore[comparison-overlap]
            else _find_project_toml()
        )
        project_toml = _load_toml(project_toml_path) if project_toml_path else {}

        # Merge toml layers — project overrides user
        merged_toml = {**user_toml, **project_toml}

        # --- Layer 2: environment variables ---
        env = _read_env()

        # --- Resolution: kwargs > env > toml > defaults ---
        def resolve_str(kwarg: str | None, env_key: str, toml_key: str, default: str) -> str:
            if kwarg is not None:
                return kwarg
            if env_key in env:
                return env[env_key]
            if toml_key in merged_toml:
                v = merged_toml[toml_key]
                return str(v) if not isinstance(v, str) else v
            return default

        def resolve_bool(kwarg: bool | None, env_key: str, toml_key: str, default: bool) -> bool:
            if kwarg is not None:
                return kwarg
            if env_key in env:
                return env[env_key].lower() not in ("0", "false", "off", "no")
            if toml_key in merged_toml:
                v = merged_toml[toml_key]
                if isinstance(v, bool):
                    return v
                return str(v).lower() not in ("0", "false", "off", "no")
            return default

        def resolve_float(kwarg: float | None, env_key: str, toml_key: str, default: float) -> float:
            if kwarg is not None:
                return kwarg
            if env_key in env:
                try:
                    return float(env[env_key])
                except ValueError:
                    pass
            if toml_key in merged_toml:
                try:
                    return float(merged_toml[toml_key])
                except (ValueError, TypeError):
                    pass
            return default

        return cls(
            default_provider=resolve_str(
                default_provider, "NINETRIX_DEFAULT_PROVIDER", "default_provider", "anthropic"
            ),
            default_model=resolve_str(
                default_model, "NINETRIX_DEFAULT_MODEL", "default_model", "claude-sonnet-4-6"
            ),
            default_temperature=resolve_float(
                default_temperature, "NINETRIX_DEFAULT_TEMPERATURE", "default_temperature", 0.0
            ),
            api_url=resolve_str(
                api_url,
                "NINETRIX_API_URL",
                "api_url",
                os.environ.get("AGENTFILE_API_URL", ""),   # legacy alias
            ),
            runner_token=resolve_str(
                runner_token,
                "NINETRIX_RUNNER_TOKEN",
                "runner_token",
                os.environ.get("AGENTFILE_RUNNER_TOKEN", ""),  # legacy alias
            ),
            mcp_gateway_url=resolve_str(
                mcp_gateway_url, "NINETRIX_MCP_GATEWAY_URL", "mcp_gateway_url", ""
            ),
            workspace_id=resolve_str(
                workspace_id, "NINETRIX_WORKSPACE_ID", "workspace_id", ""
            ),
            api_key=resolve_str(
                api_key, "NINETRIX_API_KEY", "api_key", ""
            ),
            telemetry_enabled=resolve_bool(
                telemetry_enabled, "NINETRIX_TELEMETRY", "telemetry_enabled", True
            ),
            debug=resolve_bool(
                debug, "NINETRIX_DEBUG", "debug", False
            ),
            log_level=resolve_str(
                log_level, "NINETRIX_LOG_LEVEL", "log_level", "WARNING"
            ),
        )


def _read_env() -> dict[str, str]:
    """Return all NINETRIX_* environment variables as a plain dict."""
    return {k: v for k, v in os.environ.items() if k.startswith("NINETRIX_")}
