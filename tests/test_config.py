"""
Unit tests for ninetrix._internals.config (PR 3).

Coverage:
- NinetrixConfig.load() with zero args → defaults
- Layer 1: explicit kwargs override everything
- Layer 2: NINETRIX_* env vars override toml + defaults
- Layer 3: project .ninetrix.toml overrides user toml + defaults
- Layer 4: user ~/.ninetrix/config.toml overrides defaults only
- Resolution order: kwargs > env > project toml > user toml > defaults
- Boolean parsing: "0", "false", "off", "no" → False
- Float parsing: NINETRIX_DEFAULT_TEMPERATURE
- Legacy env var aliases: AGENTFILE_API_URL, AGENTFILE_RUNNER_TOKEN
- Frozen: mutation raises FrozenInstanceError
- Missing toml files: silently ignored
- Malformed toml files: silently ignored (returns defaults)
"""

import os
import tempfile
from pathlib import Path
import pytest

from ninetrix._internals.config import NinetrixConfig, _load_toml, _find_project_toml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(content, encoding="utf-8")
    return p


def _load(**kwargs) -> NinetrixConfig:
    """Convenience: call load() with explicit empty toml paths to avoid picking
    up real user config during tests."""
    kwargs.setdefault("_user_toml", None)
    kwargs.setdefault("_project_toml", None)
    return NinetrixConfig.load(**kwargs)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_load_zero_args_returns_defaults(self, monkeypatch):
        # Strip all NINETRIX_* vars so defaults are clean
        for k in list(os.environ):
            if k.startswith("NINETRIX_") or k in ("AGENTFILE_API_URL", "AGENTFILE_RUNNER_TOKEN"):
                monkeypatch.delenv(k, raising=False)

        cfg = _load()
        assert cfg.default_provider == "anthropic"
        assert cfg.default_model == "claude-sonnet-4-6"
        assert cfg.default_temperature == 0.0
        assert cfg.api_url == ""
        assert cfg.runner_token == ""
        assert cfg.mcp_gateway_url == ""
        assert cfg.workspace_id == ""
        assert cfg.api_key == ""
        assert cfg.telemetry_enabled is True
        assert cfg.debug is False
        assert cfg.log_level == "WARNING"

    def test_returns_ninetrix_config_instance(self):
        cfg = _load()
        assert isinstance(cfg, NinetrixConfig)


# ---------------------------------------------------------------------------
# Layer 1: explicit kwargs (highest priority)
# ---------------------------------------------------------------------------

class TestExplicitKwargs:
    def test_kwargs_override_defaults(self):
        cfg = _load(default_provider="openai", default_model="gpt-4o")
        assert cfg.default_provider == "openai"
        assert cfg.default_model == "gpt-4o"

    def test_kwargs_override_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_DEFAULT_PROVIDER", "google")
        cfg = _load(default_provider="openai")
        assert cfg.default_provider == "openai"

    def test_kwargs_override_toml(self, tmp_path):
        toml = _make_toml(tmp_path, 'default_provider = "google"')
        cfg = NinetrixConfig.load(
            default_provider="openai",
            _user_toml=None,
            _project_toml=toml,
        )
        assert cfg.default_provider == "openai"

    def test_bool_kwarg_false(self):
        cfg = _load(telemetry_enabled=False)
        assert cfg.telemetry_enabled is False

    def test_bool_kwarg_true(self):
        cfg = _load(debug=True)
        assert cfg.debug is True

    def test_float_kwarg(self):
        cfg = _load(default_temperature=0.7)
        assert cfg.default_temperature == 0.7

    def test_all_string_fields_via_kwargs(self):
        cfg = _load(
            api_url="https://api.ninetrix.io",
            runner_token="tok_abc",
            mcp_gateway_url="http://localhost:8080",
            workspace_id="ws-123",
            api_key="nxt_test",
            log_level="DEBUG",
        )
        assert cfg.api_url == "https://api.ninetrix.io"
        assert cfg.runner_token == "tok_abc"
        assert cfg.mcp_gateway_url == "http://localhost:8080"
        assert cfg.workspace_id == "ws-123"
        assert cfg.api_key == "nxt_test"
        assert cfg.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Layer 2: environment variables
# ---------------------------------------------------------------------------

class TestEnvVars:
    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_DEFAULT_PROVIDER", "openai")
        cfg = _load()
        assert cfg.default_provider == "openai"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_DEFAULT_MODEL", "gpt-4o")
        cfg = _load()
        assert cfg.default_model == "gpt-4o"

    def test_temperature_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_DEFAULT_TEMPERATURE", "0.8")
        cfg = _load()
        assert cfg.default_temperature == 0.8

    def test_invalid_temperature_env_uses_default(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_DEFAULT_TEMPERATURE", "not-a-float")
        cfg = _load()
        assert cfg.default_temperature == 0.0

    def test_api_url_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_API_URL", "https://api.ninetrix.io")
        cfg = _load()
        assert cfg.api_url == "https://api.ninetrix.io"

    def test_workspace_id_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_WORKSPACE_ID", "ws-abc")
        cfg = _load()
        assert cfg.workspace_id == "ws-abc"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_API_KEY", "nxt_live_abc123")
        cfg = _load()
        assert cfg.api_key == "nxt_live_abc123"

    def test_telemetry_off_via_env(self, monkeypatch):
        for val in ("0", "false", "off", "no", "False", "OFF"):
            monkeypatch.setenv("NINETRIX_TELEMETRY", val)
            cfg = _load()
            assert cfg.telemetry_enabled is False, f"Expected False for NINETRIX_TELEMETRY={val!r}"

    def test_telemetry_on_via_env(self, monkeypatch):
        for val in ("1", "true", "yes", "True", "ON"):
            monkeypatch.setenv("NINETRIX_TELEMETRY", val)
            cfg = _load()
            assert cfg.telemetry_enabled is True, f"Expected True for NINETRIX_TELEMETRY={val!r}"

    def test_debug_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_DEBUG", "true")
        cfg = _load()
        assert cfg.debug is True

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_LOG_LEVEL", "DEBUG")
        cfg = _load()
        assert cfg.log_level == "DEBUG"

    def test_mcp_gateway_url_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_MCP_GATEWAY_URL", "http://gateway:8080")
        cfg = _load()
        assert cfg.mcp_gateway_url == "http://gateway:8080"

    def test_runner_token_from_env(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_RUNNER_TOKEN", "secret-token")
        cfg = _load()
        assert cfg.runner_token == "secret-token"


# ---------------------------------------------------------------------------
# Legacy env var aliases
# ---------------------------------------------------------------------------

class TestLegacyAliases:
    def test_agentfile_api_url_alias(self, monkeypatch):
        monkeypatch.delenv("NINETRIX_API_URL", raising=False)
        monkeypatch.setenv("AGENTFILE_API_URL", "http://localhost:8000")
        cfg = _load()
        assert cfg.api_url == "http://localhost:8000"

    def test_ninetrix_api_url_takes_priority_over_agentfile(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_API_URL", "https://new.ninetrix.io")
        monkeypatch.setenv("AGENTFILE_API_URL", "http://old.example.com")
        cfg = _load()
        assert cfg.api_url == "https://new.ninetrix.io"

    def test_agentfile_runner_token_alias(self, monkeypatch):
        monkeypatch.delenv("NINETRIX_RUNNER_TOKEN", raising=False)
        monkeypatch.setenv("AGENTFILE_RUNNER_TOKEN", "legacy-secret")
        cfg = _load()
        assert cfg.runner_token == "legacy-secret"


# ---------------------------------------------------------------------------
# Layer 3: project .ninetrix.toml
# ---------------------------------------------------------------------------

class TestProjectToml:
    def test_project_toml_overrides_defaults(self, tmp_path):
        toml = _make_toml(tmp_path, """
default_provider = "openai"
default_model = "gpt-4o-mini"
""")
        cfg = NinetrixConfig.load(_user_toml=None, _project_toml=toml)
        assert cfg.default_provider == "openai"
        assert cfg.default_model == "gpt-4o-mini"

    def test_project_toml_bool_field(self, tmp_path):
        toml = _make_toml(tmp_path, "telemetry_enabled = false\n")
        cfg = NinetrixConfig.load(_user_toml=None, _project_toml=toml)
        assert cfg.telemetry_enabled is False

    def test_project_toml_float_field(self, tmp_path):
        toml = _make_toml(tmp_path, "default_temperature = 0.5\n")
        cfg = NinetrixConfig.load(_user_toml=None, _project_toml=toml)
        assert cfg.default_temperature == 0.5

    def test_project_toml_overrides_user_toml(self, tmp_path):
        (tmp_path / "user").mkdir(exist_ok=True)
        user_toml = tmp_path / "user" / "config.toml"
        user_toml.write_text('default_provider = "google"\n')

        proj_toml = tmp_path / "proj.toml"
        proj_toml.write_text('default_provider = "openai"\n')

        cfg = NinetrixConfig.load(_user_toml=user_toml, _project_toml=proj_toml)
        assert cfg.default_provider == "openai"

    def test_env_overrides_project_toml(self, tmp_path, monkeypatch):
        toml = _make_toml(tmp_path, 'default_provider = "openai"\n')
        monkeypatch.setenv("NINETRIX_DEFAULT_PROVIDER", "google")
        cfg = NinetrixConfig.load(_user_toml=None, _project_toml=toml)
        assert cfg.default_provider == "google"


# ---------------------------------------------------------------------------
# Layer 4: user ~/.ninetrix/config.toml
# ---------------------------------------------------------------------------

class TestUserToml:
    def test_user_toml_overrides_defaults(self, tmp_path):
        user_toml = tmp_path / "config.toml"
        user_toml.write_text('log_level = "INFO"\n')
        cfg = NinetrixConfig.load(_user_toml=user_toml, _project_toml=None)
        assert cfg.log_level == "INFO"

    def test_user_toml_does_not_override_env(self, tmp_path, monkeypatch):
        user_toml = tmp_path / "config.toml"
        user_toml.write_text('log_level = "INFO"\n')
        monkeypatch.setenv("NINETRIX_LOG_LEVEL", "ERROR")
        cfg = NinetrixConfig.load(_user_toml=user_toml, _project_toml=None)
        assert cfg.log_level == "ERROR"


# ---------------------------------------------------------------------------
# Missing / malformed toml files
# ---------------------------------------------------------------------------

class TestTomlEdgeCases:
    def test_missing_user_toml_uses_defaults(self, tmp_path):
        missing = tmp_path / "nonexistent.toml"
        cfg = NinetrixConfig.load(_user_toml=missing, _project_toml=None)
        assert cfg.default_provider == "anthropic"

    def test_missing_project_toml_uses_defaults(self, tmp_path):
        missing = tmp_path / "nonexistent.toml"
        cfg = NinetrixConfig.load(_user_toml=None, _project_toml=missing)
        assert cfg.default_provider == "anthropic"

    def test_malformed_toml_silently_ignored(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("this is not valid toml ][[\n")
        cfg = NinetrixConfig.load(_user_toml=bad, _project_toml=None)
        assert cfg.default_provider == "anthropic"

    def test_none_toml_path_skips_file(self):
        cfg = NinetrixConfig.load(_user_toml=None, _project_toml=None)
        assert isinstance(cfg, NinetrixConfig)


# ---------------------------------------------------------------------------
# _load_toml helper
# ---------------------------------------------------------------------------

class TestLoadToml:
    def test_returns_dict_for_valid_toml(self, tmp_path):
        p = tmp_path / "f.toml"
        p.write_text('key = "value"\n')
        result = _load_toml(p)
        assert result == {"key": "value"}

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        assert _load_toml(tmp_path / "nope.toml") == {}

    def test_returns_empty_dict_for_malformed_toml(self, tmp_path):
        p = tmp_path / "bad.toml"
        p.write_text("[[[\n")
        assert _load_toml(p) == {}


# ---------------------------------------------------------------------------
# Frozen (immutable after load)
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_mutation_raises(self):
        cfg = _load()
        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
            cfg.default_provider = "openai"  # type: ignore[misc]

    def test_two_loads_return_equal_configs(self, monkeypatch):
        for k in list(os.environ):
            if k.startswith("NINETRIX_"):
                monkeypatch.delenv(k, raising=False)
        cfg1 = _load()
        cfg2 = _load()
        assert cfg1 == cfg2


# ---------------------------------------------------------------------------
# Usage example from usage_examples.py (PR 3 block smoke test)
# ---------------------------------------------------------------------------

def test_usage_example_pr3(monkeypatch):
    for k in list(os.environ):
        if k.startswith("NINETRIX_"):
            monkeypatch.delenv(k, raising=False)

    from ninetrix._internals.config import NinetrixConfig
    cfg = NinetrixConfig.load(_user_toml=None, _project_toml=None)
    assert cfg.default_provider in ("anthropic", "openai", "google", "groq", "mistral", "litellm")
