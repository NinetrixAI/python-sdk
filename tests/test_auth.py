"""
Unit tests for ninetrix._internals.auth (PR 4).

Coverage:
- CredentialStore.resolve(): all 4 layers + explicit_key
- CredentialStore.require(): success and CredentialError with what/why/fix message
- CredentialStore.resolve_workspace_token(): all resolution layers
- _from_env(): provider env var mapping, multiple vars per provider
- _from_credentials_toml(): valid / missing / malformed TOML
- _from_auth_json(): valid / missing / malformed JSON
- _from_machine_secret(): present / absent
- Resolution order: explicit > env > toml > auth.json > None
- No provider-specific logic — only key resolution
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from ninetrix._internals.auth import CredentialStore, _env_var_name, _PROVIDER_ENV_VARS
from ninetrix._internals.types import CredentialError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs):
    """Create a minimal NinetrixConfig with sane defaults for auth tests."""
    from ninetrix._internals.config import NinetrixConfig
    defaults = dict(
        _user_toml=None,
        _project_toml=None,
        api_key=kwargs.pop("api_key", ""),
    )
    defaults.update(kwargs)
    return NinetrixConfig.load(**defaults)


def _store(**cfg_kwargs) -> CredentialStore:
    return CredentialStore(_make_config(**cfg_kwargs))


# ---------------------------------------------------------------------------
# _env_var_name helper
# ---------------------------------------------------------------------------

class TestEnvVarName:
    def test_known_providers(self):
        assert _env_var_name("anthropic") == "ANTHROPIC_API_KEY"
        assert _env_var_name("openai") == "OPENAI_API_KEY"
        assert _env_var_name("google") == "GOOGLE_API_KEY"
        assert _env_var_name("groq") == "GROQ_API_KEY"
        assert _env_var_name("mistral") == "MISTRAL_API_KEY"

    def test_unknown_provider_generates_name(self):
        assert _env_var_name("myprovider") == "MYPROVIDER_API_KEY"


# ---------------------------------------------------------------------------
# resolve() — explicit_key (Layer 1)
# ---------------------------------------------------------------------------

class TestResolveExplicitKey:
    def test_explicit_key_returned_immediately(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = _store()
        assert store.resolve("anthropic", explicit_key="sk-explicit") == "sk-explicit"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        store = _store()
        assert store.resolve("anthropic", explicit_key="sk-explicit") == "sk-explicit"

    def test_empty_explicit_key_falls_through(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        store = _store()
        result = store.resolve("anthropic", explicit_key="")
        assert result == "sk-from-env"


# ---------------------------------------------------------------------------
# resolve() — env vars (Layer 2)
# ---------------------------------------------------------------------------

class TestResolveEnv:
    def test_anthropic_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        store = _store()
        assert store.resolve("anthropic") == "sk-ant-test"

    def test_openai_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        store = _store()
        assert store.resolve("openai") == "sk-openai-test"

    def test_google_primary_env_var(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "gkey-test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        store = _store()
        assert store.resolve("google") == "gkey-test"

    def test_google_fallback_env_var(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
        store = _store()
        assert store.resolve("google") == "gemini-test"

    def test_groq_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        store = _store()
        assert store.resolve("groq") == "gsk-test"

    def test_mistral_from_env(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "mist-test")
        store = _store()
        assert store.resolve("mistral") == "mist-test"

    def test_unknown_provider_tries_uppercase_env(self, monkeypatch):
        monkeypatch.setenv("MYPROVIDER_API_KEY", "mykey")
        store = _store()
        assert store.resolve("myprovider") == "mykey"

    def test_whitespace_only_env_var_ignored(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        store = _store()
        # Whitespace stripped → empty → falls through
        result = store.resolve("anthropic")
        assert result is None or result != "   "


# ---------------------------------------------------------------------------
# resolve() — credentials.toml (Layer 3)
# ---------------------------------------------------------------------------

class TestResolveCredentialsToml:
    def test_reads_provider_section(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ninetrix_dir = tmp_path / ".ninetrix"
        ninetrix_dir.mkdir()
        (ninetrix_dir / "credentials.toml").write_text('[anthropic]\napi_key = "sk-from-toml"\n')

        store = _store()
        store._credentials_toml = None  # ensure lazy load
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_credentials_toml("anthropic")
        assert result == "sk-from-toml"

    def test_missing_section_returns_none(self, monkeypatch, tmp_path):
        toml_content = '[openai]\napi_key = "sk-openai"\n'
        toml_path = tmp_path / ".ninetrix" / "credentials.toml"
        toml_path.parent.mkdir(parents=True)
        toml_path.write_text(toml_content)

        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_credentials_toml("anthropic")
        assert result is None

    def test_missing_toml_returns_none(self, tmp_path):
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_credentials_toml("anthropic")
        assert result is None

    def test_malformed_toml_returns_none(self, tmp_path):
        ninetrix_dir = tmp_path / ".ninetrix"
        ninetrix_dir.mkdir()
        (ninetrix_dir / "credentials.toml").write_text("[[[ not valid toml\n")

        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_credentials_toml("anthropic")
        assert result is None

    def test_toml_cached_after_first_load(self, tmp_path):
        store = _store()
        store._credentials_toml = {"anthropic": {"api_key": "cached-key"}}
        result = store._from_credentials_toml("anthropic")
        assert result == "cached-key"


# ---------------------------------------------------------------------------
# resolve() — auth.json (Layer 4)
# ---------------------------------------------------------------------------

class TestResolveAuthJson:
    def test_reads_provider_key(self, tmp_path):
        agentfile_dir = tmp_path / ".agentfile"
        agentfile_dir.mkdir()
        (agentfile_dir / "auth.json").write_text(
            json.dumps({"anthropic_api_key": "sk-from-auth-json"})
        )
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_auth_json("anthropic")
        assert result == "sk-from-auth-json"

    def test_missing_provider_in_auth_json_returns_none(self, tmp_path):
        agentfile_dir = tmp_path / ".agentfile"
        agentfile_dir.mkdir()
        (agentfile_dir / "auth.json").write_text(json.dumps({"openai_api_key": "sk-openai"}))
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_auth_json("anthropic")
        assert result is None

    def test_missing_auth_json_returns_none(self, tmp_path):
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_auth_json("anthropic")
        assert result is None

    def test_malformed_auth_json_returns_none(self, tmp_path):
        agentfile_dir = tmp_path / ".agentfile"
        agentfile_dir.mkdir()
        (agentfile_dir / "auth.json").write_text("{bad json")
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_auth_json("anthropic")
        assert result is None

    def test_auth_json_cached_after_first_load(self):
        store = _store()
        store._auth_json = {"anthropic_api_key": "cached"}
        assert store._from_auth_json("anthropic") == "cached"


# ---------------------------------------------------------------------------
# _from_machine_secret()
# ---------------------------------------------------------------------------

class TestMachineSecret:
    def test_reads_machine_secret(self, tmp_path):
        agentfile_dir = tmp_path / ".agentfile"
        agentfile_dir.mkdir()
        (agentfile_dir / ".api-secret").write_text("nxt_machine_secret\n")
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            result = store._from_machine_secret()
        assert result == "nxt_machine_secret"

    def test_missing_secret_file_returns_none(self, tmp_path):
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            assert store._from_machine_secret() is None

    def test_empty_secret_file_returns_none(self, tmp_path):
        agentfile_dir = tmp_path / ".agentfile"
        agentfile_dir.mkdir()
        (agentfile_dir / ".api-secret").write_text("   \n")
        store = _store()
        with patch.object(Path, "home", return_value=tmp_path):
            assert store._from_machine_secret() is None


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

class TestResolutionOrder:
    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        store = _store()
        assert store.resolve("anthropic", explicit_key="from-kwarg") == "from-kwarg"

    def test_env_beats_toml(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        store = _store()
        store._credentials_toml = {"anthropic": {"api_key": "from-toml"}}
        assert store.resolve("anthropic") == "from-env"

    def test_toml_beats_auth_json(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = _store()
        store._credentials_toml = {"anthropic": {"api_key": "from-toml"}}
        store._auth_json = {"anthropic_api_key": "from-auth-json"}
        assert store.resolve("anthropic") == "from-toml"

    def test_auth_json_beats_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = _store()
        store._credentials_toml = {}
        store._auth_json = {"anthropic_api_key": "from-auth-json"}
        assert store.resolve("anthropic") == "from-auth-json"

    def test_returns_none_when_nothing_found(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = _store()
        store._credentials_toml = {}
        store._auth_json = {}
        result = store.resolve("anthropic")
        assert result is None


# ---------------------------------------------------------------------------
# require() — success and error
# ---------------------------------------------------------------------------

class TestRequire:
    def test_returns_key_when_found(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        store = _store()
        assert store.require("anthropic") == "sk-ant-test"

    def test_raises_credential_error_when_missing(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = _store()
        store._credentials_toml = {}
        store._auth_json = {}
        with pytest.raises(CredentialError) as exc_info:
            store.require("anthropic")
        msg = str(exc_info.value)
        # Must contain what + why + fix
        assert "anthropic" in msg
        assert "ANTHROPIC_API_KEY" in msg
        assert "Fix" in msg or "fix" in msg.lower() or "export" in msg

    def test_error_message_mentions_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        store = _store()
        store._credentials_toml = {}
        store._auth_json = {}
        with pytest.raises(CredentialError) as exc_info:
            store.require("openai")
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_explicit_key_satisfies_require(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = _store()
        store._credentials_toml = {}
        store._auth_json = {}
        assert store.require("anthropic", explicit_key="sk-explicit") == "sk-explicit"


# ---------------------------------------------------------------------------
# resolve_workspace_token()
# ---------------------------------------------------------------------------

class TestResolveWorkspaceToken:
    def test_explicit_token_wins(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_API_KEY", "nxt_from_env")
        store = _store()
        assert store.resolve_workspace_token(explicit_token="nxt_explicit") == "nxt_explicit"

    def test_config_api_key_used(self, monkeypatch):
        monkeypatch.delenv("NINETRIX_API_KEY", raising=False)
        store = _store(api_key="nxt_from_config")
        assert store.resolve_workspace_token() == "nxt_from_config"

    def test_env_var_used_when_config_empty(self, monkeypatch):
        monkeypatch.setenv("NINETRIX_API_KEY", "nxt_from_env")
        store = _store(api_key="")
        assert store.resolve_workspace_token() == "nxt_from_env"

    def test_credentials_toml_used(self, monkeypatch, tmp_path):
        monkeypatch.delenv("NINETRIX_API_KEY", raising=False)
        store = _store(api_key="")
        store._credentials_toml = {"ninetrix": {"api_key": "nxt_from_toml"}}
        assert store.resolve_workspace_token() == "nxt_from_toml"

    def test_machine_secret_used_as_last_resort(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NINETRIX_API_KEY", raising=False)
        agentfile_dir = tmp_path / ".agentfile"
        agentfile_dir.mkdir()
        (agentfile_dir / ".api-secret").write_text("nxt_machine\n")
        store = _store(api_key="")
        store._credentials_toml = {}
        with patch.object(Path, "home", return_value=tmp_path):
            result = store.resolve_workspace_token()
        assert result == "nxt_machine"

    def test_returns_none_when_nothing_found(self, monkeypatch, tmp_path):
        monkeypatch.delenv("NINETRIX_API_KEY", raising=False)
        store = _store(api_key="")
        store._credentials_toml = {}
        store._auth_json = {}
        with patch.object(Path, "home", return_value=tmp_path):
            assert store.resolve_workspace_token() is None
