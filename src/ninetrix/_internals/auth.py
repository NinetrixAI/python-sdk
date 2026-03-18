"""
ninetrix._internals.auth
========================
L1 kernel — stdlib only, no ninetrix imports (except sibling L1 modules).

CredentialStore resolves LLM API keys and workspace tokens for a given provider.

Resolution order (highest to lowest priority):
  1. Explicit kwarg passed to resolve()        — Agent(api_key="sk-...")
  2. NINETRIX_* / provider-specific env var    — ANTHROPIC_API_KEY, OPENAI_API_KEY, …
  3. ~/.ninetrix/credentials.toml              — [anthropic] api_key = "..."
  4. ~/.agentfile/auth.json                    — legacy CLI auth file
  5. None / raise CredentialError if required

No provider-specific logic lives here — only key resolution.
Validation (format check, no live API call) is provider-specific and belongs in providers/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ninetrix._internals.config import NinetrixConfig

# TOML loading (same pattern as config.py)
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Provider → env var mapping
# ---------------------------------------------------------------------------

# Maps provider name → ordered list of env vars to check (first match wins)
_PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai":    ["OPENAI_API_KEY"],
    "google":    ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "groq":      ["GROQ_API_KEY"],
    "mistral":   ["MISTRAL_API_KEY"],
    "litellm":   ["LITELLM_API_KEY", "OPENAI_API_KEY"],  # litellm often wraps openai
}


def _env_var_name(provider: str) -> str:
    """Return the primary env var name for a provider (for error messages)."""
    vars_ = _PROVIDER_ENV_VARS.get(provider.lower(), [])
    return vars_[0] if vars_ else f"{provider.upper()}_API_KEY"


# ---------------------------------------------------------------------------
# CredentialStore
# ---------------------------------------------------------------------------

class CredentialStore:
    """
    Resolves API keys for LLM providers and workspace tokens.

    Usage:
        store = CredentialStore(config)

        # Resolve with no explicit key — walks resolution chain
        key = store.resolve("anthropic")

        # Resolve with explicit key (highest priority — skips all other layers)
        key = store.resolve("anthropic", explicit_key="sk-ant-...")

        # Require the key — raises CredentialError if not found
        key = store.require("anthropic")
    """

    def __init__(self, config: "NinetrixConfig") -> None:
        self._config = config
        self._credentials_toml: dict | None = None   # lazy-loaded
        self._auth_json: dict | None = None          # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, provider: str, *, explicit_key: str | None = None) -> str | None:
        """
        Resolve an API key for provider. Returns None if not found anywhere.

        Args:
            provider:     Provider name ("anthropic", "openai", "google", …).
            explicit_key: Highest-priority key (from Agent(api_key=) or similar).

        Returns:
            The API key string, or None if not found.
        """
        provider = provider.lower()

        # Layer 1: explicit kwarg
        if explicit_key:
            return explicit_key

        # Layer 2: env vars
        key = self._from_env(provider)
        if key:
            return key

        # Layer 3: ~/.ninetrix/credentials.toml
        key = self._from_credentials_toml(provider)
        if key:
            return key

        # Layer 4: ~/.agentfile/auth.json (legacy CLI)
        key = self._from_auth_json(provider)
        if key:
            return key

        return None

    def require(self, provider: str, *, explicit_key: str | None = None) -> str:
        """
        Like resolve(), but raises CredentialError if no key is found.

        Use this in providers/ before making any API call.

        Raises:
            CredentialError: with what/why/how message.
        """
        from ninetrix._internals.types import CredentialError

        key = self.resolve(provider, explicit_key=explicit_key)
        if key:
            return key

        env_var = _env_var_name(provider)
        raise CredentialError(
            f"No API key found for provider '{provider}'.\n"
            f"  Why: {env_var} is not set and no key was found in\n"
            f"       ~/.ninetrix/credentials.toml or ~/.agentfile/auth.json.\n"
            f"  Fix: export {env_var}=<your-key>, or\n"
            f"       pass api_key='...' to Agent(...), or\n"
            f"       run: ninetrix auth login"
        )

    def resolve_workspace_token(self, *, explicit_token: str | None = None) -> str | None:
        """
        Resolve the Ninetrix workspace API token (nxt_...).

        Resolution order:
          1. explicit_token kwarg
          2. config.api_key  (set from NINETRIX_API_KEY env or credentials.toml)
          3. NINETRIX_API_KEY env var
          4. ~/.ninetrix/credentials.toml  [ninetrix] api_key
          5. ~/.agentfile/.api-secret      (legacy machine secret)
          6. None
        """
        if explicit_token:
            return explicit_token

        if self._config.api_key:
            return self._config.api_key

        key = os.environ.get("NINETRIX_API_KEY", "")
        if key:
            return key

        key = self._from_credentials_toml("ninetrix") or ""
        if key:
            return key

        return self._from_machine_secret()

    # ------------------------------------------------------------------
    # Private resolution helpers
    # ------------------------------------------------------------------

    def _from_env(self, provider: str) -> str | None:
        """Check provider-specific env vars."""
        for var in _PROVIDER_ENV_VARS.get(provider, [f"{provider.upper()}_API_KEY"]):
            val = os.environ.get(var, "").strip()
            if val:
                return val
        return None

    def _from_credentials_toml(self, provider: str) -> str | None:
        """
        Read ~/.ninetrix/credentials.toml.

        Expected format:
            [anthropic]
            api_key = "sk-ant-..."

            [openai]
            api_key = "sk-..."

            [ninetrix]
            api_key = "nxt_..."
        """
        data = self._load_credentials_toml()
        section = data.get(provider, {})
        if isinstance(section, dict):
            return section.get("api_key", None) or None
        return None

    def _from_auth_json(self, provider: str) -> str | None:
        """
        Read ~/.agentfile/auth.json (legacy CLI auth file).

        Expected format:
            {
              "anthropic_api_key": "sk-ant-...",
              "openai_api_key": "sk-..."
            }
        """
        data = self._load_auth_json()
        # Try <provider>_api_key key
        key = data.get(f"{provider}_api_key", "")
        return key if key else None

    def _from_machine_secret(self) -> str | None:
        """Read ~/.agentfile/.api-secret (single-line machine secret)."""
        secret_path = Path.home() / ".agentfile" / ".api-secret"
        if not secret_path.exists():
            return None
        try:
            return secret_path.read_text().strip() or None
        except OSError:
            return None

    # ------------------------------------------------------------------
    # Lazy TOML / JSON loaders
    # ------------------------------------------------------------------

    def _load_credentials_toml(self) -> dict:
        if self._credentials_toml is not None:
            return self._credentials_toml

        path = Path.home() / ".ninetrix" / "credentials.toml"
        if not path.exists() or tomllib is None:
            self._credentials_toml = {}
            return self._credentials_toml

        try:
            with path.open("rb") as f:
                self._credentials_toml = tomllib.load(f)
        except Exception:
            self._credentials_toml = {}

        return self._credentials_toml

    def _load_auth_json(self) -> dict:
        if self._auth_json is not None:
            return self._auth_json

        path = Path.home() / ".agentfile" / "auth.json"
        if not path.exists():
            self._auth_json = {}
            return self._auth_json

        try:
            self._auth_json = json.loads(path.read_text())
        except Exception:
            self._auth_json = {}

        return self._auth_json
