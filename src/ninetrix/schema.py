"""
Convert Python type annotations and docstrings into JSON Schema objects.

Supports the full set of types common in tool definitions:
  str, int, float, bool, list, dict, Optional[X], list[X], Literal["a","b"]
"""

from __future__ import annotations

import inspect
import re
import typing
from typing import Any, get_args, get_origin


# ── Type → JSON Schema ────────────────────────────────────────────────────────

_PRIMITIVE_MAP: dict[Any, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    bytes: {"type": "string", "contentEncoding": "base64"},
}


def type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Recursively convert a Python type annotation to a JSON Schema dict."""
    if annotation is inspect.Parameter.empty or annotation is type(None):
        return {"type": "string"}

    if annotation in _PRIMITIVE_MAP:
        return dict(_PRIMITIVE_MAP[annotation])

    # Bare list/dict (non-generic) must be checked before get_origin
    if annotation is list:
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # list[X] / List[X]
    if origin is list:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = type_to_json_schema(args[0])
        return schema

    # dict[K, V] / Dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Optional[X]  →  Union[X, None]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return type_to_json_schema(non_none[0])
        return {"anyOf": [type_to_json_schema(a) for a in non_none]}

    # Literal["a", "b"]
    if origin is typing.Literal:  # type: ignore[comparison-overlap]
        return {"type": "string", "enum": list(args)}

    # Tuple, Set — treat as array
    if origin in (tuple, set, frozenset):
        schema = {"type": "array"}
        if args:
            schema["items"] = type_to_json_schema(args[0])
        return schema

    # Any / unknown → no type constraint
    if annotation is typing.Any:
        return {}

    # Fallback
    return {"type": "string"}


# ── Docstring parser (Google style) ──────────────────────────────────────────

_SECTION_HEADERS = frozenset(
    ("args", "arguments", "parameters", "returns", "return", "raises", "note", "notes", "example", "examples")
)
_PARAM_LINE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """
    Parse a Google-style docstring into (summary, param_descriptions).

    Returns:
        summary: The first paragraph (main description).
        param_docs: Dict mapping param name → description string.
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().splitlines()
    summary_lines: list[str] = []
    param_docs: dict[str, str] = {}
    in_args = False

    for line in lines:
        stripped = line.strip()

        # Detect a named section header: a line like "Args:" or "Returns:"
        if stripped.endswith(":") and stripped[:-1].lower() in _SECTION_HEADERS:
            in_args = stripped[:-1].lower() in ("args", "arguments", "parameters")
            continue

        if in_args:
            # Only process indented lines as parameter entries
            if line.startswith((" ", "\t")) and stripped:
                m = _PARAM_LINE.match(stripped)
                if m:
                    param_docs[m.group(1)] = m.group(2).strip()
        elif stripped:
            summary_lines.append(stripped)

    summary = " ".join(summary_lines)
    return summary, param_docs


# ── Full parameter schema builder ─────────────────────────────────────────────

def build_parameters_schema(
    fn: Any,
    param_docs: dict[str, str],
) -> dict[str, Any]:
    """
    Build a JSON Schema ``object`` for all parameters of *fn*.

    Skips ``self``, ``cls``, and ``*args``/``**kwargs``.
    Parameters without defaults are marked as required.
    """
    sig = inspect.signature(fn)
    hints = _safe_get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(param_name, inspect.Parameter.empty)

        # Skip ToolContext-annotated parameters — injected at runtime, not by LLM.
        # Detected by marker attribute to avoid importing ToolContext here (circular).
        if getattr(annotation, "_is_tool_context", False):
            continue
        prop = type_to_json_schema(annotation)

        # Inject parameter description from docstring
        if param_name in param_docs:
            prop["description"] = param_docs[param_name]

        # Inject default value
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def _safe_get_type_hints(fn: Any) -> dict[str, Any]:
    """get_type_hints can fail with forward references — return empty dict on error."""
    try:
        import typing as _typing
        return _typing.get_type_hints(fn)
    except Exception:
        return {}
