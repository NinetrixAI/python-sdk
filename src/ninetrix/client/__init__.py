"""
client/ — remote agent calling primitives.

Layer: L9 — may import all layers below.

  AgentClient  — HTTP wrapper for a running agent container's /invoke endpoint.
  RemoteAgent  — Ninetrix Cloud client (POST /v1/agents/{slug}/invoke).

Both satisfy AgentProtocol, so they are drop-in replacements for Agent in any
Workflow or Team that accepts AgentProtocol.
"""

from ninetrix.client.local import AgentClient as AgentClient
from ninetrix.client.remote import RemoteAgent as RemoteAgent

__all__ = ["AgentClient", "RemoteAgent"]
