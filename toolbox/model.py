from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

JSONSchema = Dict[str, Any]


@dataclass(frozen=True)
class ToolDesc:
    """Human-readable tool descriptions.

    userDesc: shown to users / in logs
    promptDesc: compact instruction for LLM/tool-router
    """

    userDesc: str
    promptDesc: str


@dataclass(frozen=True)
class ToolSpec:
    """Deterministic, reproducible tool interface definition."""

    uid: str
    name: str
    desc: ToolDesc

    # Operation cost used in the objective: J = ... - Cost
    # Suggested range: [0.0, 1.0]
    cost: float = 0.0

    # JSON-schema-like I/O contracts
    input_schema: JSONSchema = field(default_factory=dict)
    output_schema: JSONSchema = field(default_factory=dict)

    # Optional tags for routing/analytics
    tags: List[str] = field(default_factory=list)
    
JSONObject = Dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    uid: str
    args: JSONObject


@dataclass(frozen=True)
class ToolResult:
    uid: str
    ok: bool
    data: JSONObject
    error: Optional[str] = None