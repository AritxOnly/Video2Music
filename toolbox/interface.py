from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

from .model import *

class Toolbox:
    """
    统一工具入口：
    - 从 toolbox.json 加载 tool specs（用于 LLM/路由/校验）
    - 注册 tool 实现（纯函数/确定性实现优先）
    - execute(uid, args) 执行并返回 ToolResult
    """

    def __init__(self, spec_path: str | Path):
        self.spec_path = Path(spec_path)
        self.spec = self._load_spec(self.spec_path)
        self.tools: List[JSONObject] = list(self.spec.get("tools", []))
        self.registry: Dict[str, JSONObject] = {t["uid"]: t for t in self.tools}

        self.impls: Dict[str, Callable[[JSONObject], JSONObject]] = {}

    # -------------------------
    # Spec layer
    # -------------------------
    def _load_spec(self, path: Path) -> JSONObject:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_tools(self, tag: Optional[str] = None) -> List[JSONObject]:
        if tag is None:
            return self.tools
        out = []
        for t in self.tools:
            tags = t.get("tags", [])
            if tag in tags:
                out.append(t)
        return out 

    def get_tool(self, uid: str) -> JSONObject:
        if uid not in self.registry:
            raise KeyError(f"Unknown tool uid: {uid}")
        return self.registry[uid]

    def get_cost(self, uid: str) -> float:
        return float(self.get_tool(uid).get("cost", 0.0))

    # -------------------------
    # Runtime layer
    # -------------------------
    def register(self, uid: str, fn: Callable[[JSONObject], JSONObject]) -> None:
        """
        注册某个工具的实现函数：fn(args)->data
        """
        if uid not in self.registry:
            raise KeyError(f"Cannot register unknown tool uid: {uid}")
        self.impls[uid] = fn

    def has_impl(self, uid: str) -> bool:
        return uid in self.impls

    def execute(self, uid: str, args: JSONObject) -> ToolResult:
        """
        统一执行入口。这里不强制做 JSON schema 校验（可后续加上 jsonschema）。
        """
        if uid not in self.registry:
            return ToolResult(uid=uid, ok=False, data={}, error=f"Unknown tool uid: {uid}")
        if uid not in self.impls:
            return ToolResult(uid=uid, ok=False, data={}, error=f"Tool not implemented: {uid}")

        try:
            data = self.impls[uid](args or {})
            if not isinstance(data, dict):
                return ToolResult(uid=uid, ok=False, data={}, error="Tool impl must return a dict")
            return ToolResult(uid=uid, ok=True, data=data)
        except Exception as e:
            return ToolResult(uid=uid, ok=False, data={}, error=str(e))


# -------------------------
# Convenience: default loader
# -------------------------
def load_toolbox(toolbox_json_path: str | Path) -> Toolbox:
    return Toolbox(toolbox_json_path)


# -------------------------
# Example: wire stubs (replace with real impls)
# -------------------------
def register_default_stubs(tb: Toolbox) -> Toolbox:
    # Structural
    tb.register("struct.split", lambda a: {"movements": [], "delta_steps": +1})
    tb.register("struct.merge", lambda a: {"movements": [], "delta_steps": -1})

    # Retrieval
    tb.register("retrieval.search", lambda a: {"candidates": []})
    tb.register("retrieval.requery", lambda a: {"candidates": []})
    tb.register("retrieval.relax_constraint", lambda a: {"constraints": a.get("constraints", {})})

    # Editing
    tb.register("edit.continue", lambda a: {"track_id": a["prev_track_id"]})
    tb.register("edit.trim", lambda a: {"track_segment": {"track_id": a["track_id"], "start_time": a["start_time"], "duration": a["duration"]}})
    tb.register("edit.shift_align", lambda a: {"offset": a["offset"]})

    # Generation
    tb.register("gen.suno", lambda a: {"track_id": "generated_track_id", "metadata": {"prompt": a["prompt"]}})

    # Scoring
    tb.register("score.evaluate", lambda a: {"S_sem": 0.0, "S_harm": 0.0, "S_sync": 0.0, "S_flow": 0.0, "J": 0.0, "debug": {}})

    return tb