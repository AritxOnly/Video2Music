from __future__ import annotations

from pathlib import Path
from typing import Optional

from .interface import *
from .model import *

__all__ = [
    "Toolbox",
    "ToolCall",
    "ToolResult",
    "load_toolbox",
    "register_default_stubs",
    "get_toolbox",
]

_TOOLBOX_SINGLETON: Optional[Toolbox] = None


def get_toolbox(
    spec_path: str | Path | None = None,
    *,
    with_stubs: bool = False,
    force_reload: bool = False,
) -> Toolbox:
    """
    统一封装入口：返回 Toolbox 单例（默认读取 toolbox/toolbox.json）。
    - spec_path: 可覆盖默认路径
    - with_stubs: 是否注册一套默认 stub（便于先跑通 pipeline）
    - force_reload: 强制重载 spec + 重新创建实例
    """
    global _TOOLBOX_SINGLETON

    if _TOOLBOX_SINGLETON is not None and not force_reload:
        return _TOOLBOX_SINGLETON

    default_spec = Path(__file__).resolve().parent / "toolbox.json"
    spec = Path(spec_path) if spec_path is not None else default_spec

    tb = load_toolbox(spec)

    if with_stubs:
        register_default_stubs(tb)

    _TOOLBOX_SINGLETON = tb
    return tb