from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import re


# =========================
# 五度圈 & 调性解析工具
# =========================

# pitch class 映射（根音 -> 0..11）
_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# 五度圈顺序（pitch class），circle[i] = (7*i) mod 12
_CIRCLE = [(7 * i) % 12 for i in range(12)]
_CIRCLE_INDEX = {pc: i for i, pc in enumerate(_CIRCLE)}


@dataclass(frozen=True)
class ParsedKey:
    """解析后的调性表示：root 是根音（如 C, F#, Bb），is_minor 表示是否小调"""
    root: str
    is_minor: bool


def parse_key(key: Optional[str]) -> Optional[ParsedKey]:
    """
    解析各种常见调性字符串，尽量鲁棒：
    - 'C', 'Cm', 'C#m', 'Bb', 'Bb major', 'A minor'
    - 'C:maj', 'A:min'（librosa 常见）
    - 大小写不敏感（但输出 root 用规范形式）
    说明：
    - 目前只需要根音 + 是否小调，后续你要扩展到 mode/关系大小调可以加字段。
    """
    if not key or not isinstance(key, str):
        return None

    s = key.strip()
    if not s:
        return None

    s = s.replace("♭", "b").replace("♯", "#")  # 统一升降号
    s = s.replace(":major", "").replace(":maj", "")
    s = s.replace(":minor", "m").replace(":min", "m")
    s = s.lower()

    # 处理 "a minor" / "bb major" 这种
    s = s.replace(" major", "").replace(" minor", "m")

    # root: [a-g] + optional [#|b]
    m = re.match(r"^([a-g])([#b]?)(.*)$", s)
    if not m:
        return None

    note = m.group(1).upper()
    accidental = m.group(2)
    rest = (m.group(3) or "").strip()

    root = note + accidental

    # 是否小调：rest 以 m 开头 或者原本是 "Am" 形式
    is_minor = False
    if rest.startswith("m"):
        is_minor = True

    # 校验 root 是否能映射 pitch class
    if root not in _PC:
        return None

    return ParsedKey(root=root, is_minor=is_minor)


def circle_of_fifths_distance(key_a: Optional[str], key_b: Optional[str]) -> Optional[int]:
    """
    计算五度圈最短距离 D ∈ [0, 6]
    - 只使用根音 pitch class（暂不区分大小调的“调号关系”细节）
    - 解析失败返回 None
    """
    pa = parse_key(key_a)
    pb = parse_key(key_b)
    if pa is None or pb is None:
        return None

    pc_a = _PC[pa.root]
    pc_b = _PC[pb.root]

    ia = _CIRCLE_INDEX[pc_a]
    ib = _CIRCLE_INDEX[pc_b]
    d = abs(ia - ib)
    return min(d, 12 - d)


def harmonic_distance(key_a: Optional[str], key_b: Optional[str]) -> int:
    """
    你项目里建议统一用这个函数拿“乐理距离”：
    - 返回值保证是 int 且在 [0, 6] 内（解析失败则返回 6，按最差处理）
    """
    d = circle_of_fifths_distance(key_a, key_b)
    return d if d is not None else 6


def harmonic_relation_level(key_a: Optional[str], key_b: Optional[str]) -> int:
    """
    粗粒度关系等级（给 strict filter 用）：
    - 0: 完美/同调（distance==0）
    - 1: 近关系（distance in {1,2}）
    - 2: 远关系（distance>=3 或解析失败）
    说明：
    - 你原注释里写 “0=完美, 1=近关系, >2=远关系”，这里统一成 0/1/2 三档。
    """
    d = harmonic_distance(key_a, key_b)
    if d == 0:
        return 0
    if d <= 2:
        return 1
    return 2


def harmonic_score_from_distance(dist: int) -> float:
    """
    将五度圈距离 dist 映射为 [0,1] 的分数（给 Critic 的 S_harm 用）
    按你论文：Score = 1 - D/6
    """
    dist = max(0, min(6, int(dist)))
    return 1.0 - (dist / 6.0)