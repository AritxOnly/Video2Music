from typing import List, Dict, Any, Tuple, Callable
import math
import os

# === 依赖引入 ===
from vsem.model import VideoSemantics, SegmentSemantics
from vlm.model import TimelineEvent

# 尝试引入 sentence_transformers，如果没安装也不影响字典法运行
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_AI_BACKEND = True
except ImportError:
    HAS_AI_BACKEND = False

# ==========================================
# 策略 A: 基于 Russell 环状模型的字典规则 (Baseline)
# ==========================================
MOOD_VECTORS = {
    # High Energy
    "紧张": (-0.6, 0.8), "激动": (0.7, 0.8), "壮观": (0.5, 0.7),
    "力量": (0.3, 0.8), "危险": (-0.8, 0.9), "热烈": (0.8, 0.8),
    # Low Energy
    "宁静": (0.6, -0.6), "安静": (0.5, -0.7), "忧伤": (-0.6, -0.5),
    "悲伤": (-0.7, -0.4), "平和": (0.7, -0.6), "悠闲": (0.6, -0.5),
    "沉思": (0.0, -0.4), "平静": (0.65, -0.6), # 补充
    # Mid Energy
    "神秘": (-0.1, 0.1), "快乐": (0.9, 0.5), "中性": (0.0, 0.0),
}
DEFAULT_VECTOR = (0.0, 0.0)

def _get_mood_vector_rule(mood_str: str) -> Tuple[float, float]:
    if not mood_str: return DEFAULT_VECTOR
    if mood_str in MOOD_VECTORS: return MOOD_VECTORS[mood_str]
    for key, vec in MOOD_VECTORS.items():
        if key in mood_str: return vec
    return DEFAULT_VECTOR

def _calculate_distance_rule(mood_a: str, mood_b: str) -> float:
    v1 = _get_mood_vector_rule(mood_a)
    v2 = _get_mood_vector_rule(mood_b)
    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)


# ==========================================
# 策略 B: 基于 Embedding 的语义计算 (Advanced)
# ==========================================
_embed_model = None

def _load_ai_model():
    global _embed_model
    if _embed_model is None:
        if not HAS_AI_BACKEND:
            raise ImportError("需要安装 sentence-transformers 才能使用 AI 策略: pip install sentence-transformers")
        print(">>> [LazyLoad] Loading Embedding Model (bge-small-zh)...")
        _embed_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    return _embed_model

def _calculate_distance_ai(mood_a: str, mood_b: str) -> float:
    model = _load_ai_model()
    if not mood_a or not mood_b: return 1.0
    
    emb1 = model.encode(mood_a, convert_to_tensor=True)
    emb2 = model.encode(mood_b, convert_to_tensor=True)
    
    # Distance = 1 - Similarity
    return 1.0 - util.cos_sim(emb1, emb2).item()    # 计算他们的余弦相似度


# ==========================================
# 核心融合逻辑
# ==========================================

def assemble_video_semantics(
    events: List[TimelineEvent], 
    global_tags: List[str] = None,
    strategy: str = "ai"  # 可选: "ai" 或 "rule"
) -> VideoSemantics:
    """
    [Pipeline V2.1] 上下文融合算法
    :param strategy: 'ai' (使用语义模型) 或 'rule' (使用字典查表)
    """
    if not events:
        return VideoSemantics(global_tags=[], segments=[])
    if global_tags is None: global_tags = []

    # === 策略选择 ===
    calc_func: Callable[[str, str], float]
    threshold: float
    
    if strategy == "ai":
        calc_func = _calculate_distance_ai
        threshold = 0.30  # AI 语义距离阈值 (建议 0.25-0.35)
        print(f"    [Fusion-AI] Strategy: Embedding Similarity (Th={threshold})")
    else:
        calc_func = _calculate_distance_rule
        threshold = 0.60  # 规则欧氏距离阈值 (建议 0.5-0.8)
        print(f"    [Fusion-Rule] Strategy: Russell Circumplex Dictionary (Th={threshold})")

    merged_segments: List[SegmentSemantics] = []
    
    # 初始化第一个片段
    current_fusion = _init_fusion_block(events[0])

    for i in range(1, len(events)):
        current_event = events[i]
        last_mood = current_fusion["moods"][-1]
        
        # 调用选定的距离函数
        dist = calc_func(last_mood, current_event.label)
        
        if dist < threshold:
            # === MERGE ===
            # print(f"      Merge: '{last_mood}' + '{current_event.label}' (Dist={dist:.3f})")
            _extend_fusion_block(current_fusion, current_event)
        else:
            # === CUT ===
            print(f"      CUT: '{last_mood}' // '{current_event.label}' (Dist={dist:.3f} > {threshold})")
            merged_segments.append(_finalize_segment(current_fusion))
            current_fusion = _init_fusion_block(current_event)

    merged_segments.append(_finalize_segment(current_fusion))
    
    print(f"    [Fusion] Result: {len(events)} raw -> {len(merged_segments)} coherent movements.")

    return VideoSemantics(
        global_tags=list(set(global_tags)),
        segments=merged_segments,
        extra={"fusion_strategy": strategy}
    )


# === 辅助工具函数 ===

def _init_fusion_block(event: TimelineEvent) -> Dict[str, Any]:
    return {
        "start": event.start_sec,
        "end": event.end_sec,
        "moods": [event.label],
        "activities": [event.extra.get("activity", "")],
        "tags": event.extra.get("tags", []),
        "descriptions": [event.description]
    }

def _extend_fusion_block(block: Dict[str, Any], event: TimelineEvent):
    block["end"] = event.end_sec
    block["moods"].append(event.label)
    
    act = event.extra.get("activity", "")
    if act: block["activities"].append(act)
    
    ts = event.extra.get("tags", [])
    if ts: block["tags"].extend(ts)
    
    if event.description:
        block["descriptions"].append(event.description)

def _finalize_segment(fusion_data: Dict[str, Any]) -> SegmentSemantics:
    # 投票选出代表 Mood
    mood_counts = {}
    for m in fusion_data["moods"]:
        mood_counts[m] = mood_counts.get(m, 0) + 1
    representative_mood = max(mood_counts, key=mood_counts.get)
    
    # 摘要合并
    full_desc = " ".join(fusion_data["descriptions"])
    summary = full_desc[:50] + "..." if len(full_desc) > 50 else full_desc
    
    # 动作和标签去重
    unique_acts = sorted(list(set(fusion_data["activities"])))
    combined_activity = ", ".join(unique_acts)
    unique_tags = list(set(fusion_data["tags"]))
    
    return SegmentSemantics(
        start_sec=fusion_data["start"],
        end_sec=fusion_data["end"],
        mood=representative_mood,
        activity=combined_activity,
        tags=unique_tags,
        text_summary=summary,
        extra={"sub_event_count": len(fusion_data["moods"])}
    )