import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from vsem.model import SegmentSemantics
from mret.vector_retriever import VectorMusicRetriever

# === 简单的乐理工具 ===
# 五度圈映射：C=0, G=1, D=2, A=3, E=4, B=5, F#=6, Db=7, Ab=8, Eb=9, Bb=10, F=11
# Major 用小写，Minor 用 'm' 后缀 (简化处理)
KEY_MAP = {
    'C Major': 0, 'G Major': 1, 'D Major': 2, 'A Major': 3, 'E Major': 4, 'B Major': 5, 
    'F# Major': 6, 'Gb Major': 6, 'Db Major': 7, 'C# Major': 7, 'Ab Major': 8, 'Eb Major': 9, 'Bb Major': 10, 'F Major': 11,
    'A Minor': 0, 'E Minor': 1, 'B Minor': 2, 'F# Minor': 3, 'C# Minor': 4, 'G# Minor': 5,
    'D# Minor': 6, 'Eb Minor': 6, 'Bb Minor': 7, 'F Minor': 8, 'C Minor': 9, 'G Minor': 10, 'D Minor': 11
}

def get_harmonic_distance(key1: str, key2: str) -> int:
    """计算两个调性在五度圈上的距离。0=完全和谐，距离越小越好。"""
    if not key1 or not key2: return 0
    val1 = KEY_MAP.get(key1, 0)
    val2 = KEY_MAP.get(key2, 0)
    
    diff = abs(val1 - val2)
    # 五度圈是环状的，距离最大为 6
    return min(diff, 12 - diff)

@dataclass
class MGenOptions:
    preferred_style: str = None
    global_gain_db: float = -6.0
    crossfade_sec: float = 0.5
    allow_generation: bool = True # 是否允许生成兜底

class SimpleRuleArranger:
    def arrange(
        self, 
        timeline: List[SegmentSemantics], 
        library_path: str, 
        options: MGenOptions
    ) -> List[Dict[str, Any]]:
        
        retriever = VectorMusicRetriever(library_path)
        plans = []
        last_key = None # 记录上一段音乐的调性
        
        print(f"    [Arranger] Harmonic Planning for {len(timeline)} movements...")

        for i, seg in enumerate(timeline):
            # 1. 构造 Query
            tags_context = " ".join(seg.tags[:3])
            query = f"{options.preferred_style or ''} {seg.mood} {tags_context} {seg.activity}"
            
            # 2. 获取 Top-5 候选 (语义层)
            candidates = retriever.search(query, top_k=5)
            
            # 3. 乐理重排序 (Harmonic Re-ranking)
            # 评分公式 = 语义分 * 0.7 + 乐理分 * 0.3
            # 如果是第一段，或者是生成模式，不需要考虑上一个 Key
            
            best_track = None
            
            # === 生成兜底判断 ===
            # 如果语义匹配度太低 (比如最高分只有 0.2)，说明库里没这歌
            if candidates[0]['match_score'] < 0.25 and options.allow_generation:
                print(f"      [Gen] Low match ({candidates[0]['match_score']:.2f}) for '{seg.mood}'. Generating...")
                best_track = self._generate_track(seg) # 调用生成接口
                # 生成的歌我们假设它会自动适配 (或者后续分析它的key)
                last_key = best_track.get('key') 
                
            else:
                # === 检索模式 ===
                if last_key:
                    # 有上一首，寻找最和谐的接班人
                    best_score = -1.0
                    for cand in candidates:
                        sem_score = cand['match_score']
                        # 计算调性距离 (0~6)，归一化到 0~1 (0是最好，1是最差)
                        dist = get_harmonic_distance(last_key, cand.get('key'))
                        harmonic_penalty = dist / 6.0 
                        
                        # 综合分：语义越高越好，调性距离越小越好
                        # 这里我们给调性一个小权重，避免选了一首完全不搭界但调性对的歌
                        final_score = sem_score - (harmonic_penalty * 0.15)
                        
                        if final_score > best_score:
                            best_score = final_score
                            best_track = cand
                else:
                    # 第一首，直接选语义最好的
                    best_track = candidates[0]
                
                last_key = best_track.get('key')

            # 4. 片段截取逻辑 (Segment Selection)
            # 如果是"紧张/激烈"的高能量段落，我们从 chorus_start 开始放
            # 如果是"宁静/铺垫"，从头开始放
            start_offset = 0.0
            if seg.mood in ["紧张", "激烈", "壮观", "高潮"]:
                 start_offset = best_track.get("chorus_start", 0.0)
            
            print(f"      [{i}] {seg.mood} -> '{best_track['id']}' (Key: {best_track.get('key')}, Offset: {start_offset}s)")

            # 5. 生成 Plan
            plan = {
                "start_time": seg.start_sec,
                "end_time": seg.end_sec,
                "track_id": best_track["id"],
                "track_path": best_track["filepath"],
                "source_start": start_offset, # 告诉 ffmpeg 从哪里开始截
                "volume_db": options.global_gain_db, 
                "fade_in": options.crossfade_sec,
                "fade_out": options.crossfade_sec
            }
            plans.append(plan)
            
        return plans

    def _generate_track(self, seg: SegmentSemantics) -> Dict[str, Any]:
        """
        [Stub] 调用 Suno/Udio API 生成音乐。
        这里先返回一个 Mock 对象，你可以在这里接入真正的 API。
        """
        # TODO: Call Suno API here
        # prompt = f"{seg.mood} style, {seg.activity}, {','.join(seg.tags)}"
        # url = suno_api.generate(prompt)
        # filepath = download(url)
        
        return {
            "id": f"gen_{random.randint(1000,9999)}",
            "filepath": "assets/music/lofi_01.mp3", # 暂时指向占位符
            "key": "C Major", # 假设生成的是 C 大调
            "chorus_start": 0.0
        }