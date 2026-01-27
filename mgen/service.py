from typing import Dict, Optional, List, Any

from mgen.library import JsonMusicLibrary
from mgen.utils.scorer import Scorer
from mret.vector_retriever import VectorMusicRetriever
from utils.harmonic import harmonic_distance, harmonic_relation_level

# 假设你有一个乐理工具模块，如果没有，可以在这里简单的实现
# from .utils import get_harmonic_distance 

class MusicService:
    """
    领域服务层：封装音乐检索、生成和智能选段的核心业务逻辑。
    """
    def __init__(self, tracks_json_path: str):
        print(f"[MusicService] Loading library from {tracks_json_path}...")
        self.library = JsonMusicLibrary(tracks_json_path)
        self.retriever = VectorMusicRetriever(tracks_json_path)
        
        self.high_energy_keywords = {"紧张", "激烈", "壮观", "高潮", "intense", "climax", "epic"}
        
        self.scorer = Scorer(sem_alpha=0.75, emb_map="cos_to_01")

    def smart_search(self, 
                     query: str, 
                     visual_summary: str = "",
                     prev_track_key: Optional[str] = None, 
                     strict_harmony: bool = True,          
                     top_k: int = 5) -> Optional[Dict[str, Any]]:
        
        full_query = f"{query} {visual_summary}".strip()
        
        # 1. 向量检索 (召回)
        # 稍微多召回一点，方便后面做 Filter
        candidates = self.retriever.search(full_query, top_k=top_k * 2)
        if not candidates: return None

        # 2. 乐理过滤/重排 (The Filter/Reranker)
        filtered_candidates = []
        
        for cand in candidates:
            cand_key = cand.get('key')
            level = harmonic_relation_level(prev_track_key, cand_key)
            cand["harmonic_level"] = level  # 0/1/2

            # strict 模式：过滤“远关系”
            if strict_harmony and prev_track_key and level >= 2:
                continue
                
            sem_score, sem_comp = self.scorer.score_sem(
                query=query,
                visual_summary=visual_summary,
                track_style=cand.get("style", ""),
                track_tags=cand.get("tags", []),
                retriever_match_score=cand.get("match_score", None),
                extra_text=cand.get("title", ""),
            )
            cand["sem_score"] = sem_score
            cand["sem_components"] = {
                "emb_cos": sem_comp.emb_cos,
                "emb_score": sem_comp.emb_score,
                "kw_jaccard": sem_comp.kw_jaccard,
            }

            filtered_candidates.append(cand)
            
        # 如果过滤完没剩下了，且 strict=True，这就触发了 Failure Tag: FILTER_KILL
        # 但在 Service 层，我们可以选择回退到“不严格模式”或者返回空让 Agent 处理
        # 这里我们返回空，让 Agent 决定去调用 relax_constraint
        if strict_harmony and not filtered_candidates:
            print(f"  [Service] Strict harmony killed all candidates for key {prev_track_key}")
            return None 

        # 重新按语义分排序，取 Top-1
        filtered_candidates.sort(key=lambda x: x.get("sem_score", 0.0), reverse=True)
        best = filtered_candidates[0]

        start_offset = 0.0
        if any(k in full_query for k in self.high_energy_keywords):
            start_offset = best.get("chorus_start", 0.0)

        return self._format_track(best, start_offset)
    
    def _calculate_harmonic_distance(self, key_a, key_b):
        return harmonic_distance(key_a, key_b)

    def generate_music(self, prompt: str, duration: float) -> Dict[str, Any]:
        """
        封装生成逻辑 (Suno/MusicGen)
        """
        # TODO: 接入真实生成 API
        return {
            "id": f"gen_{hash(prompt)}",
            "source": "suno_generated",
            "duration": duration,
            "play_start": 0.0,
            "meta": {
                "title": f"Generated: {prompt[:10]}...",
                "artist": "AI Composer",
                "bpm": 100, # 假设值
                "key": "C",
                "sem_score": 0.95, # 生成音乐通常语义匹配度高
                "harm_score": 1.0, # 假设生成时考虑了上下文
                "filepath": "generated/temp.mp3"
            }
        }

    def _format_track(self, track_info: Dict, start_offset: float) -> Dict[str, Any]:
        return {
            "track_id": track_info["id"],
            "source": "library",
            "duration": track_info.get("duration", 180.0),
            "play_start": start_offset,
            "play_duration": 0.0,
            "meta": {
                "title": track_info.get("title", "Unknown"),
                "artist": track_info.get("artist", "Unknown"),
                "bpm": track_info.get("bpm", 120),
                "key": track_info.get("key", "C"),
                "harm_score": 0.0,

                "sem_score": track_info.get("sem_score", 0.0),

                "sem_raw_cos": track_info.get("match_score", None),
                "sem_components": track_info.get("sem_components", {}),

                "filepath": track_info.get("filepath", "")
            }
        }