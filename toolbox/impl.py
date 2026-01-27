from typing import Dict, Any
from toolbox.interface import Toolbox
from mgen import MusicService

_MUSIC_SERVICE: MusicService = None

def get_service(context: Dict) -> MusicService:
    global _MUSIC_SERVICE
    if _MUSIC_SERVICE is None:
        path = context.get('tracks_json_path')
        if not path:
            raise ValueError("Context missing 'tracks_json_path'")
        _MUSIC_SERVICE = MusicService(path)
    return _MUSIC_SERVICE

# TODO:

def register_real_tools(tb: Toolbox, context: Dict[str, Any] = {}):
    """
    在这里将真实的业务逻辑函数注册到 Toolbox
    """
    
    service = get_service(context=context)
    
    def impl_search(args: Dict) -> Dict:
        query = args.get("query", "")
        summary = args.get("visual_summary", "")
        prev_key = args.get("prev_track_key")
        
        print(f"    >> [Tool:Search] Query='{query}'")
        
        result = service.smart_search(query=query, visual_summary=summary, prev_track_key=prev_key, strict_harmony=True)
        
        if not result:
            return {} # 返回空字典表示没找到
            
        return result

    tb.register("retrieval.search", impl_search)
    tb.register("retrieval.requery", lambda args: impl_search(args))
    
    def impl_suno(args: Dict) -> Dict:
        prompt = args.get("prompt", "music")
        duration = args.get("duration", 30.0)
        print(f"    >> [Tool:Suno] Generating '{prompt}'")
        return service.generate_music(prompt, duration)

    tb.register("gen.suno", impl_suno)
    
    def impl_relax(args: Dict) -> Dict:
        print(f"    >> [Tool:Relax] Searching without harmonic constraints...")
        # 从 args 获取原始查询参数，如果没有则从 movement summary 推断
        query = args.get("query", "")
        summary = args.get("visual_summary", "")
        prev_key = args.get("prev_track_key") # 需要 MicroPlanner 注入
        
        result = service.smart_search(
            query=query, 
            visual_summary=summary, 
            prev_track_key=prev_key,
            strict_harmony=False
        )
        return result or {}
        
    tb.register("retrieval.relax_constraint", impl_relax)
    
    # Agent (ActionGenerator) 负责从 movement.shots 里挑一个最好的时间点 t
    # Tool 负责执行切分并分配 shots
    def impl_split(args: Dict) -> Dict:
        start = args.get("start_time")
        end = args.get("end_time")
        split_time = args.get("timestamp") 
        original_shots = args.get("shots", []) # 通过 Bridge 注入的原始 shots 列表
        
        print(f"    >> [Tool:Split] Splitting at {split_time:.2f}s")

        # 1. 分配 Shots 到两个新乐章
        shots_part1 = [s for s in original_shots if s['end_sec'] <= split_time]
        shots_part2 = [s for s in original_shots if s['end_sec'] > split_time]
        
        # 2. 重新生成 Summary (如果没有 VLM 实时调用，就简单的拼接 tags)
        # 理想情况：调用 VLM 重新总结。
        # 降级方案：聚合 tags
        def summarize(shot_list):
            if not shot_list: return "Fragment"
            tags = set()
            for s in shot_list: tags.update(s.get('tags', []))
            return " ".join(list(tags)[:5]) # 取前5个tag

        return {
            "movements": [
                {
                    "start": start, "end": split_time, 
                    "shots": shots_part1,
                    "summary": summarize(shots_part1)
                },
                {
                    "start": split_time, "end": end, 
                    "shots": shots_part2,
                    "summary": summarize(shots_part2)
                }
            ]
        }
    tb.register("struct.split", impl_split)

    def impl_merge(args: Dict) -> Dict:
        # Bridge 已经帮我们拿到了两个 movement 的信息
        start = args.get("start_time")
        end = args.get("next_end_time")
        
        shots_1 = args.get("shots", [])
        shots_2 = args.get("next_shots", [])
        all_shots = shots_1 + shots_2
        
        # 简单的文本合并
        s1 = args.get("visual_summary", "")
        s2 = args.get("next_summary", "")
        
        print(f"    >> [Tool:Merge] Merging {len(shots_1)} + {len(shots_2)} shots")

        return {
            "merged_movement": {
                "start": start, 
                "end": end, 
                "shots": all_shots,
                "summary": f"{s1} -> {s2}" # 用箭头表示情绪流转
            }
        }
    tb.register("struct.merge", impl_merge)

    tb.register("edit.continue", lambda a: {"track_id": "prev_track"})
    tb.register("retrieval.relax_constraint", lambda a: {"relaxed": True})
    tb.register("edit.shift_align", lambda a: {"shifted": True})

    return tb