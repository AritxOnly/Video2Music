from typing import Dict, Any, Optional, List, Union
from toolbox.interface import Toolbox
from mgen.service import MusicService

# 引入你的数据类，确保 Planner 拿到的是对象而不是字典
from agent.state import Track, Movement, SegmentSemantics

_MUSIC_SERVICE: Optional[MusicService] = None

def get_service(context: Dict[str, Any]) -> MusicService:
    global _MUSIC_SERVICE
    if _MUSIC_SERVICE is None:
        path = context.get('tracks_json_path')
        if not path:
            raise ValueError("Context missing 'tracks_json_path'")
        _MUSIC_SERVICE = MusicService(path)
    return _MUSIC_SERVICE

# === 核心适配器：将 Service 返回的 Dict 转为 Track 对象 ===
def _dict_to_track(d: Dict[str, Any]) -> Optional[Track]:
    """
    负责数据清洗：确保返回给 Agent 的是一个合法的 Track 对象，
    并且包含 Critic 必须的 duration, key, meta 字段。
    """
    if not d:
        return None

    # 1. 提取元数据
    # Service 返回的结构通常是 { "id":..., "meta": {...}, ... }
    # 有时 meta 是扁平的，这里做一下兼容
    meta = d.get("meta", {})
    if not meta:
        meta = d.copy() # fallback

    # 2. 关键字段提取 (防御性编程)
    track_id = str(d.get("track_id") or d.get("id") or "unknown")
    
    # 3. 时长处理 (Critic 的命门)
    # 优先从外层拿，没有从 meta 拿，还没有就设为 0 (Critic 会报警)
    duration = float(d.get("duration") or meta.get("duration") or 0.0)
    
    # 4. 构造对象
    return Track(
        id=track_id,
        source=str(d.get("source", "library")),
        duration=duration,
        play_start=float(d.get("play_start", 0.0)),
        play_duration=float(d.get("play_duration", 0.0)),
        meta=meta # 保留完整元数据供 Critic 使用 (key, sem_score, filepath)
    )

def register_real_tools(tb: Toolbox, context: Dict[str, Any] = {}):
    """
    注册真实工具。
    原则：
    1. 输入是 Dict (args)
    2. 输出是 Object (Track/Movement/List[Movement]) 或 None
    """
    
    service = get_service(context=context)
    
    # --- 1. 检索类工具 (返回 Track) ---

    def impl_search(args: Dict) -> Optional[Track]:
        query = args.get("query", "")
        summary = args.get("visual_summary", "")
        prev_key = args.get("prev_track_key")
        forbidden_ids = args.get("forbidden_ids", []) # 支持上一轮提到的“禁止列表”
        
        print(f"    >> [Tool:Search] Query='{query}' Key='{prev_key}'")
        
        # 调用 Service
        result_dict = service.smart_search(
            query=query, 
            visual_summary=summary, 
            prev_track_key=prev_key, 
            strict_harmony=True,
            forbidden_ids=forbidden_ids # 记得在 Service 加上这个参数
        )
        
        # 适配类型
        return _dict_to_track(result_dict)

    def impl_relax(args: Dict) -> Optional[Track]:
        print(f"    >> [Tool:Relax] Searching without constraints...")
        query = args.get("query", "")
        summary = args.get("visual_summary", "")
        prev_key = args.get("prev_track_key")
        
        result_dict = service.smart_search(
            query=query, 
            visual_summary=summary, 
            prev_track_key=prev_key,
            strict_harmony=False # 关闭和声约束
        )
        return _dict_to_track(result_dict)

    def impl_suno(args: Dict) -> Optional[Track]:
        prompt = args.get("prompt") or args.get("query", "music")
        duration = float(args.get("duration", 30.0))
        
        print(f"    >> [Tool:Suno] Generating '{prompt}' ({duration}s)")
        
        result_dict = service.generate_music(prompt, duration)
        return _dict_to_track(result_dict)

    # --- 2. 结构调整类工具 (返回 Movement) ---

    def impl_split(args: Dict) -> Optional[List[Movement]]:
        """
        将一个 movement 切分为两个。
        """
        mov_id = args.get("movement_id", "mov")
        start = float(args.get("start_time"))
        end = float(args.get("end_time"))
        split_time = float(args.get("timestamp"))
        
        # 获取原始 shots (注意：这里假设 Bridge 传过来的是对象列表)
        # 如果 Bridge 传的是 Dict List，这里需要做兼容
        original_shots = args.get("shots", [])
        
        print(f"    >> [Tool:Split] {mov_id} at {split_time:.2f}s")
        
        # 辅助函数：处理 shot 对象的属性访问
        def get_end(s):
            return s.end_sec if hasattr(s, 'end_sec') else s.get('end_sec', 0)

        # 1. 切分 Shots
        shots_part1 = [s for s in original_shots if get_end(s) <= split_time]
        shots_part2 = [s for s in original_shots if get_end(s) > split_time]
        
        # 2. 简单的 Summary 生成 (生产环境应调用 VLM)
        def simple_summ(shots):
            if not shots: return "Fragment"
            # 尝试提取 tags
            tags = []
            for s in shots:
                t = s.tags if hasattr(s, 'tags') else s.get('tags', [])
                if isinstance(t, list): tags.extend(t)
                elif isinstance(t, str): tags.append(t)
            return " ".join(list(set(tags))[:4])

        # 3. 构造 Movement 对象
        m1 = Movement(
            id=f"{mov_id}_a",
            start_time=start,
            end_time=split_time,
            shots=shots_part1,
            visual_summary=simple_summ(shots_part1)
        )
        
        m2 = Movement(
            id=f"{mov_id}_b",
            start_time=split_time,
            end_time=end,
            shots=shots_part2,
            visual_summary=simple_summ(shots_part2)
        )
        
        return [m1, m2]

    def impl_merge(args: Dict) -> Optional[Movement]:
        """
        将当前 movement 和下一个 movement 合并。
        """
        mov_id = args.get("movement_id", "mov")
        start = float(args.get("start_time"))
        end = float(args.get("next_end_time")) # Bridge 需注入此参数
        
        shots_1 = args.get("shots", [])
        shots_2 = args.get("next_shots", [])
        all_shots = shots_1 + shots_2
        
        s1 = args.get("visual_summary", "")
        s2 = args.get("next_summary", "")
        
        print(f"    >> [Tool:Merge] Merging {len(shots_1)} + {len(shots_2)} shots")
        
        merged = Movement(
            id=f"{mov_id}_merged",
            start_time=start,
            end_time=end,
            shots=all_shots,
            visual_summary=f"{s1} -> {s2}"
        )
        return merged

    # --- 3. 注册工具 ---
    
    tb.register("retrieval.search", impl_search)
    tb.register("retrieval.requery", impl_search) # Requery 复用 Search 逻辑
    tb.register("retrieval.relax_constraint", impl_relax)
    tb.register("gen.suno", impl_suno)
    
    tb.register("struct.split", impl_split)
    tb.register("struct.merge", impl_merge)
    
    # [关键修改] 移除 edit.continue
    # CONTINUE 是 Planner 内部的状态流转，不是 Tool 的职责。
    # 如果强行注册为空 lambda，会误导 Planner 以为执行成功并返回了 None/Dict，导致后续逻辑中断。
    
    # tb.register("edit.shift_align", impl_shift_align) # 暂未实现

    return tb