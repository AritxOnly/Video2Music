from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Any
import cv2
import json
from dataclasses import asdict

from vlm import (
    VideoInput, AnalysisOptions, TaskType,
    get_vlm, global_sampler, TimelineEvent
)
from vsem import assemble_video_semantics
from mgen import JsonMusicLibrary, SimpleRuleArranger, MGenOptions
from render import render_with_bgm
from clip import detect_shot_changes, cut_video, generate_timeline

def get_video_frame_data(video_path: str) -> tuple[float, float]:
    """辅助工具：获取视频FPS"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (fps if fps > 0 else 30.0, total_frames)

def _save_cache(events: List[TimelineEvent], path: Path):
    """将 TimelineEvent 列表序列化为 JSON 保存"""
    print(f"    [Cache] Saving VLM results to {path.name}...")
    data = [asdict(e) for e in events]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_cache(path: Path) -> List[TimelineEvent]:
    """从 JSON 加载 TimelineEvent 列表"""
    print(f"    [Cache] Loading VLM results from {path.name}...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 重建 Dataclass 对象
    events = []
    for item in data:
        # 兼容性处理：如果 JSON 里没有 extra，给个默认空字典
        if 'extra' not in item: item['extra'] = {}
        events.append(TimelineEvent(**item))
    return events

def run_video2music(
    backend: str, 
    api_key: str,
    video_path: str,
    tracks_json: str,
    output_path: str,
    lang: str = "zh",
    music_style: Optional[str] = None,
    # === 新增参数 ===
    fusion_strategy: str = "ai",  # "ai" or "rule"
    use_cache: bool = False       # 是否使用缓存
) -> Dict[str, Any]:
    
    # 0. 准备路径
    video_abs = str(Path(video_path).expanduser().absolute())
    tracks_json_abs = str(Path(tracks_json).expanduser().absolute())
    output_abs = str(Path(output_path).expanduser().absolute())
    
    # 定义缓存文件路径: output.mp4 -> output.mp4.cache.json
    cache_path = Path(output_abs + ".cache.json")
    
    print(f"\n====== Video2Music Agent Pipeline V2.1 (Frozen Mode Supported) ======")
    print(f"Input: {video_abs}")
    print(f"Strategy: {fusion_strategy.upper()} | Cache Mode: {use_cache}")

    collected_events = []

    # === 核心逻辑：缓存判断 ===
    if use_cache and cache_path.exists():
        print(f"\n>>> [FROZEN MODE] Skipping Step 1-3 (CV & VLM), loading from cache...")
        collected_events = _load_cache(cache_path)
        print(f"    Loaded {len(collected_events)} events from disk.")
        
    else:
        # === 没命中缓存，老老实实跑全流程 ===
        
        # 1. [CV层] 生成时间线注册表
        print("\n>>> Step 1: CV Analysis (Shot Detection)...")
        fps, total = get_video_frame_data(video_abs)
        shot_changes = detect_shot_changes(video_path=video_abs)
        registry = generate_timeline(shot_changes, fps, total_frames=int(total))
        print(f"    Detected {len(registry)} shots.")

        # 2. [物理层] 切割视频
        print("\n>>> Step 2: Physical Cutting...")
        temp_clips_dir = Path(output_abs).parent / "temp_clips_processing"
        temp_clips_dir.mkdir(parents=True, exist_ok=True)
        cut_video(video_path=video_abs, shot_changes=shot_changes, output_dir=temp_clips_dir)

        # 3. [语义层] 分段 VLM 分析
        print("\n>>> Step 3: Semantic Analysis (Segment VLM)...")
        
        # 强制使用我们写好的 Segment Interface
        vlm_segment_runner = get_vlm(name='qwen-seg-web', api_key=api_key)
        
        for i, item in enumerate(registry):
            clip_filename = item["filename"]
            abs_start = item["start_sec"]
            abs_end = item["end_sec"]
            clip_path = temp_clips_dir / clip_filename
            
            if not clip_path.exists():
                continue

            print(f"    [{i+1}/{len(registry)}] Analyzing {clip_filename} ({abs_start:.1f}s - {abs_end:.1f}s)...")

            # 构造输入
            vinput = VideoInput(path=str(clip_path))
            res = vlm_segment_runner.analyze(
                vinput, 
                AnalysisOptions(task=TaskType.TAGGING, language=lang, need_timeline=False)
            )
            
            if res.timeline:
                vlm_event = res.timeline[0] 
                # 注入绝对时间与元数据
                vlm_event.start_sec = abs_start
                vlm_event.end_sec = abs_end
                vlm_event.extra['shot_id'] = i
                vlm_event.extra['original_filename'] = clip_filename
                
                collected_events.append(vlm_event)
                activity = vlm_event.extra.get('activity', 'N/A')
                print(f"      -> Mood: {vlm_event.label} | Activity: {activity}")

        # 清理图片
        global_sampler.cleanup()
        
        # === 保存缓存 ===
        if collected_events:
            _save_cache(collected_events, cache_path)

    # 4. [聚合层] 上下文组装 (支持策略切换)
    print(f"\n>>> Step 4: Context Assembly (Strategy: {fusion_strategy})...")
    # 这里透传 strategy 参数给 vsem
    video_semantics = assemble_video_semantics(collected_events, strategy=fusion_strategy)

    # 5. [生成层] 音乐编排
    print("\n>>> Step 5: Music Arrangement...")
    lib = JsonMusicLibrary(tracks_json_abs)
    arranger = SimpleRuleArranger()
    mopts = MGenOptions(preferred_style=music_style, global_gain_db=-6.0, crossfade_sec=0.5)

    plans = arranger.arrange(timeline=video_semantics.segments, library=lib, options=mopts)
    print(f"    Generated {len(plans)} music cues.")

    # 6. [渲染层] 合成
    print("\n>>> Step 6: Rendering Final Video...")
    
    with open(tracks_json_abs, 'r') as f:
        raw_tracks = json.load(f)
    
    from collections import namedtuple
    SimpleTrack = namedtuple('SimpleTrack', ['id', 'filepath'])
    
    track_map = {}
    for t in raw_tracks:
        track_map[t['id']] = SimpleTrack(id=t['id'], filepath=t['filepath'])
    
    render_with_bgm(
        video_path=video_abs,
        plans=plans,
        track_map=track_map,
        output_path=output_abs,
    )

    print(f"\n====== Done! Output saved to: {output_abs} ======")
    
    return {
        "semantics": video_semantics,
        "plans": plans,
        "video_in": video_abs,
        "video_out": output_abs
    }