from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import cv2  # 需要用到opencv读取fps

from vlm import get_vlm, VideoInput, AnalysisOptions, TaskType
from vsem.builder import build_from_vlm
from vsem.model import VideoSemantics
from mgen import JsonMusicLibrary, SimpleRuleArranger, MGenOptions
from render.ffmpeg_renderer import render_with_bgm
from clip import detect_shot_changes, cut_video, generate_timeline

def get_video_meta(video_path: str):
    """辅助函数：获取视频 FPS 和总时长"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, frame_count

def run_video2music(
    backend: str,
    api_key: str,
    video_path: str,
    tracks_json: str,
    output_path: str,
    lang: str = "zh",
    music_style: Optional[str] = None,
) -> Dict[str, Any]:
    
    # 0. 路径与元数据准备
    video_abs = str(Path(video_path).expanduser().absolute())
    tracks_json_abs = str(Path(tracks_json).expanduser().absolute())
    output_abs = str(Path(output_path).expanduser().absolute())
    
    fps, _ = get_video_meta(video_abs)
    print(f">>> Meta: Video FPS is {fps}")

    # 1. 物理层：检测与切分
    print(">>> Step 1: Detecting shot changes...")
    shot_changes = detect_shot_changes(video_path=video_abs)
    
    # 【关键改进】先生成全局时间线注册表
    # 这份 timeline 是我们的“绝对真理”，包含了准确的 start_sec, end_sec 和预期的 filename
    global_timeline_registry = generate_timeline(shot_changes, fps)
    
    print(">>> Step 2: Cutting video based on registry...")
    output_dir = Path(output_abs).parent / "temp_clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 注意：你需要确保 cut_video 内部生成的命名逻辑和 generate_timeline 里的 f"clip_{i}.mp4" 是一致的
    # 如果 cut_video 还没改，建议传入 global_timeline_registry 让它照着生成
    cut_video(video_path=video_abs, shot_changes=shot_changes, output_dir=output_dir)

    # 2. 语义层：VLM 分析
    vlm = get_vlm(backend, api_key=api_key)
    all_segment_semantics = []

    print(f">>> Step 3: Analyzing {len(global_timeline_registry)} segments...")
    
    # 【关键改进】不再遍历文件夹，而是遍历注册表
    for event in global_timeline_registry:
        # 从注册表中获取预期文件名和绝对时间
        clip_filename = event["filename"]  # e.g., "clip_0.mp4"
        abs_start = event["start_sec"]
        abs_end = event["end_sec"]
        
        clip_file_path = output_dir / clip_filename
        
        if not clip_file_path.exists():
            print(f"[Warn] Missing clip file: {clip_filename}, skipping...")
            continue

        # 2.1 VLM 视觉分析
        vinput = VideoInput(path=str(clip_file_path))
        # 针对短片段，可能不再需要 need_timeline=True，除非你要做片段内的动作精细定位
        # 但为了获取 mood/tags，Structure 和 Tagging 依然必要
        struct_res = vlm.analyze(vinput, AnalysisOptions(task=TaskType.STRUCTURE, language=lang))
        tag_res = vlm.analyze(vinput, AnalysisOptions(task=TaskType.TAGGING, language=lang))

        # 2.2 构建语义对象
        semantics_obj = build_from_vlm(struct_res, tag_res)
        
        # 2.3 【核心逻辑】时间戳对齐
        # 我们不仅是 append，而是要用注册表的绝对时间“校准”VLM的分析结果
        # 这里假设 semantics_obj.segments 里的片段是对当前 clip 的描述
        # 我们把它们强行映射到全局时间轴上
        
        for seg in semantics_obj.segments:
            # 这里的逻辑取决于：你是把整个 clip 当作一个原子事件，还是 clip 里还有细分事件？
            # 方案 A：如果 clip 很短（比如镜头切换），通常只有一个主事件
            seg.start_sec = abs_start
            seg.end_sec = abs_end
            
            # 方案 B：如果 VLM 返回了相对时间（比如 clip 的第 1s 发生爆炸），则：
            # seg.start_sec = abs_start + vlm_relative_start 
            
            # 将“注册表”里的一些元数据也可以塞进去 (optional)
            seg.extra['shot_label'] = event['label'] 
            
            all_segment_semantics.append(seg)
            
        print(f"    - Analyzed {clip_filename}: Global {abs_start:.2f}s -> {abs_end:.2f}s | Mood: {semantics_obj.segments[0].mood if semantics_obj.segments else 'N/A'}")

    # 3. 聚合层：语义融合 (即将进行的步骤)
    # 目前先简单聚合
    global_semantics = VideoSemantics(
        global_tags=[], 
        segments=all_segment_semantics
    )

    # 4. 编排与渲染 (保持不变)
    print(">>> Step 4: Arranging & Rendering...")
    lib = JsonMusicLibrary(tracks_json_abs)
    arranger = SimpleRuleArranger()
    mopts = MGenOptions(preferred_style=music_style, global_gain_db=-6.0, crossfade_sec=0.3)

    plans = arranger.arrange(timeline=global_semantics.segments, library=lib, options=mopts)
    
    track_map = {t.id: t for t in lib.list_tracks(style=music_style) or lib.list_tracks()}
    render_with_bgm(video_path=video_abs, plans=plans, track_map=track_map, output_path=output_abs)

    return {
        "semantics": global_semantics,
        "plans": plans,
        "video_in": video_abs,
        "video_out": output_abs,
    }