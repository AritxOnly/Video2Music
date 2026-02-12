import os
import sys
import json
import cv2
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from dotenv import load_dotenv

from utils.logger import setup_logging

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.append(os.getcwd())

# === 模块导入 ===
# VLM 相关
from agent.macro_planner import MacroPlanner
from vlm import get_vlm, VideoInput, AnalysisOptions, TaskType, TimelineEvent, global_sampler

# MGen & Agent 相关
from mgen.service import MusicService
from agent.state import AgentState, Movement, SegmentSemantics, Track
from agent.critic import Critic
from agent.micro_planner import MicroPlanner
from agent.action_generator import ActionGenerator

# Toolbox 相关
from toolbox.interface import Toolbox
from toolbox.bridge import AgentToolboxBridge
from toolbox.impl import register_real_tools
from llm import get_llm

from render.ffmpeg_renderer import FFmpegRenderer
from clip import detect_shot_changes, cut_video, generate_timeline

DISABLE_MACRO = True

# === 辅助函数 ===
def get_video_frame_data(video_path: str) -> tuple[float, float]:
    """获取视频FPS和总帧数"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 30.0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (fps if fps > 0 else 30.0, total_frames)

def _save_cache(events: List[TimelineEvent], path: Path):
    """序列化保存 VLM 结果"""
    print(f"    [Cache] Saving VLM results to {path.name}...")
    data = [asdict(e) for e in events]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class Video2MusicAgent:
    def __init__(self, 
                 tracks_path: str,
                 toolbox_config: str = "toolbox/toolbox.json"):
        
        # 1. Init Infrastructure
        print("[Init] Loading Tools & Service...")
        if not os.path.exists(toolbox_config):
             # 确保目录存在
             os.makedirs(os.path.dirname(toolbox_config), exist_ok=True)
             with open(toolbox_config, 'w') as f: json.dump({"tools": []}, f)
             
        self.raw_toolbox = Toolbox(toolbox_config)
        register_real_tools(self.raw_toolbox, context={'tracks_json_path': tracks_path})    # TODO: implements real tools
        self.bridge = AgentToolboxBridge(self.raw_toolbox)
        
        # 2. Init Agent Brain
        print("[Init] Loading LLM (DeepSeek)...")
        self.llm = get_llm(name='deepseek-v3')
        self.action_gen = ActionGenerator(self.llm)
        self.critic = Critic() # TODO: v1: sem/harm/cost stable; sync/flow requires cut_times & visual_energy
        self.macro_planner = MacroPlanner(self.llm)
        self.planner = MicroPlanner(self.action_gen, self.critic, self.bridge)
        
        # 3. Init Renderer
        self.renderer = FFmpegRenderer() # TODO: fix rendering bugs
        
        # ======= Initialize review finished =======

    def run(self, video_path: str, output_path: str, use_cache: bool = True, lang: str = "zh"):
        # TODO: migrate logic
        video_path = str(Path(video_path).absolute())
        cache_path = Path(video_path + ".cache.json")
        output_abs = str(Path(output_path).absolute())
        
        # === Phase 1: Perception (VLM) ===
        movements = []
        if use_cache and cache_path.exists():
            print(f"\n>>> [Phase 1] Loading Perception from Cache: {cache_path.name}")
            movements = self._load_cache(str(cache_path))
        else:
            print(f"\n>>> [Phase 1] Running Perception (Shot Detection -> VLM)...")
            movements = self._run_perception(video_path, str(cache_path), lang=lang)

        if not movements:
            print("[Error] Perception failed. No movements generated.")
            return
        
        raw_movements = movements # VLM 出来的原始结果
        if not raw_movements: return

        # === Phase 1.5: Macro Planning (Structure) ===
        print(f"\n>>> [Phase 1.5] Macro Planning (Grouping {len(raw_movements)} raw shots)...")
        
        # 调用 MacroPlanner 进行聚合
        # 你可以根据视频总时长调整 min_duration，比如总长300s，每段至少20s
        structured_movements = self.macro_planner.plan(raw_movements, min_duration=15.0)
        
        if not structured_movements or DISABLE_MACRO:
            print("[Error] Macro planning failed. Using raw movements.")
            structured_movements = raw_movements

        # === Phase 2: Micro Planning (Content) ===
        print(f"\n>>> [Phase 2] Agent Planning ({len(structured_movements)} grouped movements)...")
        
        # 使用聚合后的 movements 初始化状态
        initial_state = AgentState(movements=structured_movements)
        final_state = self.planner.plan(initial_state)
        
        print(f"Plan Completed. Total Score: {final_state.total_score:.2f}")

        # === Phase 3: Rendering ===
        print(f"\n>>> [Phase 3] Rendering Final Video...")
        audio_plan = self._convert_state_to_plan(final_state)
        
        if not audio_plan:
            print("[Warning] No audio plan generated. Skipping render.")
            return
        
        print(f'Audio Plan >>> \n', audio_plan)

        self.renderer.render(video_path, audio_plan, output_abs)
        
        print(f"\n=== ALL DONE. Output saved to: {output_abs} ===")

    def _run_perception(self, video_path: str, cache_path_str: str, lang: str = "zh") -> list[Movement]:
        """
        运行完整的感知流：Shot Detect -> VLM -> Movement Generation
        """
        video_abs = str(Path(video_path).absolute())
        cache_path = Path(cache_path_str)
        
        # 1. [CV层] 生成时间线注册表
        print("  Running Shot Detection...")
        fps, total = get_video_frame_data(video_abs)
        shot_changes = detect_shot_changes(video_path=video_abs) # TODO: 加入光流分析
        registry = generate_timeline(shot_changes, fps, total_frames=int(total))
        print(f"    Detected {len(registry)} shots.")

        # 2. [物理层] 切割视频 (用于 VLM 分析)
        print("  Physical Cutting (Creating temp clips)...")
        # 临时目录设为 output 同级目录下的 temp_clips
        temp_clips_dir = Path(video_abs).parent / "temp_clips_processing"
        temp_clips_dir.mkdir(parents=True, exist_ok=True)
        
        cut_video(video_path=video_abs, shot_changes=shot_changes, output_dir=temp_clips_dir)

        # 3. [语义层] 分段 VLM 分析
        print("  Semantic Analysis (Segment VLM)...")
        
        # 初始化 VLM Runner
        vlm_segment_runner = get_vlm(name='qwen-seg-web')
        collected_events = []

        for i, item in enumerate(registry):
            clip_filename = item["filename"]
            abs_start = item["start_sec"]
            abs_end = item["end_sec"]
            clip_path = temp_clips_dir / clip_filename
            
            if not clip_path.exists():
                print(f"    [Warning] Clip missing: {clip_path}")
                continue

            print(f"    [{i+1}/{len(registry)}] Analyzing {clip_filename} ({abs_start:.1f}s - {abs_end:.1f}s)...")

            # 构造输入
            vinput = VideoInput(path=str(clip_path))
            try:
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
                else:
                    print("      -> No timeline result from VLM.")
                    
            except Exception as e:
                print(f"      [Error] VLM analysis failed for shot {i}: {e}")

        # 清理全局采样器缓存
        global_sampler.cleanup()
        
        # 4. 保存缓存
        if collected_events:
            _save_cache(collected_events, cache_path)
            
            # 转换 TimelineEvent 为 Movement 对象
            return self._timeline_events_to_movements(collected_events)
        else:
            return []

    def _load_cache(self, path: str) -> list[Movement]:
        """从缓存文件加载并转换为 Movement 列表"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        events = []
        for item in data:
            if 'extra' not in item: item['extra'] = {}
            events.append(TimelineEvent(**item))
            
        return self._timeline_events_to_movements(events)

    def _timeline_events_to_movements(self, events: List[TimelineEvent]) -> list[Movement]:
        """将 VLM 的 TimelineEvent 转换为 Agent 的 Movement 对象"""
        movements = []
        for i, event in enumerate(events):
            start = event.start_sec
            end = event.end_sec
            label = event.label  # Mood
            
            # 构造丰富的 Visual Summary
            summary = event.description
            if not summary:
                extra = event.extra
                tags = extra.get('tags', [])
                act = extra.get('activity', '')
                tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
                summary = f"Mood: {label}. Activity: {act}. Visual tags: {tags_str}"

            # 构造 Movement
            # 默认每个 Event 对应一个初始 Movement
            mov = Movement(
                id=f"mov_{i}",
                start_time=start,
                end_time=end,
                visual_summary=summary,
                # 注入 Shots 信息，方便 ActionGenerator 做 SPLIT 决策
                shots=[SegmentSemantics(start, end, [], label, "")]
            )
            movements.append(mov)
        return movements

    def _convert_state_to_plan(self, state: AgentState) -> list[dict]:
        """将 AgentState 转换为 Renderer 需要的 Audio Plan"""
        plan = []
        
        # 遍历 assigned_tracks 生成渲染计划
        for idx, track in state.assigned_tracks.items():
            if idx >= len(state.movements): continue
            
            mov = state.movements[idx]
            
            # [关键逻辑] 计算 CONTINUE 模式下的 source_start
            current_track_base_play_start = track.play_start # 歌曲本身的起始点 (e.g. chorus start)
            
            # 回溯查找这首歌是从哪个 movement 开始被 assign 的
            start_mov_idx = idx
            while start_mov_idx > 0:
                prev_track = state.assigned_tracks.get(start_mov_idx - 1)
                # 判断是不是同一首歌的延续 (简单判断 track_id)
                if prev_track and prev_track.id == track.id:
                    start_mov_idx -= 1
                else:
                    break
            
            # 计算从 start_mov_idx 到当前 idx 之前所有 movements 的时长总和
            time_elapsed = 0.0
            for k in range(start_mov_idx, idx):
                m = state.movements[k]
                time_elapsed += (m.end_time - m.start_time)
            
            final_source_start = current_track_base_play_start + time_elapsed
            
            item = {
                "start_time": mov.start_time,
                "end_time": mov.end_time,
                "file_path": track.meta.get('filepath'),
                "source_start": final_source_start,
                "volume": 1.0, 
                "fade": 0.5    # 默认淡入淡出
            }
            plan.append(item)
            
        return plan

if __name__ == "__main__":
    logger = setup_logging(log_dir="logs", prefix="video2music")

    TRACKS = "mgen/tracks.auto.json"
    VIDEO = "z___outputs/result_v2.mp4"
    OUTPUT = "final_output_agent_v3.mp4"

    try:
        agent = Video2MusicAgent(tracks_path=TRACKS)
        agent.run(video_path=VIDEO, output_path=OUTPUT, use_cache=True, lang="zh")
    except Exception as e:
        logger.exception(f"[Fatal] Pipeline crashed: {e}")
        raise