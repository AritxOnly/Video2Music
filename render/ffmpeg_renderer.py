import os
import subprocess
import json
from typing import List, Dict, Any
from pathlib import Path

class FFmpegRenderer:
    def __init__(self):
        pass

    def render(self, 
               video_path: str, 
               audio_plan: List[Dict[str, Any]], 
               output_path: str):
        """
        根据音频计划合成最终视频。
        audio_plan 结构: 
        [
            {'start_time': 0.0, 'end_time': 10.0, 'file_path': 'a.mp3', 'source_start': 0.0, 'volume': 1.0, 'fade': 0.5},
            ...
        ]
        """
        video_path = str(Path(video_path).absolute())
        output_path = str(Path(output_path).absolute())
        
        print(f"[Renderer] Rendering {len(audio_plan)} audio clips to {output_path}...")

        # 1. 生成复杂的 FFmpeg Filter Complex
        # 这是一个多路混音逻辑：
        #   [0:a] atrim=... [a0];
        #   [1:a] atrim=... [a1];
        #   [a0][a1]... amix=inputs=N [out]
        
        inputs = []
        filter_complex = []
        mix_inputs = []
        
        # 输入 0 是视频文件 (我们只取视频流，忽略原声)
        inputs.append(f'-i "{video_path}"')
        
        # 处理每个音频片段
        for i, clip in enumerate(audio_plan):
            path = clip['file_path']
            # 安全检查
            if not os.path.exists(path):
                print(f"[Renderer] Warning: Audio file not found: {path}. Skipping.")
                continue
                
            inputs.append(f'-i "{path}"')
            input_idx = len(inputs) - 1 # 当前音频的输入索引
            
            # 计算时长
            duration = clip['end_time'] - clip['start_time']
            src_start = clip.get('source_start', 0.0)
            delay = clip['start_time'] * 1000 # 毫秒
            fade_len = clip.get('fade', 0.5)
            
            # 构造 Filter 链：
            # [idx] atrim=start:end, asetpts=PTS-STARTPTS, afade=in/out, adelay [out_idx]
            
            # 1. Trim & Reset PTS
            filter_chain = f"[{input_idx}:a]atrim={src_start}:{src_start+duration},asetpts=PTS-STARTPTS"
            
            # 2. Fade In/Out
            # afade=t=in:ss=0:d=0.5,afade=t=out:st={duration-0.5}:d=0.5
            filter_chain += f",afade=t=in:ss=0:d={fade_len}"
            # 注意：st 是相对于截取后片段的时间，即 duration - fade_len
            if duration > fade_len:
                filter_chain += f",afade=t=out:st={duration-fade_len}:d={fade_len}"
            
            # 3. Volume
            vol = clip.get('volume', 1.0)
            if vol != 1.0:
                 filter_chain += f",volume={vol}"

            # 4. Delay (定位到视频时间轴)
            # adelay 使用毫秒，且如果是立体声需要 "delay|delay"
            filter_chain += f",adelay={int(delay)}|{int(delay)}"
            
            label = f"a{i}"
            filter_chain += f"[{label}]"
            
            filter_complex.append(filter_chain)
            mix_inputs.append(f"[{label}]")
        
        if not mix_inputs:
            print("[Renderer] No audio inputs. Copying video only.")
            cmd = f'ffmpeg -y -i "{video_path}" -c copy "{output_path}"'
        else:
            # 5. Mix 混音
            # amix=inputs=N:dropout_transition=0,volume=N (为了防止混音后音量衰减)
            mix_cmd = "".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:dropout_transition=0,volume={len(mix_inputs)}[aout]"
            filter_complex.append(mix_cmd)
            
            filter_str = ";".join(filter_complex)
            
            # 6. 最终命令
            # -map 0:v (使用原视频画面) -map "[aout]" (使用合成音频)
            cmd = f'ffmpeg -y {" ".join(inputs)} -filter_complex "{filter_str}" -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k "{output_path}"'

        # 执行
        try:
            # print(f"CMD: {cmd}") # Debug Use
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if p.returncode != 0:
                print(p.stdout[-4000:])  # 打印末尾，别爆屏
                raise subprocess.CalledProcessError(p.returncode, cmd)
            else:
                # 可选：调试期也打印末尾几行，看有没有 atrim 警告
                tail = p.stdout.strip().splitlines()[-20:]
                print("\n".join(tail))
            print(f"[Renderer] Success! Saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[Renderer] FFmpeg failed: {e}")