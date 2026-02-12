import os
# === 1. 资源限制 (防卡死) ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
import torchaudio
import librosa
import numpy as np
import gc
from pathlib import Path
from typing import List, Dict, Any
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

class HybridIndexer:
    def __init__(self, assets_dir: str, index_file: str):
        self.assets_dir = Path(assets_dir)
        self.index_file = Path(index_file)
        
        # === Device ===
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f">>> [Indexer] Loading CLAP on {self.device}...")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model.eval()

        # === Anchors (语义锚点) ===
        self.va_anchors = {
            "val_pos": ["happy music", "joyful", "positive"],
            "val_neg": ["sad music", "depressing", "negative"],
            "aro_high": ["energetic", "intense", "fast"],
            "aro_low": ["calm", "slow", "relaxing"]
        }
        
        self.candidate_tags = [
            "epic", "cinematic", "action", "battle", 
            "happy", "upbeat", "pop", "dance",        
            "sad", "emotional", "piano", "melancholy",
            "calm", "ambient", "nature", "lofi", "jazz"
        ]
        
        self.text_embeds = self._precompute_text_embeds()

    def _precompute_text_embeds(self):
        print(">>> [Indexer] Pre-computing semantic anchors...")
        embeds = {}
        with torch.no_grad():
            for key, texts in self.va_anchors.items():
                inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
                embed = self.model.get_text_features(**inputs).mean(dim=0)
                embeds[key] = embed / embed.norm(dim=-1, keepdim=True)
            
            inputs = self.processor(text=[f"{t} music" for t in self.candidate_tags], return_tensors="pt", padding=True).to(self.device)
            tag_embeds = self.model.get_text_features(**inputs)
            embeds["tags"] = tag_embeds / tag_embeds.norm(dim=-1, keepdim=True)
        return embeds

    def scan_and_index(self):
        files = list(self.assets_dir.glob("**/*.mp3")) + list(self.assets_dir.glob("**/*.wav"))
        print(f">>> [Indexer] Processing {len(files)} files...")
        
        new_tracks = []
        for i, fpath in enumerate(tqdm(files, desc="Indexing", unit="track")):
            rel = str(fpath.relative_to(Path(".").absolute()) if fpath.is_absolute() else fpath)
            
            try:
                meta = self._analyze_track(str(fpath))
                meta['filepath'] = rel
                meta['id'] = fpath.stem.replace(" ", "_").lower()
                new_tracks.append(meta)
            except Exception as e:
                # 遇到错误不中断，只打印
                if i < 5: tqdm.write(f"Skip {fpath.name}: {e}")
                pass
            
            # 定期清理显存/内存
            if i % 25 == 0:
                if self.device == "mps": torch.mps.empty_cache()
                gc.collect()

        print(f">>> [Indexer] Saving {len(new_tracks)} tracks to JSON...")
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(new_tracks, f, ensure_ascii=False, indent=2)
        print(f">>> [Indexer] Done.")

    def _analyze_track(self, fpath: str) -> Dict[str, Any]:
        """
        核心分析：Multi-Window CLAP + Physical Feature Fusion (Duration, BPM, Key)
        """
        # === 1. 音频读取 & 时长计算 ===
        waveform = None
        sr = 48000
        total_duration = 0.0

        try:
            # torchaudio 加载
            waveform, original_sr = torchaudio.load(fpath, normalize=True)
            
            if original_sr != 48000:
                resampler = torchaudio.transforms.Resample(original_sr, 48000)
                waveform = resampler(waveform)
            
            # [CRITICAL FIX] 截断前计算真实时长
            total_duration = waveform.shape[1] / sr
            
        except Exception:
            # Fallback to librosa
            y_np, _ = librosa.load(fpath, sr=48000) 
            waveform = torch.from_numpy(y_np)
            if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
            
            # [CRITICAL FIX] 截断前计算真实时长
            total_duration = waveform.shape[1] / sr
            
        if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
        
        # 为了分析效率，截取前 90s (不影响 total_duration)
        max_frames = 90 * sr
        if waveform.shape[1] > max_frames: waveform = waveform[:, :max_frames]
        
        y_np = waveform.squeeze().numpy()
        
        # === 2. 物理特征 ===
        # A. 能量 (RMS)
        rms = librosa.feature.rms(y=y_np)[0]
        if len(rms) == 0: return self._get_empty_meta()
            
        phys_energy = np.mean(rms)
        phys_arousal = np.clip((phys_energy - 0.02) / 0.15, 0.0, 1.0)
        
        # B. 亮度 (Spectral Centroid)
        try:
            cent = librosa.feature.spectral_centroid(y=y_np, sr=sr)[0]
        except: pass
        
        # C. 调性 (Key)
        y_30s = y_np[:30*sr] # 取前30秒算调性足够了
        key_name = "Unknown"
        if len(y_30s) > 0:
            try:
                chroma = librosa.feature.chroma_cqt(y=y_30s, sr=sr)
                key_name = self._detect_key(np.mean(chroma, axis=1))
            except: pass
        
        phys_valence_bias = 0.2 if "Major" in key_name else -0.2

        # D. [新增] 节奏 (BPM)
        bpm_val = 0.0
        try:
            # 计算 onset envelope
            onset_env = librosa.onset.onset_strength(y=y_np, sr=sr)
            # 计算 tempo
            tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
            
            # 兼容性处理：librosa 新旧版本返回值可能是 float 或 array
            if np.ndim(tempo) == 0:
                bpm_val = float(tempo)
            else:
                bpm_val = float(tempo[0]) if len(tempo) > 0 else 0.0
        except Exception:
            pass

        # === 3. 语义特征 (Multi-Window CLAP) ===
        windows = []
        total_frames = waveform.shape[1]
        for start_sec in [0, 30, 60]:
            start_frame = start_sec * sr
            end_frame = (start_sec + 7) * sr
            if end_frame < total_frames:
                windows.append(waveform[:, start_frame:end_frame])
        
        if not windows: windows.append(waveform)

        # 批量推理
        window_embeds = []
        with torch.no_grad():
            for w in windows:
                audio_input = w.squeeze().numpy()
                inputs = self.processor(audio=audio_input, sampling_rate=48000, return_tensors="pt", padding=True).to(self.device)
                emb = self.model.get_audio_features(**inputs)
                window_embeds.append(emb / emb.norm(dim=-1, keepdim=True))
        
        if not window_embeds: return self._get_empty_meta()

        avg_audio_embed = torch.stack(window_embeds).mean(dim=0)
        avg_audio_embed = avg_audio_embed / avg_audio_embed.norm(dim=-1, keepdim=True)

        # === 4. 融合计算 ===
        sem_val = (avg_audio_embed @ self.text_embeds["val_pos"].t()).item() - (avg_audio_embed @ self.text_embeds["val_neg"].t()).item()
        sem_aro = (avg_audio_embed @ self.text_embeds["aro_high"].t()).item() - (avg_audio_embed @ self.text_embeds["aro_low"].t()).item()
        
        final_arousal = 0.6 * (sem_aro * 5.0) + 0.4 * (phys_arousal * 2.0 - 1.0)
        final_valence = 0.8 * (sem_val * 5.0) + 0.2 * phys_valence_bias

        final_valence = max(-1.0, min(1.0, final_valence))
        final_arousal = max(-1.0, min(1.0, final_arousal))

        # Tags
        tag_scores = (avg_audio_embed @ self.text_embeds["tags"].t()).squeeze()
        top_indices = tag_scores.topk(3).indices.tolist()
        predicted_tags = [self.candidate_tags[i] for i in top_indices]
        
        if len(rms) > 0:
            chorus_start = float(librosa.frames_to_time(np.argmax(rms), sr=sr))
        else:
            chorus_start = 0.0

        # === 5. 返回结果 ===
        return {
            "duration": float(round(total_duration, 2)), # 真实时长
            "bpm": int(round(bpm_val)),                  # [新增] BPM 取整
            "key": str(key_name),
            "style": str(predicted_tags[0]),
            "tags": predicted_tags,
            "valence": float(round(final_valence, 3)),
            "arousal": float(round(final_arousal, 3)),
            "chorus_start": float(round(chorus_start, 2))
        }

    def _detect_key(self, chroma_avg):
        major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        chroma_avg = (chroma_avg - chroma_avg.min()) / (chroma_avg.max() - chroma_avg.min())
        maj_corrs = [np.corrcoef(chroma_avg, np.roll(major, i))[0,1] for i in range(12)]
        min_corrs = [np.corrcoef(chroma_avg, np.roll(minor, i))[0,1] for i in range(12)]
        pcs = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
        return f"{pcs[np.argmax(maj_corrs)]} Major" if max(maj_corrs) > max(min_corrs) else f"{pcs[np.argmax(min_corrs)]} Minor"

    def _get_empty_meta(self):
        return {
            "duration": 0.0, "bpm": 0, "key": "Unknown", "style": "Unknown",
            "tags": [], "valence": 0.0, "arousal": 0.0, "chorus_start": 0.0
        }

if __name__ == "__main__":
    indexer = HybridIndexer("assets/music", "mgen/tracks.auto.json")
    indexer.scan_and_index()