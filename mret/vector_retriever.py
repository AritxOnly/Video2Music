from typing import List, Dict, Any
import json
import torch
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

class VectorMusicRetriever:
    """
    [Module 3.3] 基于 BGE 语义向量的音乐检索器。
    将 'Mood' 文本映射到 'Music Tags' 向量空间。
    """
    def __init__(self, tracks_json_path: str):
        self.json_path = Path(tracks_json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Music library not found at {tracks_json_path}")

        # 1. 加载数据
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.tracks = json.load(f)
            
        if not self.tracks:
            raise ValueError(f"Music library is empty!")

        # 2. 加载模型 (与 VSEM 保持一致，复用 BGE-Small)
        # 注意：这里我们假设外部环境已经准备好了模型，或者首次运行会自动下载
        print(f">>> [MRet] Loading Embedding Model (bge-small-zh)...")
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        
        # 3. 构建索引 (Offline Indexing)
        print(f">>> [MRet] Indexing {len(self.tracks)} tracks...")
        self.track_vectors = self._build_index()

    def _build_index(self):
        """将每首歌的 tags + style 拼接成由空格分隔的文本，计算 Embedding"""
        descriptions = []
        for t in self.tracks:
            # 构造文档: "cinematic 壮观 史诗 宏大..."
            # 增加 style 的权重（放在前面）
            tags_str = " ".join(t.get('tags', []))
            desc = f"{t.get('style', '')} {tags_str}"
            descriptions.append(desc)
            
        # 批量编码，转换为 Tensor 以便计算
        return self.model.encode(descriptions, convert_to_tensor=True)

    def search(self, query_text: str, top_k: int = 5) -> List[Dict]: # 默认扩大搜索范围
        """返回 Top-K 候选，供下游做乐理筛选"""
        if not query_text: return [self.tracks[0]]

        query_vec = self.model.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(query_vec, self.track_vectors)[0]
        
        # 获取 Top-K
        k = min(top_k, len(self.tracks))
        top_results = torch.topk(scores, k=k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            track = self.tracks[idx.item()].copy()
            track["match_score"] = score.item()
            results.append(track)
            
        return results