from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .interface import MusicLibraryInterface
from .model import MusicTrack


class JsonMusicLibrary(MusicLibraryInterface):
    """
    从一个 JSON 文件加载音乐库。
    JSON 格式见 tracks.example.json。
    """

    def __init__(self, json_path: str | Path, base_dir: str | Path | None = None):
        self._json_path = Path(json_path)
        self._base_dir = Path(base_dir) if base_dir is not None else self._json_path.parent
        self._tracks: List[MusicTrack] = []
        self._load()

    def _load(self) -> None:
        with open(self._json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracks: List[MusicTrack] = []
        for item in data:
            path = item.get("path")
            if path is None:
                continue
            full_path = str((self._base_dir / path).expanduser().absolute())
            track = MusicTrack(
                id=item["id"],
                path=full_path,
                style=item.get("style", "unknown"),
                mood=item.get("mood", "neutral"),
                energy=float(item.get("energy", 0.5)),
                bpm=item.get("bpm"),
                duration_sec=item.get("duration_sec"),
                metadata={k: v for k, v in item.items()
                          if k not in {"id", "path", "style", "mood", "energy", "bpm", "duration_sec"}},
            )
            tracks.append(track)

        self._tracks = tracks

    def list_tracks(self, style: Optional[str] = None) -> List[MusicTrack]:
        if style is None:
            return list(self._tracks)
        style = style.lower()
        return [t for t in self._tracks if t.style.lower() == style]