from .model import MusicTrack, SegmentMusicPlan, MGenOptions
from .interface import (
    MusicLibraryInterface,
    MusicArrangerInterface,
    MusicGeneratorInterface,
)
from .library import JsonMusicLibrary
from .simple_arranger import SimpleRuleArranger

__all__ = [
    "MusicTrack",
    "SegmentMusicPlan",
    "MGenOptions",
    "MusicLibraryInterface",
    "MusicArrangerInterface",
    "MusicGeneratorInterface",
    "JsonMusicLibrary",
    "SimpleRuleArranger",
]


def get_default_mgen(json_path: str = "mgen/tracks.json") -> tuple[JsonMusicLibrary, SimpleRuleArranger]:
    """
    简单工厂：返回 (库, 编排器)
    """
    lib = JsonMusicLibrary(json_path=json_path)
    arranger = SimpleRuleArranger()
    return lib, arranger