# -*- coding: utf-8 -*-
"""
HLP - High Level Policy for Sweep to Shapes

该模块提供用于"Sweep to Shapes"任务的高层策略实现。
"""

from .high_level_policy import HighLevelPolicy
from .config import (
    DEFAULT_CFG,
    validate_cfg,
    load_cfg_from_yaml,
    get_T1_from_cfg,
    load_cfg_and_T1,
    get_table_corners_from_cfg,
    compute_perspective_transforms,
    load_cfg_and_transforms,
)
from .types import (
    ImgBGR, ImgBin01, ImgBin255, Mat33, Mat23,
    Direction, BBox, ChunkExec, HLPResult
)

__all__ = [
    "HighLevelPolicy",
    "DEFAULT_CFG",
    "validate_cfg",
    "load_cfg_from_yaml",
    # 新接口（v1.2）
    "get_table_corners_from_cfg",
    "compute_perspective_transforms",
    "load_cfg_and_transforms",
    # 旧接口（向后兼容）
    "get_T1_from_cfg",
    "load_cfg_and_T1",
    # 类型
    "ImgBGR",
    "ImgBin01",
    "ImgBin255",
    "Mat33",
    "Mat23",
    "Direction",
    "BBox",
    "ChunkExec",
    "HLPResult",
]

__version__ = "1.2.0"
