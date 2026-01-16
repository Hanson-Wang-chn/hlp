# -*- coding: utf-8 -*-
"""
类型定义模块
包含项目中使用的所有类型别名、数据类和TypedDict定义
"""

from dataclasses import dataclass
from typing import Literal, TypedDict, Optional, Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

# ========== numpy 类型别名 ==========
U8 = np.uint8
F32 = np.float32
F64 = np.float64

# 图像类型
ImgBGR = NDArray[U8]         # shape=(224,224,3), BGR格式
ImgBin01 = NDArray[U8]       # shape=(224,224), 值域 {0,1}
ImgBin255 = NDArray[U8]      # shape=(224,224), 值域 {0,255}

# 矩阵类型
Mat33 = NDArray[F64]         # shape=(3,3), 单应矩阵
Mat23 = NDArray[F64]         # shape=(2,3), 仿射变换矩阵

# 方向类型
Direction = Literal["Up", "Down", "Left", "Right"]

# 边界框类型 (x, y, w, h)
BBox = Tuple[int, int, int, int]


# ========== 结构化输出数据结构 ==========
@dataclass(frozen=True)
class ChunkExec:
    """
    可执行的sweep区域块
    包含mask、方向、梯度统计、优先级分数等信息
    """
    mask_table_exec_255: ImgBin255      # 0/255 二值mask
    direction: Direction                 # 扫除方向
    G_min: float                         # 区域内最小梯度模长
    G_max: float                         # 区域内最大梯度模长
    area: int                            # 区域面积（像素数）
    bbox: BBox                           # 边界框 (x, y, w, h)
    priority_score: float                # 优先级分数
    file_path: Optional[str] = None      # 落盘路径（若已保存）


class HLPResult(TypedDict):
    """
    HLP推理结果字典
    """
    img_rgb_main_exec_224: ImgBGR        # 叠加可视化后的主相机图像
    mask_main_exec_224: ImgBin255        # 选中执行区域的mask（主相机视角）
    meta: Dict[str, Any]                 # 元信息（方向、梯度统计、bbox、排序分数等）
    debug: Dict[str, Any]                # 调试信息（中间产物路径或数组）
