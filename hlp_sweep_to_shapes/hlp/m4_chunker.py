# -*- coding: utf-8 -*-
"""
M4模块：从Flow Field生成可执行mask列表

功能：
1. 方向离散化：根据梯度方向将residual区域划分为Up/Down/Left/Right四个方向
2. 连通域提取：对每个方向提取连通域
3. 高度约束切分：对过高的区域进行垂直切分
4. 梯度统计和优先级分数计算
5. 落盘保存
"""

import os
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from .types import ImgBin01, ChunkExec, BBox, Direction
from .io_utils import ensure_dir, save_u8_png


# 方向编码：Right=0, Left=1, Down=2, Up=3, invalid=-1
DIR_RIGHT = 0
DIR_LEFT = 1
DIR_DOWN = 2
DIR_UP = 3
DIR_INVALID = -1

# 方向名称映射
DIR_NAMES: Dict[int, Direction] = {
    DIR_RIGHT: "Right",
    DIR_LEFT: "Left",
    DIR_DOWN: "Down",
    DIR_UP: "Up",
}


def m4_direction_discretize(
    map_grad: np.ndarray,
    residual_01: ImgBin01,
    cfg_m4: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    方向离散化：将连续梯度方向离散为Up/Down/Left/Right

    判定规则（带水平偏置hysteresis）：
    - valid = (residual==1) & (mag >= mag_min)
    - is_h = (abs(fx)*lambda_h >= abs(fy))  # 水平偏置
    - 方向编码：
        - Right if is_h & (fx>0)
        - Left if is_h & (fx<=0)
        - Down if ~is_h & (fy>0)
        - Up if ~is_h & (fy<=0)
    - 非valid像素：-1

    Args:
        map_grad: Flow Field float32 (224,224,2)
        residual_01: 需要扫除的区域 (0/1)
        cfg_m4: M4配置字典

    Returns:
        tuple:
            - dir_map: 方向图 int8 (224,224)，值域{-1,0,1,2,3}
            - mag: 梯度模长 float32 (224,224)
    """
    fx = map_grad[..., 0]
    fy = map_grad[..., 1]

    # 计算梯度模长
    mag = np.sqrt(fx ** 2 + fy ** 2).astype(np.float32)

    # 获取参数
    mag_min = cfg_m4["mag_min"]
    lambda_h = cfg_m4["lambda_h"]

    # 有效像素掩码
    valid = (residual_01 == 1) & (mag >= mag_min)

    # 水平偏置判定
    is_h = (np.abs(fx) * lambda_h >= np.abs(fy))

    # 初始化方向图为invalid
    dir_map = np.full((224, 224), DIR_INVALID, dtype=np.int8)

    # 分配方向（向量化操作）
    # Right: is_h & (fx > 0) & valid
    dir_map[valid & is_h & (fx > 0)] = DIR_RIGHT

    # Left: is_h & (fx <= 0) & valid
    dir_map[valid & is_h & (fx <= 0)] = DIR_LEFT

    # Down: ~is_h & (fy > 0) & valid
    dir_map[valid & (~is_h) & (fy > 0)] = DIR_DOWN

    # Up: ~is_h & (fy <= 0) & valid
    dir_map[valid & (~is_h) & (fy <= 0)] = DIR_UP

    return dir_map, mag


def m4_extract_cc_masks(
    dir_map: np.ndarray,
    residual_01: ImgBin01,
    cfg_m4: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    按方向提取连通域

    处理流程：
    1. 对每个方向构造mask
    2. 开运算去噪
    3. 提取连通域
    4. 过滤小面积区域

    Args:
        dir_map: 方向图 int8 (224,224)
        residual_01: 需要扫除的区域 (0/1)
        cfg_m4: M4配置字典

    Returns:
        raw_chunks列表，每个元素为dict:
            {"mask_255": (224,224) uint8, "direction": str, "bbox": (x,y,w,h), "area": int}
    """
    A_min = cfg_m4["A_min"]
    k = cfg_m4["opening_kernel_size"]
    opening_iter = cfg_m4["opening_iter"]

    # 开运算核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    raw_chunks = []

    # 对4个方向分别处理
    for dir_code in [DIR_RIGHT, DIR_LEFT, DIR_DOWN, DIR_UP]:
        direction = DIR_NAMES[dir_code]

        # 构造该方向的mask
        mask_dir = ((dir_map == dir_code) & (residual_01 == 1)).astype(np.uint8) * 255

        # 开运算去噪
        if opening_iter > 0:
            mask_dir = cv2.morphologyEx(
                mask_dir, cv2.MORPH_OPEN, kernel,
                iterations=opening_iter
            )

        # 提取连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_dir, connectivity=8
        )

        # 遍历连通域（跳过背景label=0）
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            # 过滤小面积
            if area < A_min:
                continue

            # 提取该连通域的mask
            cc_mask = (labels == label).astype(np.uint8) * 255

            # 边界框
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            bbox: BBox = (x, y, w, h)

            raw_chunks.append({
                "mask_255": cc_mask,
                "direction": direction,
                "bbox": bbox,
                "area": area,
            })

    return raw_chunks


def m4_height_constrained_split(
    raw_chunks: List[Dict[str, Any]],
    cfg_m4: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    高度约束切分：对过高的区域进行垂直切分

    切分规则：
    - 若bbox高度H_curr <= H_max：保留
    - 若H_curr > H_max：沿y切分为若干条
        1. n = ceil(H_curr / H_max)
        2. S = ceil(H_curr / n)
        3. 若S < H_min：令n = max(1, floor(H_curr / H_min))，再S = ceil(H_curr / n)
        4. 对每条带k：取交集submask = chunk_mask & strip，面积>=A_min则保留

    Args:
        raw_chunks: 原始连通域列表
        cfg_m4: M4配置字典

    Returns:
        切分后的连通域列表
    """
    H_min = cfg_m4["H_min"]
    H_max = cfg_m4["H_max"]
    A_min = cfg_m4["A_min"]

    final_chunks = []

    for chunk in raw_chunks:
        mask_255 = chunk["mask_255"]
        direction = chunk["direction"]
        x, y, w, h = chunk["bbox"]

        # 如果高度在范围内，直接保留
        if h <= H_max:
            final_chunks.append(chunk)
            continue

        # 需要切分
        H_curr = h

        # 计算切分数量
        n = int(np.ceil(H_curr / H_max))
        S = int(np.ceil(H_curr / n))

        # 确保每条高度不小于H_min
        if S < H_min:
            n = max(1, int(np.floor(H_curr / H_min)))
            S = int(np.ceil(H_curr / n))

        # 执行切分
        for k in range(n):
            # 计算条带范围
            strip_y_start = y + k * S
            strip_y_end = min(y + (k + 1) * S, y + H_curr)

            # 创建条带mask
            strip_mask = np.zeros_like(mask_255)
            strip_mask[strip_y_start:strip_y_end, :] = 255

            # 取交集
            submask = cv2.bitwise_and(mask_255, strip_mask)

            # 计算面积
            sub_area = int(np.sum(submask > 0))

            # 过滤小面积
            if sub_area < A_min:
                continue

            # 重新计算边界框
            ys, xs = np.where(submask > 0)
            if len(ys) == 0:
                continue

            new_x = int(xs.min())
            new_y = int(ys.min())
            new_w = int(xs.max() - xs.min() + 1)
            new_h = int(ys.max() - ys.min() + 1)
            new_bbox: BBox = (new_x, new_y, new_w, new_h)

            final_chunks.append({
                "mask_255": submask,
                "direction": direction,
                "bbox": new_bbox,
                "area": sub_area,
            })

    return final_chunks


def m4_attach_grad_stats_and_score(
    chunks: List[Dict[str, Any]],
    mag: np.ndarray
) -> List[ChunkExec]:
    """
    计算梯度统计和优先级分数

    统计项：
    - G_min: mask内最小梯度模长
    - G_max: mask内最大梯度模长
    - priority_score: G_max（越大越优先）

    Args:
        chunks: 连通域列表（dict格式）
        mag: 梯度模长图 float32 (224,224)

    Returns:
        ChunkExec对象列表
    """
    result = []

    for chunk in chunks:
        mask_255 = chunk["mask_255"]
        direction = chunk["direction"]
        bbox = chunk["bbox"]
        area = chunk["area"]

        # 获取mask内的梯度值
        mask_bool = mask_255 > 0
        mag_values = mag[mask_bool]

        if len(mag_values) == 0:
            G_min = 0.0
            G_max = 0.0
        else:
            G_min = float(mag_values.min())
            G_max = float(mag_values.max())

        # 优先级分数
        priority_score = G_max

        chunk_exec = ChunkExec(
            mask_table_exec_255=mask_255,
            direction=direction,
            G_min=G_min,
            G_max=G_max,
            area=area,
            bbox=bbox,
            priority_score=priority_score,
            file_path=None
        )

        result.append(chunk_exec)

    return result


def m4_save_masks(
    chunks: List[ChunkExec],
    save_dir: str,
    mkdir_exist_ok: bool = True
) -> List[ChunkExec]:
    """
    保存mask到磁盘

    文件名规范：mask_{direction}_min{G_min:.3f}_max{G_max:.3f}_area{area}.png

    Args:
        chunks: ChunkExec对象列表
        save_dir: 保存目录
        mkdir_exist_ok: 创建目录时是否允许已存在

    Returns:
        更新了file_path的ChunkExec对象列表
    """
    masks_dir = os.path.join(save_dir, "m4_masks")
    ensure_dir(masks_dir, exist_ok=mkdir_exist_ok)

    result = []

    for i, chunk in enumerate(chunks):
        # 生成文件名
        filename = f"mask_{chunk.direction}_min{chunk.G_min:.3f}_max{chunk.G_max:.3f}_area{chunk.area}.png"
        filepath = os.path.join(masks_dir, filename)

        # 保存
        save_u8_png(filepath, chunk.mask_table_exec_255)

        # 创建新的ChunkExec对象（因为frozen=True）
        new_chunk = ChunkExec(
            mask_table_exec_255=chunk.mask_table_exec_255,
            direction=chunk.direction,
            G_min=chunk.G_min,
            G_max=chunk.G_max,
            area=chunk.area,
            bbox=chunk.bbox,
            priority_score=chunk.priority_score,
            file_path=filepath
        )

        result.append(new_chunk)

    return result


def m4_run(
    map_grad: np.ndarray,
    residual_01: ImgBin01,
    cfg_m4: Dict[str, Any],
    save_dir: Optional[str],
    io_cfg: Dict[str, Any]
) -> Tuple[List[ChunkExec], Dict[str, Any]]:
    """
    M4模块主函数：从Flow Field生成可执行mask列表

    Args:
        map_grad: Flow Field float32 (224,224,2)
        residual_01: 需要扫除的区域 (0/1)
        cfg_m4: M4配置字典
        save_dir: 保存目录（可选）
        io_cfg: I/O配置字典

    Returns:
        tuple:
            - chunks: ChunkExec对象列表
            - stats: 统计信息字典
    """
    # 步骤1: 方向离散化
    dir_map, mag = m4_direction_discretize(map_grad, residual_01, cfg_m4)

    # 步骤2: 提取连通域
    raw_chunks = m4_extract_cc_masks(dir_map, residual_01, cfg_m4)

    # 步骤3: 高度约束切分
    split_chunks = m4_height_constrained_split(raw_chunks, cfg_m4)

    # 步骤4: 计算梯度统计和优先级分数
    chunks = m4_attach_grad_stats_and_score(split_chunks, mag)

    # 步骤5: 保存mask（如果启用）
    if save_dir and io_cfg.get("save_m4_masks", True):
        chunks = m4_save_masks(chunks, save_dir, io_cfg.get("mkdir_exist_ok", True))

    # 统计信息
    stats = {
        "chunks": len(chunks),
        "raw_chunks": len(raw_chunks),
        "split_chunks": len(split_chunks),
    }

    return chunks, stats
