# -*- coding: utf-8 -*-
"""
M5模块：选择执行mask + 反变换回主相机 + 方向上色叠加输出

功能：
1. 按优先级分数排序并选择要执行的mask
2. 将table视角的mask反变换回主相机视角
3. 在主相机图像上叠加方向颜色可视化
"""

import os
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from .types import ImgBGR, ImgBin255, Mat33, ChunkExec, Direction
from .io_utils import ensure_dir, save_u8_png


def m5_sort_chunks(chunks: List[ChunkExec]) -> List[ChunkExec]:
    """
    按优先级排序chunk列表

    排序规则：
    1. 按priority_score降序
    2. 同分按area降序（稳定排序）

    Args:
        chunks: ChunkExec对象列表

    Returns:
        排序后的列表
    """
    # 稳定排序：先按area排序，再按priority_score排序
    sorted_chunks = sorted(chunks, key=lambda c: c.area, reverse=True)
    sorted_chunks = sorted(sorted_chunks, key=lambda c: c.priority_score, reverse=True)
    return sorted_chunks


def m5_select_chunk(
    chunks_sorted: List[ChunkExec],
    index_mask_exec: int
) -> Tuple[Optional[ChunkExec], Dict[str, Any]]:
    """
    选择要执行的chunk

    选择规则：
    - 若列表为空：返回None
    - 若index_mask_exec >= len(list)：取最后一个
    - 否则取指定索引

    Args:
        chunks_sorted: 排序后的ChunkExec列表
        index_mask_exec: 要选择的索引

    Returns:
        tuple:
            - chunk: 选中的ChunkExec或None
            - meta_partial: 部分meta信息
    """
    if not chunks_sorted:
        return None, {"status": "empty", "reason": "no chunks available"}

    # 处理索引越界
    actual_idx = min(index_mask_exec, len(chunks_sorted) - 1)
    chunk = chunks_sorted[actual_idx]

    meta_partial = {
        "status": "ok",
        "chosen_idx_sorted": actual_idx,
        "total_chunks": len(chunks_sorted),
    }

    return chunk, meta_partial


def m5_warp_table_mask_to_main(
    mask_table_exec_255: ImgBin255,
    T1_inv: Mat33
) -> ImgBin255:
    """
    将table视角的mask反变换回主相机视角

    使用INTER_NEAREST插值以保持二值mask的锐利边缘

    Args:
        mask_table_exec_255: table视角的mask (0/255)
        T1_inv: T1的逆矩阵（table→main）

    Returns:
        主相机视角的mask (0/255)
    """
    mask_main = cv2.warpPerspective(
        mask_table_exec_255, T1_inv, (224, 224),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return mask_main.astype(np.uint8)


def m5_overlay_direction(
    img_rgb_main_cur_224: ImgBGR,
    mask_main_exec_255: ImgBin255,
    direction: Direction,
    cfg_m5: Dict[str, Any]
) -> ImgBGR:
    """
    在主相机图像上叠加方向颜色

    Args:
        img_rgb_main_cur_224: 主相机BGR图像 (224,224,3)
        mask_main_exec_255: 执行区域mask (0/255)
        direction: 扫除方向
        cfg_m5: M5配置字典

    Returns:
        叠加后的BGR图像 (224,224,3)
    """
    # 获取方向颜色（BGR）
    color_map = {
        "Up": cfg_m5["color_up"],
        "Down": cfg_m5["color_down"],
        "Left": cfg_m5["color_left"],
        "Right": cfg_m5["color_right"],
    }
    color = color_map.get(direction, (255, 255, 255))

    # 透明度
    alpha = cfg_m5["overlay_alpha"]

    # 复制原图
    result = img_rgb_main_cur_224.copy()

    # 创建颜色叠加层
    overlay = np.zeros_like(result)
    overlay[:] = color

    # 在mask区域进行alpha混合
    mask_bool = mask_main_exec_255 > 0
    result[mask_bool] = cv2.addWeighted(
        result[mask_bool].reshape(-1, 3),
        1 - alpha,
        overlay[mask_bool].reshape(-1, 3),
        alpha,
        0
    ).reshape(-1, 3)

    return result


def m5_run(
    chunks: List[ChunkExec],
    index_mask_exec: int,
    img_rgb_main_cur_224: ImgBGR,
    T1_inv: Mat33,
    cfg_m5: Dict[str, Any],
    save_dir: Optional[str],
    io_cfg: Dict[str, Any],
    step_id: Optional[int]
) -> Tuple[ImgBin255, ImgBGR, Dict[str, Any]]:
    """
    M5模块主函数：选择、变换、叠加

    Args:
        chunks: ChunkExec对象列表
        index_mask_exec: 要选择的chunk索引
        img_rgb_main_cur_224: 主相机BGR图像 (224,224,3)
        T1_inv: T1的逆矩阵（table→main）
        cfg_m5: M5配置字典
        save_dir: 保存目录（可选）
        io_cfg: I/O配置字典
        step_id: 步骤ID（用于文件名）

    Returns:
        tuple:
            - mask_main_exec_255: 主相机视角的执行mask (0/255)
            - img_rgb_main_exec_224: 叠加后的可视化图像
            - meta: 完整的meta信息
    """
    # 步骤1: 排序
    chunks_sorted = m5_sort_chunks(chunks)

    # 步骤2: 选择
    chunk, meta = m5_select_chunk(chunks_sorted, index_mask_exec)

    # 如果没有chunk，返回原图和空mask
    if chunk is None:
        empty_mask = np.zeros((224, 224), dtype=np.uint8)
        return empty_mask, img_rgb_main_cur_224.copy(), meta

    # 步骤3: 反变换
    mask_main_exec_255 = m5_warp_table_mask_to_main(
        chunk.mask_table_exec_255, T1_inv
    )

    # 步骤4: 叠加可视化
    img_rgb_main_exec = m5_overlay_direction(
        img_rgb_main_cur_224, mask_main_exec_255,
        chunk.direction, cfg_m5
    )

    # 更新meta
    meta.update({
        "direction": chunk.direction,
        "G_min": chunk.G_min,
        "G_max": chunk.G_max,
        "area": chunk.area,
        "bbox": chunk.bbox,
        "priority_score": chunk.priority_score,
    })

    # 步骤5: 保存（如果启用）
    if save_dir and io_cfg.get("save_m5_overlay", True):
        overlay_dir = os.path.join(save_dir, "m5_overlay")
        ensure_dir(overlay_dir, exist_ok=io_cfg.get("mkdir_exist_ok", True))

        step_str = f"step{step_id}" if step_id is not None else "step"
        filename = f"overlay_{step_str}_idx{meta['chosen_idx_sorted']}_dir{chunk.direction}.png"
        filepath = os.path.join(overlay_dir, filename)

        save_u8_png(filepath, img_rgb_main_exec)
        meta["overlay_path"] = filepath

    return mask_main_exec_255, img_rgb_main_exec, meta
