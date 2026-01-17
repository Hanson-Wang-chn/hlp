# -*- coding: utf-8 -*-
"""
M1模块：主相机图像矫正 + 红色积木分割

功能：
1. 通过单应变换T1将主相机图像矫正到table平面视角
2. 使用HSV双区间阈值分割红色积木
3. 形态学去噪得到最终的二值分割结果
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple
from .types import ImgBGR, ImgBin01, Mat33


def m1_rectify_to_table(img_rgb_main_cur_224: ImgBGR, T1: Mat33) -> ImgBGR:
    """
    使用单应变换T1将主相机图像矫正到table平面视角

    Args:
        img_rgb_main_cur_224: 主相机图像 (224,224,3) BGR格式
        T1: 3x3单应变换矩阵

    Returns:
        矫正后的table视角图像 (224,224,3) BGR格式
    """
    return cv2.warpPerspective(
        img_rgb_main_cur_224, T1, (224, 224),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


def m1_segment_red_blocks(img_rgb_table_cur_224: ImgBGR, cfg_m1: Dict[str, Any]) -> ImgBin01:
    """
    使用HSV双区间阈值分割红色积木

    红色在HSV空间中跨越0度附近，需要用两个区间：
    - 区间1: H ∈ [0,10]
    - 区间2: H ∈ [170,180]

    Args:
        img_rgb_table_cur_224: table视角BGR图像 (224,224,3)
        cfg_m1: M1配置字典

    Returns:
        二值分割结果 (224,224), 值域{0,1}
    """
    # BGR转HSV
    hsv = cv2.cvtColor(img_rgb_table_cur_224, cv2.COLOR_BGR2HSV)

    # 获取HSV阈值参数
    low1 = np.array(cfg_m1["hsv_red_low1"], dtype=np.uint8)
    high1 = np.array(cfg_m1["hsv_red_high1"], dtype=np.uint8)
    low2 = np.array(cfg_m1["hsv_red_low2"], dtype=np.uint8)
    high2 = np.array(cfg_m1["hsv_red_high2"], dtype=np.uint8)

    # 双区间阈值分割
    m1 = cv2.inRange(hsv, low1, high1)   # 区间1: H ∈ [0,10]
    m2 = cv2.inRange(hsv, low2, high2)   # 区间2: H ∈ [170,180]
    mask = cv2.bitwise_or(m1, m2)        # 合并两个区间，结果为 0/255

    # 形态学去噪
    k = cfg_m1["morph_kernel_size"]
    shape_str = str(cfg_m1.get("morph_kernel_shape", "ellipse")).lower()
    shape_map = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
    }
    kernel_shape = shape_map.get(shape_str, cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(kernel_shape, (k, k))

    # 开运算去除小噪点
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel,
        iterations=cfg_m1["morph_open_iter"]
    )

    # 闭运算填充小孔洞
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel,
        iterations=cfg_m1["morph_close_iter"]
    )

    # 转换为0/1二值图
    return (mask > 0).astype(np.uint8)


def m1_run(
    img_rgb_main_cur_224: ImgBGR,
    T1: Mat33,
    cfg_m1: Dict[str, Any]
) -> Tuple[ImgBGR, ImgBin01]:
    """
    M1模块主函数：执行图像矫正和红色积木分割

    Args:
        img_rgb_main_cur_224: 主相机图像 (224,224,3) BGR格式
        T1: 3x3单应变换矩阵
        cfg_m1: M1配置字典

    Returns:
        tuple:
            - img_rgb_table_cur_224: 矫正后的table视角图像
            - img_bin_table_cur_224: 红色积木二值分割结果 (0/1)
    """
    # 步骤1: 透视矫正
    img_rgb_table = m1_rectify_to_table(img_rgb_main_cur_224, T1)

    # 步骤2: 红色积木分割
    bin_cur = m1_segment_red_blocks(img_rgb_table, cfg_m1)

    return img_rgb_table, bin_cur
