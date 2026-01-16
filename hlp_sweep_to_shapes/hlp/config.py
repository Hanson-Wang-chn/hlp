# -*- coding: utf-8 -*-
"""
配置模块
包含默认配置和配置校验函数
"""

import copy
import numpy as np
from typing import Dict, Any, Optional, Tuple

# ========== 默认配置 ==========
DEFAULT_CFG: Dict[str, Any] = dict(
    # ---------- table_corners: 4个角点坐标 ----------
    # 表示224x224的img_rgb_main_cur上的4个点，按左上、右上、右下、左下顺序
    # 这4个点围成的四边形将被仿射变换到224x224的img_rgb_table_cur上
    # 默认为整个图像的4个角点（无变换）
    table_corners=[
        [0.0, 0.0],       # 左上
        [223.0, 0.0],     # 右上
        [223.0, 223.0],   # 右下
        [0.0, 223.0],     # 左下
    ],

    # ---------- 通用调试配置 ----------
    debug=dict(
        save_intermediate=True,     # 是否保存中间产物到磁盘
        return_intermediate=False,  # 是否把中间数组也塞进 result["debug"]（注意内存）
    ),

    # ---------- M1: 主相机矫正 + 红色积木分割 ----------
    m1=dict(
        hsv_red_low1=(0, 120, 60),       # 红色HSV下界1
        hsv_red_high1=(10, 255, 255),    # 红色HSV上界1
        hsv_red_low2=(170, 120, 60),     # 红色HSV下界2
        hsv_red_high2=(180, 255, 255),   # 红色HSV上界2
        morph_kernel_shape="ellipse",    # 形态学核形状（固定为ellipse）
        morph_kernel_size=3,             # 形态学核大小 (3,3)
        morph_open_iter=1,               # 开运算迭代次数
        morph_close_iter=1,              # 闭运算迭代次数
    ),

    # ---------- M2: 目标对齐优化 ----------
    m2=dict(
        seed=0,                          # 随机种子
        maxiter=30,                      # 最大迭代次数
        popsize=10,                      # 种群大小
        tol=1e-3,                        # 收敛容差
        lambda1=0.6,                     # C_fill 权重
        lambda2=0.4,                     # C_remove 权重
        bounds=dict(
            tx=18.0,                     # 平移x范围 [-18, 18]
            ty=18.0,                     # 平移y范围 [-18, 18]
            theta_deg=8.0,               # 旋转角度范围 [-8, 8] 度
            scale_low=0.92,              # 缩放下界
            scale_high=1.08,             # 缩放上界
        ),
        force_reopt_each_step=False,     # 是否每步强制重新优化
        workers=1,                       # 并行工作数（1为确定性）
        updating="deferred",             # 更新策略
    ),

    # ---------- M3: 障碍感知测地线距离场 + Flow Field ----------
    m3=dict(
        bfs_connectivity=4,              # BFS连通性（固定为4邻域）
        obstacle_fill_dist_margin=5.0,   # 障碍填充距离边缘值
        sobel_ksize=3,                   # Sobel算子核大小
    ),

    # ---------- M4: 方向离散化 + 连通域切分 ----------
    m4=dict(
        lambda_h=1.2,                    # 水平偏置系数（必须>1）
        mag_min=1e-3,                    # 最小梯度模长阈值
        A_min=60,                        # 最小面积阈值
        H_min=18,                        # 最小高度
        H_max=42,                        # 最大高度
        opening_kernel_size=3,           # 开运算核大小
        opening_iter=1,                  # 开运算迭代次数
    ),

    # ---------- M5: 选择 + 反变换 + 叠加可视化 ----------
    m5=dict(
        overlay_alpha=0.35,              # 叠加透明度
        color_up=(255, 0, 0),            # Up方向颜色 (BGR)
        color_down=(0, 0, 255),          # Down方向颜色 (BGR)
        color_left=(0, 255, 255),        # Left方向颜色 (BGR)
        color_right=(0, 255, 0),         # Right方向颜色 (BGR)
    ),

    # ---------- I/O 配置 ----------
    io=dict(
        save_m4_masks=True,              # 是否保存M4的masks
        save_m5_overlay=True,            # 是否保存M5的叠加图
        save_m1=True,                    # 是否保存M1的输出
        save_m2=True,                    # 是否保存M2的输出
        save_m3=True,                    # 是否保存M3的输出
        mkdir_exist_ok=True,             # 创建目录时是否允许已存在
    ),

    # ---------- 日志配置 ----------
    log=dict(
        level="INFO",                    # 日志级别
        structured=True,                 # 是否输出结构化日志
        print_to_console=True,           # 是否打印到控制台
    ),
)


def validate_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    校验并补齐配置字段

    Args:
        cfg: 用户提供的配置字典

    Returns:
        补齐后的配置字典

    Raises:
        ValueError: 配置参数不合法时抛出
    """
    # 深拷贝默认配置作为基础
    result = copy.deepcopy(DEFAULT_CFG)

    # 递归合并用户配置
    def merge_dict(base: dict, override: dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value

    if cfg:
        merge_dict(result, cfg)

    # ========== 参数范围校验 ==========

    # M2: lambda1 + lambda2 应该为正数
    if result["m2"]["lambda1"] < 0:
        raise ValueError("m2.lambda1 必须 >= 0")
    if result["m2"]["lambda2"] < 0:
        raise ValueError("m2.lambda2 必须 >= 0")
    if result["m2"]["lambda1"] + result["m2"]["lambda2"] <= 0:
        raise ValueError("m2.lambda1 + m2.lambda2 必须 > 0")

    # M2: bounds 校验
    if result["m2"]["bounds"]["scale_low"] >= result["m2"]["bounds"]["scale_high"]:
        raise ValueError("m2.bounds.scale_low 必须 < m2.bounds.scale_high")
    if result["m2"]["bounds"]["scale_low"] <= 0:
        raise ValueError("m2.bounds.scale_low 必须 > 0")

    # M4: lambda_h 必须 > 1
    if result["m4"]["lambda_h"] <= 1.0:
        raise ValueError("m4.lambda_h 必须 > 1.0")

    # M4: H_min <= H_max
    if result["m4"]["H_min"] > result["m4"]["H_max"]:
        raise ValueError("m4.H_min 必须 <= m4.H_max")

    # M4: A_min 必须为正
    if result["m4"]["A_min"] <= 0:
        raise ValueError("m4.A_min 必须 > 0")

    # M5: overlay_alpha 必须在 (0, 1) 之间
    if not (0 < result["m5"]["overlay_alpha"] < 1):
        raise ValueError("m5.overlay_alpha 必须在 (0, 1) 之间")

    # M3: obstacle_fill_dist_margin 必须为正
    if result["m3"]["obstacle_fill_dist_margin"] <= 0:
        raise ValueError("m3.obstacle_fill_dist_margin 必须 > 0")

    return result


def load_cfg_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置

    Args:
        yaml_path: YAML配置文件路径

    Returns:
        配置字典
    """
    import yaml

    with open(yaml_path, 'r', encoding='utf-8') as f:
        user_cfg = yaml.safe_load(f) or {}

    return validate_cfg(user_cfg)


def get_table_corners_from_cfg(cfg: Dict[str, Any]) -> np.ndarray:
    """
    从配置字典中提取table_corners（4个角点坐标）

    Args:
        cfg: 配置字典（已经过validate_cfg校验）

    Returns:
        (4, 2) 角点坐标数组 (float32)，顺序为：左上、右上、右下、左下

    Raises:
        ValueError: table_corners配置格式不正确时抛出
    """
    corners_list = cfg.get("table_corners", DEFAULT_CFG["table_corners"])

    # 转换为numpy数组
    corners = np.array(corners_list, dtype=np.float32)

    # 校验形状
    if corners.shape != (4, 2):
        raise ValueError(
            f"table_corners形状必须为(4,2)，当前为{corners.shape}。"
            "请确保table_corners配置为4个[x,y]坐标的列表。"
        )

    # 校验坐标范围（应在0-223之间）
    if np.any(corners < 0) or np.any(corners > 223):
        raise ValueError(
            "table_corners中的坐标应在[0, 223]范围内。"
        )

    return corners


def compute_perspective_transforms(table_corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据4个角点计算透视变换矩阵及其逆矩阵

    将table_corners定义的四边形变换到224x224的目标矩形

    Args:
        table_corners: (4, 2) 角点坐标，顺序为：左上、右上、右下、左下

    Returns:
        (T1, T1_inv) 元组:
            - T1: main->table 透视变换矩阵 (3x3)
            - T1_inv: table->main 逆透视变换矩阵 (3x3)
    """
    import cv2

    # 源点（主相机图像上的4个角点）
    src_pts = table_corners.astype(np.float32)

    # 目标点（224x224填满的矩形）
    dst_pts = np.array([
        [0.0, 0.0],       # 左上
        [223.0, 0.0],     # 右上
        [223.0, 223.0],   # 右下
        [0.0, 223.0],     # 左下
    ], dtype=np.float32)

    # 计算透视变换矩阵（main -> table）
    T1 = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 计算逆变换矩阵（table -> main）
    T1_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    return T1.astype(np.float64), T1_inv.astype(np.float64)


def load_cfg_and_transforms(yaml_path: Optional[str] = None) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    同时加载配置和透视变换矩阵

    便捷函数，一次性返回配置、T1矩阵和T1逆矩阵。

    Args:
        yaml_path: YAML配置文件路径（可选，不提供则使用默认配置）

    Returns:
        (cfg, T1, T1_inv) 元组
    """
    if yaml_path:
        cfg = load_cfg_from_yaml(yaml_path)
    else:
        cfg = validate_cfg({})

    table_corners = get_table_corners_from_cfg(cfg)
    T1, T1_inv = compute_perspective_transforms(table_corners)
    return cfg, T1, T1_inv


# 保留旧接口以保持向后兼容
def get_T1_from_cfg(cfg: Dict[str, Any]) -> np.ndarray:
    """
    [已弃用] 从配置字典中提取T1矩阵

    请使用 get_table_corners_from_cfg() 和 compute_perspective_transforms() 代替。

    Args:
        cfg: 配置字典

    Returns:
        3x3 透视变换矩阵 (float64)
    """
    table_corners = get_table_corners_from_cfg(cfg)
    T1, _ = compute_perspective_transforms(table_corners)
    return T1


def load_cfg_and_T1(yaml_path: Optional[str] = None) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    [已弃用] 同时加载配置和T1矩阵

    请使用 load_cfg_and_transforms() 代替。

    Args:
        yaml_path: YAML配置文件路径（可选，不提供则使用默认配置）

    Returns:
        (cfg, T1) 元组
    """
    cfg, T1, _ = load_cfg_and_transforms(yaml_path)
    return cfg, T1
