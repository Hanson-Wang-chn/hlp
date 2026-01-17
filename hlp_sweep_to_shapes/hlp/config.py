# -*- coding: utf-8 -*-
"""
配置模块
包含默认配置和配置校验函数
"""

import copy
import numpy as np
from pathlib import Path
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

    # ---------- 目标处理配置 ----------
    goal=dict(
        use_T1_for_goal=False,      # 是否在进入M2前先对目标mask应用T1透视（默认关闭，保持旧逻辑）
    ),

    # ---------- M1: 主相机矫正 + 红色积木分割 ----------
    m1=dict(
        hsv_red_low1=(0, 120, 60),       # 红色HSV下界1
        hsv_red_high1=(10, 255, 255),    # 红色HSV上界1
        hsv_red_low2=(170, 120, 60),     # 红色HSV下界2
        hsv_red_high2=(180, 255, 255),   # 红色HSV上界2
        morph_kernel_shape="ellipse",    # 形态学核形状：ellipse/rect/cross
        morph_kernel_size=3,             # 形态学核大小 (3,3)
        morph_open_iter=1,               # 开运算迭代次数
        morph_close_iter=1,              # 闭运算迭代次数
    ),

    # ---------- M2: 目标对齐优化 ----------
    # 完全同步自 Residual-Perception-Preprocessor/hlp_preprocessor.py
    m2=dict(
        # 优化器选择
        optimizer="differential_evolution",  # "hybrid", "grid", "scipy", "differential_evolution"

        # 代价函数权重
        # J(T) = λ1·C_fill + λ2·C_remove + λ3·C_edge + λ4·C_sweep
        # 默认值与 Residual-Perception-Preprocessor/config/config.yaml 保持一致
        lambda_fill=2.0,                 # λ1: C_fill权重
        lambda_remove=1.0,               # λ2: C_remove权重
        lambda_edge=0.0,                 # λ3: C_edge权重
        lambda_sweep=0.0,                # λ4: C_sweep权重

        # 变换参数正则化权重（Reg(T) = Σ ρ · Δ²）
        reg_tx=0.0,                      # ρ_tx: 平移tx惩罚系数
        reg_ty=0.0,                      # ρ_ty: 平移ty惩罚系数
        reg_theta=0.0,                   # ρ_theta: 旋转角度惩罚系数
        reg_scale=0.0,                   # ρ_scale: 缩放偏离参考值的惩罚系数
        reg_scale_ref=1.0,               # 缩放正则参考值（通常为1.0）

        # 扩展代价参数
        sigma_edge=10.0,                 # 边缘代价高斯核σ
        alpha_sweep=2.0,                 # 扫除代价距离场指数

        # 积木堆mask平滑（用于M2优化与residual计算；同步RPP的MaskSmoother）
        smooth_kernel=5,                 # 形态学核大小（奇数）
        smooth_sigma=2.0,                # Gaussian blur sigma
        smooth_morph_iterations=2,       # closing/opening迭代次数
        smooth_use_closing=True,
        smooth_use_opening=True,

        # 混合优化参数
        grid_resolution=20000,           # 网格搜索分辨率（与 RPP config 默认保持一致）
        local_maxiter=2000,              # L-BFGS-B最大迭代次数

        # 差分进化参数
        maxiter=1000,                    # 差分进化最大迭代次数
        seed=42,                         # 随机种子

        # 变换参数约束
        theta_range=[-np.pi/4, np.pi/4], # 旋转角度范围（弧度）
        # 与RPP配置对齐的缩放范围字段（保留bounds以向后兼容）
        scale_min=0.3,
        scale_max=1.5,
        tx_abs_max=None,                # 平移范围（像素），None时使用默认范围
        ty_abs_max=None,
        bounds=dict(
            scale_low=0.3,               # [兼容] 缩放下界
            scale_high=1.5,              # [兼容] 缩放上界
        ),

        # 缓存控制
        # RPP每次process都会重新优化；HLP侧默认也改为每步重算以保证一致性
        force_reopt_each_step=True,      # 是否每步强制重新优化
    ),

    # ---------- M3: 障碍感知测地线距离场 + Flow Field ----------
    m3=dict(
        bfs_connectivity=4,              # BFS连通性（4或8）
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

    # M2: 代价函数权重校验
    if result["m2"]["lambda_fill"] < 0:
        raise ValueError("m2.lambda_fill 必须 >= 0")
    if result["m2"]["lambda_remove"] < 0:
        raise ValueError("m2.lambda_remove 必须 >= 0")
    if result["m2"]["lambda_fill"] + result["m2"]["lambda_remove"] <= 0:
        raise ValueError("m2.lambda_fill + m2.lambda_remove 必须 > 0")
    for key in ("reg_tx", "reg_ty", "reg_theta", "reg_scale"):
        if result["m2"][key] < 0:
            raise ValueError(f"m2.{key} 必须 >= 0")

    # M2: 优化器类型校验
    valid_optimizers = ["hybrid", "grid", "scipy", "differential_evolution"]
    if result["m2"]["optimizer"] not in valid_optimizers:
        raise ValueError(f"m2.optimizer 必须是 {valid_optimizers} 之一")

    # M2: bounds 校验
    # 兼容字段映射：bounds.scale_low/high -> scale_min/max
    if "scale_min" not in result["m2"] or result["m2"]["scale_min"] is None:
        result["m2"]["scale_min"] = result["m2"]["bounds"].get("scale_low", 0.3)
    if "scale_max" not in result["m2"] or result["m2"]["scale_max"] is None:
        result["m2"]["scale_max"] = result["m2"]["bounds"].get("scale_high", 1.5)

    # 反向回填，保证外部仍可读取bounds字段
    result["m2"]["bounds"]["scale_low"] = float(result["m2"]["scale_min"])
    result["m2"]["bounds"]["scale_high"] = float(result["m2"]["scale_max"])

    if result["m2"]["scale_min"] >= result["m2"]["scale_max"]:
        raise ValueError("m2.scale_min 必须 < m2.scale_max")
    if result["m2"]["scale_min"] <= 0:
        raise ValueError("m2.scale_min 必须 > 0")

    # M2: 其他参数校验
    if result["m2"]["sigma_edge"] <= 0:
        raise ValueError("m2.sigma_edge 必须 > 0")
    if result["m2"]["grid_resolution"] < 1:
        raise ValueError("m2.grid_resolution 必须 >= 1")
    if result["m2"]["smooth_kernel"] is not None and result["m2"]["smooth_kernel"] > 0:
        if result["m2"]["smooth_kernel"] % 2 == 0:
            raise ValueError("m2.smooth_kernel 必须为奇数")
        if result["m2"]["smooth_kernel"] < 3:
            raise ValueError("m2.smooth_kernel 必须 >= 3")
    if result["m2"]["smooth_sigma"] < 0:
        raise ValueError("m2.smooth_sigma 必须 >= 0")
    if result["m2"]["smooth_morph_iterations"] < 0:
        raise ValueError("m2.smooth_morph_iterations 必须 >= 0")
    if result["m2"]["tx_abs_max"] is not None and result["m2"]["tx_abs_max"] < 0:
        raise ValueError("m2.tx_abs_max 必须 >= 0 或为 null")
    if result["m2"]["ty_abs_max"] is not None and result["m2"]["ty_abs_max"] < 0:
        raise ValueError("m2.ty_abs_max 必须 >= 0 或为 null")

    # M1: morph_kernel_shape 合法性校验
    valid_kernel_shapes = ["ellipse", "rect", "cross"]
    if result["m1"]["morph_kernel_shape"] not in valid_kernel_shapes:
        raise ValueError(f"m1.morph_kernel_shape 必须是 {valid_kernel_shapes} 之一")

    # M3: bfs_connectivity 合法性校验
    if result["m3"]["bfs_connectivity"] not in (4, 8):
        raise ValueError("m3.bfs_connectivity 只能是 4 或 8")

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

    # goal: use_T1_for_goal 必须为bool
    if not isinstance(result["goal"]["use_T1_for_goal"], bool):
        raise ValueError("goal.use_T1_for_goal 必须是布尔值")

    return result


def _to_nested_m2(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    将仅包含 M2 字段的扁平配置自动包装为嵌套结构
    （便于直接加载仅包含 M2 键的配置文件）
    """
    if not cfg:
        return {}

    if "m2" in cfg:
        return cfg

    m2_keys = set(DEFAULT_CFG["m2"].keys())
    if set(cfg.keys()).issubset(m2_keys):
        return {"m2": cfg}

    return cfg


def load_cfg_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置

    Args:
        yaml_path: YAML配置文件路径

    Returns:
        配置字典
    """
    import yaml

    yaml_path = Path(yaml_path)

    def merge_dict(base: dict, override: dict) -> None:
        for key, value in (override or {}).items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value

    # 加载配置
    with open(yaml_path, 'r', encoding='utf-8') as f:
        user_cfg = _to_nested_m2(yaml.safe_load(f) or {})

    merged_cfg: Dict[str, Any] = {}
    merge_dict(merged_cfg, user_cfg)

    return validate_cfg(merged_cfg)


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
