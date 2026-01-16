# -*- coding: utf-8 -*-
"""
M2模块：目标对齐优化（T2）+ residual计算

功能：
1. 使用差分进化算法优化相似变换T2，使目标图像与当前积木分布对齐
2. 计算对齐后的目标mask和需要扫除的residual区域

目标函数：J = λ1*C_fill + λ2*C_remove
- C_fill: 目标区域内缺失积木的比例
- C_remove: 当前积木在目标区域外的比例
"""

import numpy as np
import cv2
from scipy.optimize import differential_evolution
from typing import Dict, Any, Tuple
from .types import ImgBin01, Mat23


def m2_normalize_goal_to01(img_bin_table_goal_224: ImgBin01) -> ImgBin01:
    """
    将目标二值图标准化为0/1格式

    Args:
        img_bin_table_goal_224: 输入二值图，可能是0/1或0/255格式

    Returns:
        标准化后的0/1二值图
    """
    g = img_bin_table_goal_224.astype(np.uint8)
    if g.max() > 1:
        # 如果是0/255格式，转换为0/1
        g = (g > 127).astype(np.uint8)
    return g


def m2_params_to_T2(tx: float, ty: float, theta_deg: float, s: float) -> Mat23:
    """
    将参数转换为2x3相似变换矩阵T2

    相似变换参数化：
    - tx, ty: 平移
    - theta_deg: 旋转角度（度）
    - s: 缩放因子

    Args:
        tx: x方向平移
        ty: y方向平移
        theta_deg: 旋转角度（度）
        s: 缩放因子

    Returns:
        2x3仿射变换矩阵
    """
    th = np.deg2rad(theta_deg)
    c, sn = np.cos(th), np.sin(th)

    # 相似变换矩阵：A = s * R(theta)
    A00, A01 = s * c, -s * sn
    A10, A11 = s * sn, s * c

    T2 = np.array([
        [A00, A01, tx],
        [A10, A11, ty]
    ], dtype=np.float64)

    return T2


def m2_warp_goal(goal_01: ImgBin01, T2: Mat23) -> ImgBin01:
    """
    使用T2变换目标图像

    变换规则：先转float32，warpAffine用INTER_LINEAR，再阈值>=0.5得到0/1

    Args:
        goal_01: 0/1格式的目标二值图
        T2: 2x3仿射变换矩阵

    Returns:
        变换后的0/1二值图
    """
    goal_f = goal_01.astype(np.float32)
    warp = cv2.warpAffine(
        goal_f, T2, (224, 224),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return (warp >= 0.5).astype(np.uint8)


def m2_compute_cost(
    B_01: ImgBin01,
    G_01: ImgBin01,
    lambda1: float,
    lambda2: float
) -> Tuple[float, float, float]:
    """
    计算对齐代价

    代价项定义：
    - C_fill = sum(G & (1-B)) / (sum(G) + eps)  目标区域内缺失积木的比例
    - C_remove = sum(B & (1-G)) / (sum(B) + eps)  当前积木在目标区域外的比例
    - J = λ1*C_fill + λ2*C_remove

    Args:
        B_01: 当前积木分布二值图 (0/1)
        G_01: 变换后的目标二值图 (0/1)
        lambda1: C_fill权重
        lambda2: C_remove权重

    Returns:
        tuple: (J, C_fill, C_remove)
    """
    eps = 1e-6
    Gsum = float(G_01.sum())
    Bsum = float(B_01.sum())

    # C_fill: 目标区域内缺失积木的比例
    C_fill = float(np.logical_and(G_01 == 1, B_01 == 0).sum()) / (Gsum + eps)

    # C_remove: 当前积木在目标区域外的比例
    C_remove = float(np.logical_and(B_01 == 1, G_01 == 0).sum()) / (Bsum + eps)

    # 总代价
    J = lambda1 * C_fill + lambda2 * C_remove

    return J, C_fill, C_remove


def m2_optimize_T2_de(
    B_01: ImgBin01,
    goal_01: ImgBin01,
    cfg_m2: Dict[str, Any]
) -> Tuple[Mat23, ImgBin01, float, float, float]:
    """
    使用差分进化算法优化T2变换

    Args:
        B_01: 当前积木分布二值图 (0/1)
        goal_01: 目标二值图 (0/1)
        cfg_m2: M2配置字典

    Returns:
        tuple:
            - T2: 优化后的变换矩阵
            - G_01: 变换后的目标二值图
            - J: 最优代价值
            - C_fill: 填充代价
            - C_remove: 移除代价
    """
    # 获取参数边界
    b = cfg_m2["bounds"]
    bounds = [
        (-b["tx"], b["tx"]),                      # tx
        (-b["ty"], b["ty"]),                      # ty
        (-b["theta_deg"], b["theta_deg"]),        # theta_deg
        (b["scale_low"], b["scale_high"]),        # scale
    ]

    lam1, lam2 = cfg_m2["lambda1"], cfg_m2["lambda2"]

    # 目标函数
    def obj(p):
        tx, ty, theta_deg, s = p
        T2 = m2_params_to_T2(tx, ty, theta_deg, s)
        G = m2_warp_goal(goal_01, T2)
        J, _, _ = m2_compute_cost(B_01, G, lam1, lam2)
        return J

    # 差分进化优化
    res = differential_evolution(
        obj,
        bounds=bounds,
        seed=cfg_m2["seed"],
        maxiter=cfg_m2["maxiter"],
        popsize=cfg_m2["popsize"],
        tol=cfg_m2["tol"],
        workers=cfg_m2["workers"],
        updating=cfg_m2["updating"],
        polish=True,
        disp=False
    )

    # 提取最优参数
    tx, ty, theta_deg, s = res.x
    T2 = m2_params_to_T2(tx, ty, theta_deg, s)
    G = m2_warp_goal(goal_01, T2)
    J, C_fill, C_remove = m2_compute_cost(B_01, G, lam1, lam2)

    return T2, G, float(J), float(C_fill), float(C_remove)


def m2_run(
    B_01: ImgBin01,
    goal_any: ImgBin01,
    cfg_m2: Dict[str, Any]
) -> Tuple[ImgBin01, ImgBin01, Mat23, float, Dict[str, Any]]:
    """
    M2模块主函数：执行目标对齐优化

    Args:
        B_01: 当前积木分布二值图 (0/1)
        goal_any: 目标二值图（可以是0/1或0/255格式）
        cfg_m2: M2配置字典

    Returns:
        tuple:
            - mask_table_goal_01: 对齐后的目标mask (0/1)
            - mask_table_residual_01: 需要扫除的residual区域 (0/1)
            - T2: 优化后的变换矩阵
            - J_best: 最优代价值
            - stats: 统计信息字典
    """
    # 标准化目标图像为0/1
    goal_01 = m2_normalize_goal_to01(goal_any)

    # 执行优化
    T2, G_01, J, C_fill, C_remove = m2_optimize_T2_de(B_01, goal_01, cfg_m2)

    # 计算residual：当前积木在目标区域外的部分（需要扫除）
    residual = np.logical_and(B_01 == 1, G_01 == 0).astype(np.uint8)

    # 统计信息
    stats = dict(
        C_fill=C_fill,
        C_remove=C_remove,
        goal_pixels=int(G_01.sum()),
        cur_pixels=int(B_01.sum())
    )

    return G_01, residual, T2, J, stats
