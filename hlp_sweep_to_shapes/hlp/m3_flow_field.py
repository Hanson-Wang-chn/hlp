# -*- coding: utf-8 -*-
"""
M3模块：Obstacle-Aware Geodesic Distance → Flow Field

功能：
1. 使用BFS计算障碍感知的测地线距离场（从边界向内传播，目标区域为障碍）
2. 通过Sobel算子计算距离场梯度，生成Flow Field
"""

import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Tuple
from .types import ImgBin01


def m3_bfs_distance_obstacle_aware(goal_01: ImgBin01, cfg_m3: Dict[str, Any]) -> np.ndarray:
    """
    使用BFS计算障碍感知的测地线距离场

    距离场定义：
    - 源点集合：所有边界像素（x==0 or x==W-1 or y==0 or y==H-1）且goal==0
    - 传播规则：4邻域BFS，每步代价1，禁止进入goal==1像素
    - 输出：可达像素为非负整数距离，goal==1及不可达像素为-1

    Args:
        goal_01: 目标区域二值图 (0/1)，1表示障碍
        cfg_m3: M3配置字典

    Returns:
        距离场 int32 shape=(224,224)，-1表示障碍或不可达
    """
    H, W = goal_01.shape
    dist = -np.ones((H, W), dtype=np.int32)

    q = deque()

    # 将边界非障碍像素作为源点入队
    # 顶边和底边
    for x in range(W):
        if goal_01[0, x] == 0:
            dist[0, x] = 0
            q.append((0, x))
        if goal_01[H - 1, x] == 0:
            dist[H - 1, x] = 0
            q.append((H - 1, x))

    # 左边和右边（避免重复入队角点）
    for y in range(H):
        if goal_01[y, 0] == 0 and dist[y, 0] < 0:
            dist[y, 0] = 0
            q.append((y, 0))
        if goal_01[y, W - 1] == 0 and dist[y, W - 1] < 0:
            dist[y, W - 1] = 0
            q.append((y, W - 1))

    # 4邻域BFS
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1

        # 上
        if y > 0 and dist[y - 1, x] < 0 and goal_01[y - 1, x] == 0:
            dist[y - 1, x] = d
            q.append((y - 1, x))

        # 下
        if y + 1 < H and dist[y + 1, x] < 0 and goal_01[y + 1, x] == 0:
            dist[y + 1, x] = d
            q.append((y + 1, x))

        # 左
        if x > 0 and dist[y, x - 1] < 0 and goal_01[y, x - 1] == 0:
            dist[y, x - 1] = d
            q.append((y, x - 1))

        # 右
        if x + 1 < W and dist[y, x + 1] < 0 and goal_01[y, x + 1] == 0:
            dist[y, x + 1] = d
            q.append((y, x + 1))

    return dist


def m3_postprocess_dist(
    dist_int: np.ndarray,
    goal_01: ImgBin01,
    cfg_m3: Dict[str, Any]
) -> np.ndarray:
    """
    后处理距离场：填充障碍和不可达区域，归一化

    处理规则：
    1. 取可行域最大距离dist_max
    2. 障碍（goal==1）填充值为dist_max + margin
    3. 不可达像素（-1且非goal）同样填dist_max + margin
    4. 归一化：dist_norm = dist_filled / (dist_max + eps)

    Args:
        dist_int: BFS输出的距离场 int32
        goal_01: 目标区域二值图
        cfg_m3: M3配置字典

    Returns:
        归一化距离场 float32 shape=(224,224)
    """
    margin = float(cfg_m3["obstacle_fill_dist_margin"])

    # 找到可行域（dist >= 0）的最大距离
    valid = dist_int >= 0
    dist_max = float(dist_int[valid].max()) if np.any(valid) else 0.0

    # 填充值
    fill = dist_max + margin

    # 转为float并填充
    dist_f = dist_int.astype(np.float32)

    # 填充障碍区域
    dist_f[goal_01 == 1] = fill

    # 填充不可达区域（-1且不是障碍）
    dist_f[(dist_int < 0) & (goal_01 == 0)] = fill

    # 归一化
    eps = 1e-6
    if dist_max > 0:
        dist_norm = dist_f / (dist_max + eps)
    else:
        dist_norm = dist_f * 0.0

    return dist_norm.astype(np.float32)


def m3_flow_from_dist(dist_norm: np.ndarray, cfg_m3: Dict[str, Any]) -> np.ndarray:
    """
    从距离场计算Flow Field（梯度场）

    Flow定义：
    - 使用Sobel算子计算距离场梯度
    - Flow取负梯度（指向距离减小的方向，即指向边界）

    Args:
        dist_norm: 归一化距离场 float32
        cfg_m3: M3配置字典

    Returns:
        Flow Field float32 shape=(224,224,2)，[...,0]=fx, [...,1]=fy
    """
    ksize = int(cfg_m3["sobel_ksize"])

    # 计算梯度
    dx = cv2.Sobel(dist_norm, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(dist_norm, cv2.CV_32F, 0, 1, ksize=ksize)

    # Flow取负梯度
    fx = -dx
    fy = -dy

    # 堆叠为(H,W,2)
    flow = np.stack([fx, fy], axis=-1)

    return flow.astype(np.float32)


def m3_run(
    goal_01: ImgBin01,
    cfg_m3: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    M3模块主函数：计算障碍感知距离场和Flow Field

    Args:
        goal_01: 目标区域二值图 (0/1)，1表示需要保留积木的区域（作为障碍）
        cfg_m3: M3配置字典

    Returns:
        tuple:
            - dist_norm: 归一化距离场 float32 (224,224)
            - map_grad: Flow Field float32 (224,224,2)
            - stats: 统计信息字典
    """
    # 步骤1: BFS计算距离场
    dist_int = m3_bfs_distance_obstacle_aware(goal_01, cfg_m3)

    # 步骤2: 后处理距离场
    dist_norm = m3_postprocess_dist(dist_int, goal_01, cfg_m3)

    # 步骤3: 计算Flow Field
    map_grad = m3_flow_from_dist(dist_norm, cfg_m3)

    # 统计信息
    valid = dist_int >= 0
    dist_max = float(dist_int[valid].max()) if np.any(valid) else 0.0
    stats = dict(dist_max=dist_max)

    return dist_norm, map_grad, stats
