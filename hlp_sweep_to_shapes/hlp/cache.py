# -*- coding: utf-8 -*-
"""
缓存模块
管理与goal相关的缓存对象，避免重复计算
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .types import Mat23, ImgBin01


@dataclass
class GoalCache:
    """
    目标图像相关的缓存对象

    用于缓存M2优化得到的T2变换矩阵、mask_table_goal，
    以及M3计算得到的距离场和梯度场
    """

    # 目标图像哈希值
    goal_hash: Optional[str] = None

    # M2缓存：优化后的变换矩阵和目标mask
    T2: Optional[Mat23] = None
    mask_table_goal_01: Optional[ImgBin01] = None

    # M3缓存：距离场和梯度场
    dist_norm_f32: Optional[np.ndarray] = None
    map_grad_f32: Optional[np.ndarray] = None

    def clear_goal_related(self) -> None:
        """
        清空所有与goal相关的缓存
        """
        self.goal_hash = None
        self.T2 = None
        self.mask_table_goal_01 = None
        self.dist_norm_f32 = None
        self.map_grad_f32 = None

    def clear_m3_cache(self) -> None:
        """
        仅清空M3相关缓存
        """
        self.dist_norm_f32 = None
        self.map_grad_f32 = None

    def update_m2_cache(
        self,
        goal_hash: str,
        T2: Mat23,
        mask_table_goal_01: ImgBin01
    ) -> None:
        """
        更新M2缓存

        Args:
            goal_hash: 目标图像哈希值
            T2: 优化后的变换矩阵
            mask_table_goal_01: 变换后的目标mask
        """
        self.goal_hash = goal_hash
        self.T2 = T2.copy()
        self.mask_table_goal_01 = mask_table_goal_01.copy()
        # M2更新后需要清空M3缓存
        self.clear_m3_cache()

    def update_m3_cache(
        self,
        dist_norm_f32: np.ndarray,
        map_grad_f32: np.ndarray
    ) -> None:
        """
        更新M3缓存

        Args:
            dist_norm_f32: 归一化距离场
            map_grad_f32: 梯度场
        """
        self.dist_norm_f32 = dist_norm_f32.copy()
        self.map_grad_f32 = map_grad_f32.copy()

    def has_valid_m2_cache(self, current_goal_hash: str) -> bool:
        """
        检查M2缓存是否有效

        Args:
            current_goal_hash: 当前目标图像的哈希值

        Returns:
            缓存是否有效
        """
        return (
            self.goal_hash is not None and
            self.goal_hash == current_goal_hash and
            self.T2 is not None and
            self.mask_table_goal_01 is not None
        )

    def has_valid_m3_cache(self) -> bool:
        """
        检查M3缓存是否有效

        Returns:
            缓存是否有效
        """
        return (
            self.dist_norm_f32 is not None and
            self.map_grad_f32 is not None
        )
