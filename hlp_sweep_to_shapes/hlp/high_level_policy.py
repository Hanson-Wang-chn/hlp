# -*- coding: utf-8 -*-
"""
High Level Policy 顶层类

负责调度M1~M5模块，管理缓存和日志，提供统一的推理接口
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional

from .types import ImgBGR, ImgBin01, ImgBin255, Mat33, HLPResult
from .config import validate_cfg, DEFAULT_CFG, get_table_corners_from_cfg, compute_perspective_transforms
from .cache import GoalCache
from .logger import make_logger, new_run_id, HLPLogger
from .io_utils import (
    ensure_dir, save_u8_png, bin01_to_255, float01_to_u8, hash_bin01
)
from .m1_rectify_segment import m1_run
from .m2_align_goal import m2_run, m2_normalize_goal_to01, m2_warp_goal
from .m3_flow_field import m3_run
from .m4_chunker import m4_run
from .m5_select_overlay import m5_run


class HighLevelPolicy:
    """
    High Level Policy 类

    在每一步根据当前主相机图像与二值目标图像，
    生成下一次最应该执行的sweep区域mask及其方向，
    并输出可视化叠加图供低层策略执行。
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        T1: Optional[np.ndarray] = None,
        logger: Optional[logging.Logger] = None,
        table_corners: Optional[np.ndarray] = None
    ):
        """
        初始化HLP

        Args:
            cfg: 配置字典（会自动校验和补齐默认值）
            T1: [已弃用] 3x3单应变换矩阵，保留以向后兼容。
                如果同时提供T1和table_corners，优先使用table_corners。
                如果都不提供，则从cfg中读取table_corners。
            logger: 可选的Logger对象，如不提供则内部创建
            table_corners: (4,2) 角点坐标数组，表示224x224主相机图像上的4个点，
                          顺序为：左上、右上、右下、左下。
                          这4个点围成的四边形将被透视变换到224x224的table视角图像。
        """
        # 校验并补齐配置
        self.cfg = validate_cfg(cfg)

        # 确定透视变换矩阵
        if table_corners is not None:
            # 使用传入的table_corners
            self.table_corners = np.array(table_corners, dtype=np.float32)
            self.T1, self.T1_inv = compute_perspective_transforms(self.table_corners)
        elif T1 is not None:
            # 向后兼容：使用传入的T1矩阵
            self.T1 = T1.astype(np.float64)
            self.T1_inv = np.linalg.inv(self.T1)
            self.table_corners = None  # 从T1无法反推corners
        else:
            # 从配置中读取table_corners
            self.table_corners = get_table_corners_from_cfg(self.cfg)
            self.T1, self.T1_inv = compute_perspective_transforms(self.table_corners)

        # 初始化缓存
        self.cache = GoalCache()

        # 初始化日志
        if logger is None:
            logger = make_logger(self.cfg)
        self.logger = logger

    def reset(self, new_goal_bin_table: Optional[np.ndarray] = None) -> None:
        """
        重置HLP状态

        清空与goal相关的缓存；如传入new_goal，则立即写入缓存。

        Args:
            new_goal_bin_table: 可选的新目标二值图
        """
        self.cache.clear_goal_related()

        if new_goal_bin_table is not None:
            # 标准化并缓存
            goal_01 = m2_normalize_goal_to01(new_goal_bin_table)
            goal_hash = hash_bin01(goal_01)
            self.cache.goal_hash = goal_hash

    def infer(
        self,
        img_rgb_main_cur_224: np.ndarray,
        img_bin_table_goal_224: np.ndarray,
        index_mask_exec: int = 0,
        step_id: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> HLPResult:
        """
        执行一次HLP推理

        Args:
            img_rgb_main_cur_224: 主相机图像 (224,224,3) uint8, BGR
            img_bin_table_goal_224: 目标二值图 (224,224) uint8 {0,1}或{0,255}
                                    1表示目标保留积木区域
            index_mask_exec: 选择执行的mask索引（默认0，即最高优先级）
            step_id: 步骤ID（用于日志和文件名）
            save_dir: 保存目录（可选）

        Returns:
            HLPResult字典，包含：
                - img_rgb_main_exec_224: 叠加可视化后的主相机图像
                - mask_main_exec_224: 选中执行区域的mask
                - meta: 元信息
                - debug: 调试信息
        """
        # 创建运行ID和日志器
        run_id = new_run_id()
        hlp_logger = HLPLogger(
            self.logger, run_id,
            structured=self.cfg["log"]["structured"]
        )

        # 记录总体开始时间
        t0 = time.time()
        hlp_logger.start("HLP", step_id)

        # 初始化结果字典
        debug_dict: Dict[str, Any] = {}
        meta: Dict[str, Any] = {
            "step_id": step_id,
            "run_id": run_id,
            "cache": {"m2_hit": False, "m3_hit": False},
            "counts": {},
        }

        try:
            # ==================== 准备工作 ====================
            # 创建保存目录
            if save_dir and self.cfg["debug"]["save_intermediate"]:
                ensure_dir(save_dir, exist_ok=self.cfg["io"]["mkdir_exist_ok"])

            # 标准化目标图像为0/1
            goal_01 = m2_normalize_goal_to01(img_bin_table_goal_224)
            goal_hash = hash_bin01(goal_01)

            # ==================== M1: 矫正 + 分割 ====================
            t_m1 = hlp_logger.start("M1", step_id)
            img_rgb_table, B_01 = m1_run(
                img_rgb_main_cur_224, self.T1, self.cfg["m1"]
            )
            m1_elapsed = hlp_logger.end(
                "M1", t_m1, step_id, "ok",
                {"red_pixels": int(B_01.sum())}
            )

            # 保存M1输出
            if save_dir and self.cfg["io"]["save_m1"]:
                m1_dir = f"{save_dir}/m1"
                ensure_dir(m1_dir)
                save_u8_png(f"{m1_dir}/img_rgb_table_cur.png", img_rgb_table)
                save_u8_png(f"{m1_dir}/img_bin_table_cur.png", bin01_to_255(B_01))

            # ==================== M2: 目标对齐 ====================
            t_m2 = hlp_logger.start("M2", step_id)

            # 检查缓存
            m2_hit = (
                self.cache.has_valid_m2_cache(goal_hash) and
                not self.cfg["m2"]["force_reopt_each_step"]
            )

            if m2_hit:
                # 使用缓存的T2和goal mask
                hlp_logger.cache_hit("M2", step_id)
                G_01 = self.cache.mask_table_goal_01
                T2 = self.cache.T2
                J_best = 0.0  # 缓存时不重新计算
                m2_stats = {}
            else:
                # 重新优化
                hlp_logger.cache_miss("M2", step_id)
                G_01, _, T2, J_best, m2_stats = m2_run(
                    B_01, goal_01, self.cfg["m2"]
                )
                # 更新缓存
                self.cache.update_m2_cache(goal_hash, T2, G_01)

            # residual每步必须重算（因为B会变）
            residual_01 = np.logical_and(B_01 == 1, G_01 == 0).astype(np.uint8)

            meta["cache"]["m2_hit"] = m2_hit
            m2_elapsed = hlp_logger.end(
                "M2", t_m2, step_id, "ok",
                {
                    "J": J_best,
                    "goal_pixels": int(G_01.sum()),
                    "residual_pixels": int(residual_01.sum()),
                    **m2_stats
                }
            )

            # 保存M2输出
            if save_dir and self.cfg["io"]["save_m2"]:
                m2_dir = f"{save_dir}/m2"
                ensure_dir(m2_dir)
                save_u8_png(f"{m2_dir}/mask_table_goal.png", bin01_to_255(G_01))
                save_u8_png(f"{m2_dir}/mask_table_residual.png", bin01_to_255(residual_01))

            # 检查residual是否为空
            if residual_01.sum() == 0:
                # 没有需要扫除的区域
                meta["status"] = "empty"
                meta["reason"] = "no residual to sweep"
                meta["counts"] = {
                    "cur_pixels": int(B_01.sum()),
                    "goal_pixels": int(G_01.sum()),
                    "residual_pixels": 0,
                    "chunks": 0
                }

                hlp_logger.end("HLP", t0, step_id, "empty", meta)

                return HLPResult(
                    img_rgb_main_exec_224=img_rgb_main_cur_224.copy(),
                    mask_main_exec_224=np.zeros((224, 224), dtype=np.uint8),
                    meta=meta,
                    debug=debug_dict
                )

            # ==================== M3: Flow Field ====================
            t_m3 = hlp_logger.start("M3", step_id)

            # 检查缓存
            m3_hit = m2_hit and self.cache.has_valid_m3_cache()

            if m3_hit:
                hlp_logger.cache_hit("M3", step_id)
                dist_norm = self.cache.dist_norm_f32
                map_grad = self.cache.map_grad_f32
                m3_stats = {}
            else:
                hlp_logger.cache_miss("M3", step_id)
                dist_norm, map_grad, m3_stats = m3_run(G_01, self.cfg["m3"])
                # 更新缓存
                self.cache.update_m3_cache(dist_norm, map_grad)

            meta["cache"]["m3_hit"] = m3_hit
            m3_elapsed = hlp_logger.end("M3", t_m3, step_id, "ok", m3_stats)

            # 保存M3输出
            if save_dir and self.cfg["io"]["save_m3"]:
                m3_dir = f"{save_dir}/m3"
                ensure_dir(m3_dir)
                # 归一化距离场可视化
                dist_vis = float01_to_u8(dist_norm / (dist_norm.max() + 1e-6))
                save_u8_png(f"{m3_dir}/dist_norm.png", dist_vis)
                # 梯度模长可视化
                mag = np.sqrt(map_grad[..., 0]**2 + map_grad[..., 1]**2)
                mag_vis = float01_to_u8(mag / (mag.max() + 1e-6))
                save_u8_png(f"{m3_dir}/mag.png", mag_vis)

            # ==================== M4: Chunking ====================
            t_m4 = hlp_logger.start("M4", step_id)
            chunks, m4_stats = m4_run(
                map_grad, residual_01, self.cfg["m4"],
                save_dir, self.cfg["io"]
            )
            m4_elapsed = hlp_logger.end("M4", t_m4, step_id, "ok", m4_stats)

            # 检查chunks是否为空
            if len(chunks) == 0:
                meta["status"] = "empty"
                meta["reason"] = "no valid chunks after filtering"
                meta["counts"] = {
                    "cur_pixels": int(B_01.sum()),
                    "goal_pixels": int(G_01.sum()),
                    "residual_pixels": int(residual_01.sum()),
                    "chunks": 0
                }

                hlp_logger.end("HLP", t0, step_id, "empty", meta)

                return HLPResult(
                    img_rgb_main_exec_224=img_rgb_main_cur_224.copy(),
                    mask_main_exec_224=np.zeros((224, 224), dtype=np.uint8),
                    meta=meta,
                    debug=debug_dict
                )

            # ==================== M5: Select + Overlay ====================
            t_m5 = hlp_logger.start("M5", step_id)
            mask_main_255, img_exec, m5_meta = m5_run(
                chunks, index_mask_exec,
                img_rgb_main_cur_224, self.T1_inv,
                self.cfg["m5"], save_dir, self.cfg["io"], step_id
            )
            m5_elapsed = hlp_logger.end(
                "M5", t_m5, step_id, "ok",
                {
                    "chosen_idx": m5_meta.get("chosen_idx_sorted"),
                    "direction": m5_meta.get("direction"),
                    "area": m5_meta.get("area"),
                    "priority_score": m5_meta.get("priority_score")
                }
            )

            # ==================== 汇总结果 ====================
            meta["status"] = m5_meta.get("status", "ok")
            meta["counts"] = {
                "cur_pixels": int(B_01.sum()),
                "goal_pixels": int(G_01.sum()),
                "residual_pixels": int(residual_01.sum()),
                "chunks": len(chunks)
            }
            meta["chosen"] = {
                "direction": m5_meta.get("direction"),
                "G_min": m5_meta.get("G_min"),
                "G_max": m5_meta.get("G_max"),
                "area": m5_meta.get("area"),
                "bbox": m5_meta.get("bbox"),
                "priority_score": m5_meta.get("priority_score"),
                "chosen_idx_sorted": m5_meta.get("chosen_idx_sorted")
            }

            # 总耗时
            total_elapsed = hlp_logger.end("HLP", t0, step_id, "ok", meta)

            # 返回结果
            if self.cfg["debug"]["return_intermediate"]:
                debug_dict.update({
                    "img_rgb_table": img_rgb_table,
                    "B_01": B_01,
                    "G_01": G_01,
                    "residual_01": residual_01,
                    "dist_norm": dist_norm,
                    "map_grad": map_grad,
                    "chunks": chunks,
                })

            return HLPResult(
                img_rgb_main_exec_224=img_exec,
                mask_main_exec_224=mask_main_255,
                meta=meta,
                debug=debug_dict
            )

        except Exception as e:
            # 异常处理
            meta["status"] = "fail"
            meta["reason"] = str(e)
            hlp_logger.error("HLP", str(e), step_id)

            return HLPResult(
                img_rgb_main_exec_224=img_rgb_main_cur_224.copy(),
                mask_main_exec_224=np.zeros((224, 224), dtype=np.uint8),
                meta=meta,
                debug=debug_dict
            )
