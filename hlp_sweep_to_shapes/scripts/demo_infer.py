#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_infer.py - HLP 最小可运行示例

演示如何使用 HighLevelPolicy 进行单步推理。

用法：
    python scripts/demo_infer.py \
        --main_img data/snapshot.png \
        --goal_img data/E.png \
        --config config.yaml \
        --output outputs/demo
"""

import os
import sys
import argparse
import numpy as np
import cv2

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hlp import (
    HighLevelPolicy, load_cfg_from_yaml, DEFAULT_CFG, validate_cfg,
    get_T1_from_cfg, load_cfg_and_T1,
    get_table_corners_from_cfg, compute_perspective_transforms, load_cfg_and_transforms
)


def load_table_corners(cfg: dict, corners_path: str = None) -> np.ndarray:
    """
    加载table_corners（4个角点坐标）

    优先级：
    1. 如果提供了 --corners 参数（.npy文件），则使用该文件
    2. 否则从 config.yaml 的 table_corners 字段读取

    Args:
        cfg: 配置字典
        corners_path: 角点坐标文件路径（.npy格式，可选）

    Returns:
        (4, 2) 角点坐标数组
    """
    if corners_path and os.path.exists(corners_path):
        # 优先使用命令行指定的.npy文件
        corners = np.load(corners_path)
        print(f"[INFO] 从 {corners_path} 加载 table_corners")
    else:
        # 从配置文件读取table_corners
        corners = get_table_corners_from_cfg(cfg)
        print("[INFO] 从 config.yaml 读取 table_corners")
    return corners


def load_T1(cfg: dict, t1_path: str = None) -> np.ndarray:
    """
    [已弃用] 加载T1单应变换矩阵

    优先级：
    1. 如果提供了 --t1 参数（.npy文件），则使用该文件
    2. 否则从 config.yaml 的 table_corners 字段计算

    Args:
        cfg: 配置字典
        t1_path: T1矩阵文件路径（.npy格式，可选）

    Returns:
        3x3单应变换矩阵
    """
    if t1_path and os.path.exists(t1_path):
        # 优先使用命令行指定的.npy文件
        T1 = np.load(t1_path)
        print(f"[INFO] 从 {t1_path} 加载 T1 矩阵")
    else:
        # 从配置文件读取table_corners并计算T1
        T1 = get_T1_from_cfg(cfg)
        print("[INFO] 从 config.yaml 计算 T1 矩阵")
    return T1


def load_image(path: str, resize_to_224: bool = True) -> np.ndarray:
    """
    加载图像

    Args:
        path: 图像路径
        resize_to_224: 是否调整大小为224x224

    Returns:
        加载的图像数组
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像文件不存在: {path}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")

    if resize_to_224:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    return img


def main():
    parser = argparse.ArgumentParser(
        description="HLP 单步推理演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/demo_infer.py --main_img data/snapshot.png --goal_img data/E.png
    python scripts/demo_infer.py --main_img data/snapshot.png --goal_img data/E.png --config config.yaml
    python scripts/demo_infer.py --main_img data/snapshot.png --goal_img data/E.png --output outputs/demo --step_id 0
        """
    )

    parser.add_argument(
        "--main_img", "-m",
        type=str,
        required=True,
        help="主相机图像路径（BGR格式）"
    )
    parser.add_argument(
        "--goal_img", "-g",
        type=str,
        required=True,
        help="目标二值图像路径（1=保留区域）"
    )
    parser.add_argument(
        "--corners", "-r",
        type=str,
        default=None,
        help="table_corners角点坐标文件路径（.npy格式，shape=(4,2)），不提供则从配置读取"
    )
    parser.add_argument(
        "--t1", "-t",
        type=str,
        default=None,
        help="[已弃用] T1单应矩阵文件路径（.npy格式），建议使用 --corners 代替"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径（YAML格式），不提供则使用默认配置"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="输出目录路径（默认: outputs）"
    )
    parser.add_argument(
        "--step_id", "-s",
        type=int,
        default=0,
        help="步骤ID（默认: 0）"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        help="选择执行的mask索引（默认: 0，即最高优先级）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HLP - High Level Policy 单步推理演示")
    print("=" * 60)

    # 加载配置
    if args.config and os.path.exists(args.config):
        print(f"[INFO] 从 {args.config} 加载配置")
        cfg = load_cfg_from_yaml(args.config)
    else:
        print("[INFO] 使用默认配置")
        cfg = validate_cfg({})

    # 加载table_corners或T1
    # 优先使用 --corners，其次使用 --t1，最后从 config 读取
    if args.corners:
        table_corners = load_table_corners(cfg, args.corners)
        T1, T1_inv = compute_perspective_transforms(table_corners)
    elif args.t1:
        T1 = load_T1(cfg, args.t1)
        table_corners = None
    else:
        table_corners = load_table_corners(cfg, None)
        T1, T1_inv = compute_perspective_transforms(table_corners)

    # 加载图像
    print(f"[INFO] 加载主相机图像: {args.main_img}")
    main_img = load_image(args.main_img, resize_to_224=True)
    if len(main_img.shape) == 2:
        # 灰度图转BGR
        main_img = cv2.cvtColor(main_img, cv2.COLOR_GRAY2BGR)
    elif main_img.shape[2] == 4:
        # BGRA转BGR
        main_img = cv2.cvtColor(main_img, cv2.COLOR_BGRA2BGR)

    print(f"[INFO] 加载目标图像: {args.goal_img}")
    goal_img = load_image(args.goal_img, resize_to_224=True)
    if len(goal_img.shape) == 3:
        # 彩色图转灰度
        goal_img = cv2.cvtColor(goal_img, cv2.COLOR_BGR2GRAY)

    print(f"[INFO] 主相机图像尺寸: {main_img.shape}")
    print(f"[INFO] 目标图像尺寸: {goal_img.shape}")

    # 创建HLP实例
    print("[INFO] 初始化 HighLevelPolicy...")
    if args.corners or (not args.t1):
        # 使用 table_corners（推荐）
        hlp = HighLevelPolicy(cfg, table_corners=table_corners)
    else:
        # 向后兼容：使用 T1
        hlp = HighLevelPolicy(cfg, T1=T1)

    # 执行推理
    print(f"[INFO] 执行推理 (step_id={args.step_id}, index={args.index})...")
    result = hlp.infer(
        img_rgb_main_cur_224=main_img,
        img_bin_table_goal_224=goal_img,
        index_mask_exec=args.index,
        step_id=args.step_id,
        save_dir=args.output
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)

    meta = result["meta"]
    print(f"状态: {meta.get('status', 'unknown')}")

    if meta.get("status") == "ok":
        chosen = meta.get("chosen", {})
        print(f"扫除方向: {chosen.get('direction')}")
        print(f"区域面积: {chosen.get('area')}")
        print(f"优先级分数: {chosen.get('priority_score'):.4f}")
        print(f"梯度范围: [{chosen.get('G_min'):.4f}, {chosen.get('G_max'):.4f}]")
        print(f"边界框: {chosen.get('bbox')}")

        counts = meta.get("counts", {})
        print(f"\n统计信息:")
        print(f"  - 当前积木像素数: {counts.get('cur_pixels')}")
        print(f"  - 目标区域像素数: {counts.get('goal_pixels')}")
        print(f"  - 需扫除像素数: {counts.get('residual_pixels')}")
        print(f"  - 可执行区块数: {counts.get('chunks')}")

        cache = meta.get("cache", {})
        print(f"\n缓存状态:")
        print(f"  - M2缓存命中: {cache.get('m2_hit')}")
        print(f"  - M3缓存命中: {cache.get('m3_hit')}")
    elif meta.get("status") == "empty":
        print(f"原因: {meta.get('reason')}")
    else:
        print(f"错误原因: {meta.get('reason')}")

    # 保存最终输出
    output_path = os.path.join(args.output, "result_overlay.png")
    cv2.imwrite(output_path, result["img_rgb_main_exec_224"])
    print(f"\n[INFO] 结果图像已保存到: {output_path}")

    mask_path = os.path.join(args.output, "result_mask.png")
    cv2.imwrite(mask_path, result["mask_main_exec_224"])
    print(f"[INFO] 结果mask已保存到: {mask_path}")

    print("\n[INFO] 完成!")


if __name__ == "__main__":
    main()
