# -*- coding: utf-8 -*-
"""
I/O 工具模块
包含文件保存、目录创建、哈希计算等工具函数
"""

import os
import hashlib
import numpy as np
import cv2


def ensure_dir(path: str, exist_ok: bool = True) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        path: 目录路径
        exist_ok: 如果目录已存在是否忽略错误
    """
    os.makedirs(path, exist_ok=exist_ok)


def save_u8_png(path: str, img_u8: np.ndarray) -> None:
    """
    保存uint8图像为PNG格式

    Args:
        path: 保存路径
        img_u8: uint8类型图像，单通道或三通道
    """
    # 确保父目录存在
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_dir(parent_dir, exist_ok=True)
    cv2.imwrite(path, img_u8)


def bin01_to_255(img01: np.ndarray) -> np.ndarray:
    """
    将0/1二值图转换为0/255二值图

    Args:
        img01: 值域为{0,1}的二值图

    Returns:
        值域为{0,255}的二值图
    """
    return (img01.astype(np.uint8) * 255)


def bin255_to_01(img255: np.ndarray) -> np.ndarray:
    """
    将0/255二值图转换为0/1二值图

    Args:
        img255: 值域为{0,255}的二值图

    Returns:
        值域为{0,1}的二值图
    """
    return (img255 > 127).astype(np.uint8)


def float01_to_u8(img01: np.ndarray) -> np.ndarray:
    """
    将[0,1]浮点图像转换为[0,255] uint8图像

    Args:
        img01: 值域为[0,1]的浮点图像

    Returns:
        值域为[0,255]的uint8图像
    """
    x = np.clip(img01, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def hash_bin01(img01: np.ndarray) -> str:
    """
    计算二值图像的SHA1哈希值

    注意：计算前会将图像标准化为0/1，避免0/255与0/1造成误判

    Args:
        img01: 二值图像

    Returns:
        SHA1哈希字符串
    """
    # 确保是0/1格式
    img_normalized = (img01 > 0).astype(np.uint8)
    # 确保内存连续
    b = np.ascontiguousarray(img_normalized).tobytes()
    return hashlib.sha1(b).hexdigest()


def load_image_bgr(path: str) -> np.ndarray:
    """
    加载BGR格式图像

    Args:
        path: 图像路径

    Returns:
        BGR格式的numpy数组
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img


def load_image_gray(path: str) -> np.ndarray:
    """
    加载灰度图像

    Args:
        path: 图像路径

    Returns:
        灰度图像的numpy数组
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img


def resize_to_224(img: np.ndarray) -> np.ndarray:
    """
    将图像缩放到224x224

    Args:
        img: 输入图像

    Returns:
        224x224的图像
    """
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
