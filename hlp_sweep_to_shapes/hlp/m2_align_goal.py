# -*- coding: utf-8 -*-
"""
M2: 目标对齐优化模块

负责优化相似变换 T2，使得变换后的目标图案与当前积木堆尽可能对齐。
完全同步自 Residual-Perception-Preprocessor/hlp_preprocessor.py 的优化算法。

代价函数: J(T) = λ1·C_fill + λ2·C_remove + λ3·C_edge + λ4·C_sweep + Reg(T)
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
from scipy.optimize import minimize, differential_evolution

from .types import ImgBin01, Mat23


# ============================================================================
# 数据类定义（同步自 RPP）
# ============================================================================

@dataclass
class TransformParams:
    """变换参数：平移、旋转、缩放"""
    tx: float = 0.0      # 平移 x（像素）
    ty: float = 0.0      # 平移 y（像素）
    theta: float = 0.0   # 旋转角度（弧度）
    scale: float = 1.0   # 缩放因子

    def to_array(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.theta, self.scale])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TransformParams':
        return cls(tx=arr[0], ty=arr[1], theta=arr[2], scale=arr[3])


# ============================================================================
# Mask 平滑（完全同步 RPP 的 MaskSmoother）
# ============================================================================

def m2_prepare_pile_mask(B_01: ImgBin01, cfg_m2: Dict[str, Any]) -> ImgBin01:
    """
    平滑积木堆 mask（同步 RPP 的 MaskSmoother.smooth）

    Args:
        B_01: 二值积木堆 mask (H, W), dtype=uint8, 值为0/1
        cfg_m2: M2 配置字典

    Returns:
        平滑后的 mask (H, W), dtype=uint8, 值为0/1
    """
    kernel_size = cfg_m2.get("smooth_kernel", 5)
    gaussian_sigma = cfg_m2.get("smooth_sigma", 2.0)
    morph_iterations = cfg_m2.get("smooth_morph_iterations", 2)
    use_closing = cfg_m2.get("smooth_use_closing", True)
    use_opening = cfg_m2.get("smooth_use_opening", True)

    result = B_01.copy()

    # 创建形态学核（同 RPP：椭圆形）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # 形态学闭运算（填充小孔）
    if use_closing:
        result = cv2.morphologyEx(
            result, cv2.MORPH_CLOSE, kernel,
            iterations=morph_iterations
        )

    # 形态学开运算（去除小噪点）
    if use_opening:
        result = cv2.morphologyEx(
            result, cv2.MORPH_OPEN, kernel,
            iterations=morph_iterations
        )

    # Gaussian 模糊 + 重新二值化
    if gaussian_sigma > 0:
        result = cv2.GaussianBlur(
            result.astype(np.float32),
            (0, 0),
            gaussian_sigma
        )
        result = (result > 0.5).astype(np.uint8)

    return result


# ============================================================================
# 目标图像预处理
# ============================================================================

def m2_normalize_goal_to01(img_bin_table_goal_224: np.ndarray) -> ImgBin01:
    """
    将目标二值图标准化为 0/1 格式

    Args:
        img_bin_table_goal_224: 目标二值图 (224,224)，可能是 {0,1} 或 {0,255}

    Returns:
        标准化后的 0/1 二值图 (224,224), dtype=uint8
    """
    goal = img_bin_table_goal_224.astype(np.float32)
    if goal.max() > 1:
        goal = goal / 255.0
    return (goal > 0.5).astype(np.uint8)


# ============================================================================
# 变换矩阵计算（同步 RPP 的 GoalImageProcessor.transform_mask）
# ============================================================================

def m2_params_to_T2(tx: float, ty: float, theta: float, scale: float) -> Mat23:
    """
    将变换参数转换为 2x3 仿射变换矩阵

    变换顺序：缩放 -> 旋转 -> 平移（相对于图像中心）

    Args:
        tx: 平移 x（像素）
        ty: 平移 y（像素）
        theta: 旋转角度（弧度）
        scale: 缩放因子

    Returns:
        2x3 仿射变换矩阵, dtype=float64
    """
    # 图像中心（224x224）
    cx, cy = 112.0, 112.0

    # 构建变换矩阵：先缩放旋转，再平移
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 缩放 + 旋转矩阵（相对于原点）
    # [s*cos, -s*sin]
    # [s*sin,  s*cos]
    a = scale * cos_t
    b = -scale * sin_t
    c = scale * sin_t
    d = scale * cos_t

    # 平移分量：保持图像中心不变 + 额外平移
    # 新中心 = 旧中心 + (tx, ty)
    # 变换后的点: p' = R*S*(p - center) + center + (tx, ty)
    #            p' = R*S*p - R*S*center + center + (tx, ty)
    # 所以平移分量 = -R*S*center + center + (tx, ty)
    tx_final = -a * cx - b * cy + cx + tx
    ty_final = -c * cx - d * cy + cy + ty

    T2 = np.array([
        [a, b, tx_final],
        [c, d, ty_final]
    ], dtype=np.float64)

    return T2


def m2_warp_goal(goal_01: ImgBin01, params: TransformParams) -> ImgBin01:
    """
    使用与 RPP 同步的几何逻辑（_transform_mask_rpp）变换目标 mask

    Args:
        goal_01: 目标 mask (224,224), dtype=uint8, 值为0/1
        params: 变换参数

    Returns:
        变换后的 mask (224,224), dtype=uint8, 值为0/1
    """
    return _transform_mask_rpp(goal_01, params, goal_01.shape)


def _transform_mask_rpp(mask: np.ndarray, params: TransformParams,
                        output_shape: Tuple[int, int]) -> np.ndarray:
    """
    变换 mask（完全同步 RPP 的 GoalImageProcessor.transform_mask）

    这是 RPP 中的原始实现，用于保证与 RPP 完全一致的行为。

    Args:
        mask: 输入 mask
        params: 变换参数
        output_shape: 输出形状 (H, W)

    Returns:
        变换后的 mask
    """
    h_out, w_out = output_shape
    h_in, w_in = mask.shape

    # 首先缩放 mask
    if params.scale != 1.0:
        new_h = int(h_in * params.scale)
        new_w = int(w_in * params.scale)
        scaled = cv2.resize(mask.astype(np.float32), (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)
    else:
        scaled = mask.astype(np.float32)
        new_h, new_w = h_in, w_in

    # 创建输出画布
    result = np.zeros((h_out, w_out), dtype=np.float32)

    # 计算缩放后 mask 的中心
    cx_scaled = new_w / 2
    cy_scaled = new_h / 2

    # 计算输出中的中心位置
    cx_out = w_out / 2 + params.tx
    cy_out = h_out / 2 + params.ty

    # 创建绕中心的旋转矩阵
    M = cv2.getRotationMatrix2D((cx_scaled, cy_scaled), np.degrees(params.theta), 1.0)

    # 旋转缩放后的 mask
    rotated = cv2.warpAffine(scaled, M, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 计算放置坐标
    x_start = int(cx_out - cx_scaled)
    y_start = int(cy_out - cy_scaled)

    # 计算有效的复制区域
    src_x_start = max(0, -x_start)
    src_y_start = max(0, -y_start)
    src_x_end = min(new_w, w_out - x_start)
    src_y_end = min(new_h, h_out - y_start)

    dst_x_start = max(0, x_start)
    dst_y_start = max(0, y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    # 复制有效区域
    if src_x_end > src_x_start and src_y_end > src_y_start:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            rotated[src_y_start:src_y_end, src_x_start:src_x_end]

    return (result > 0.5).astype(np.uint8)


# ============================================================================
# 代价函数（完全同步 RPP 的 CostFunction）
# ============================================================================

class CostFunction:
    """
    优化代价函数（完全同步 RPP）

    J(T) = λ1·C_fill + λ2·C_remove + λ3·C_edge + λ4·C_sweep + Reg(T)
    """

    def __init__(self, cfg_m2: Dict[str, Any]):
        """
        初始化代价函数

        Args:
            cfg_m2: M2 配置字典
        """
        self.lambda_fill = cfg_m2.get("lambda_fill", 2.0)
        self.lambda_remove = cfg_m2.get("lambda_remove", 1.0)
        self.lambda_edge = cfg_m2.get("lambda_edge", 0.0)
        self.lambda_sweep = cfg_m2.get("lambda_sweep", 0.0)
        self.reg_tx = cfg_m2.get("reg_tx", 0.0)
        self.reg_ty = cfg_m2.get("reg_ty", 0.0)
        self.reg_theta = cfg_m2.get("reg_theta", 0.0)
        self.reg_scale = cfg_m2.get("reg_scale", 0.0)
        self.reg_scale_ref = cfg_m2.get("reg_scale_ref", 1.0)
        self.sigma_edge = cfg_m2.get("sigma_edge", 10.0)
        self.alpha_sweep = cfg_m2.get("alpha_sweep", 2.0)

        self._pile_mask: Optional[np.ndarray] = None
        self._distance_field: Optional[np.ndarray] = None
        self._pile_edges: Optional[np.ndarray] = None

    def set_pile_mask(self, mask: np.ndarray) -> None:
        """
        设置积木堆 mask 并预计算相关量

        Args:
            mask: 积木堆二值 mask (H, W)
        """
        self._pile_mask = mask

        # 计算距离场（到最近边界的距离）
        self._distance_field = ndimage.distance_transform_edt(mask)

        # 计算边缘像素
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        self._pile_edges = dilated - eroded

    def compute_fill_cost(self, goal_mask: np.ndarray) -> float:
        """
        计算填充代价：目标区域内但不在积木堆中的像素

        C_fill = Σ I(p ∈ M_goal ∧ p ∉ M_pile)
        """
        fill_region = goal_mask.astype(bool) & (~self._pile_mask.astype(bool))
        return float(np.sum(fill_region))

    def compute_remove_cost(self, goal_mask: np.ndarray) -> float:
        """
        计算移除代价：积木堆中但不在目标区域的像素

        C_remove = Σ I(p ∉ M_goal ∧ p ∈ M_pile)
        """
        remove_region = (~goal_mask.astype(bool)) & self._pile_mask.astype(bool)
        return float(np.sum(remove_region))

    def compute_edge_cost(self, goal_mask: np.ndarray) -> float:
        """
        计算边缘对齐代价（负值表示奖励）

        C_edge = -Σ exp(-min_dist²/(2σ²))
        """
        # 获取目标边缘
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(goal_mask, kernel, iterations=1)
        eroded = cv2.erode(goal_mask, kernel, iterations=1)
        goal_edges = dilated - eroded

        # 找到目标边缘点
        goal_edge_points = np.argwhere(goal_edges > 0)

        if len(goal_edge_points) == 0:
            return 0.0

        # 找到积木堆边缘点
        pile_edge_points = np.argwhere(self._pile_edges > 0)

        if len(pile_edge_points) == 0:
            return 0.0

        # 使用距离变换计算最小距离
        pile_edge_mask = self._pile_edges > 0
        dist_from_pile_edge = ndimage.distance_transform_edt(~pile_edge_mask)

        # 获取目标边缘点的距离
        distances = dist_from_pile_edge[goal_edge_points[:, 0], goal_edge_points[:, 1]]

        # 计算高斯加权奖励
        sigma = self.sigma_edge
        rewards = np.exp(-distances**2 / (2 * sigma**2))

        return -float(np.sum(rewards))

    def compute_sweep_cost(self, goal_mask: np.ndarray) -> float:
        """
        计算扫除难度代价

        C_sweep = Σ [I(p ∉ M_goal ∧ p ∈ M_pile) · D_pile(p)^α]
        """
        # 找到需要移除的像素
        remove_region = (~goal_mask.astype(bool)) & self._pile_mask.astype(bool)

        # 按距离场加权
        alpha = self.alpha_sweep
        weighted_cost = np.sum(
            remove_region * (self._distance_field ** alpha)
        )

        return float(weighted_cost)

    def compute_regularization(self, params: Optional[TransformParams]) -> float:
        """
        计算变换参数的正则化项

        Reg(T) = ρ_tx·tx² + ρ_ty·ty² + ρ_theta·theta² + ρ_scale·(scale-ref)²
        """
        if params is None:
            return 0.0

        tx_penalty = self.reg_tx * (params.tx ** 2)
        ty_penalty = self.reg_ty * (params.ty ** 2)
        theta_penalty = self.reg_theta * (params.theta ** 2)
        scale_delta = params.scale - self.reg_scale_ref
        scale_penalty = self.reg_scale * (scale_delta ** 2)

        return float(tx_penalty + ty_penalty + theta_penalty + scale_penalty)

    def compute_total_cost(self, goal_mask: np.ndarray, params: Optional[TransformParams] = None) -> float:
        """
        计算总代价

        J(T) = λ1·C_fill + λ2·C_remove + λ3·C_edge + λ4·C_sweep + Reg(T)
        """
        c_fill = self.compute_fill_cost(goal_mask)
        c_remove = self.compute_remove_cost(goal_mask)
        c_edge = self.compute_edge_cost(goal_mask)
        c_sweep = self.compute_sweep_cost(goal_mask)
        c_reg = self.compute_regularization(params)

        total = (
            self.lambda_fill * c_fill +
            self.lambda_remove * c_remove +
            self.lambda_edge * c_edge +
            self.lambda_sweep * c_sweep +
            c_reg
        )

        return total

    def compute_cost_breakdown(self, goal_mask: np.ndarray, params: Optional[TransformParams] = None) -> Dict[str, float]:
        """获取详细的代价分解"""
        c_fill = self.compute_fill_cost(goal_mask)
        c_remove = self.compute_remove_cost(goal_mask)
        c_edge = self.compute_edge_cost(goal_mask)
        c_sweep = self.compute_sweep_cost(goal_mask)
        c_reg = self.compute_regularization(params)

        return {
            'fill': c_fill,
            'remove': c_remove,
            'edge': c_edge,
            'sweep': c_sweep,
            'reg': c_reg,
            'total': (
                self.lambda_fill * c_fill +
                self.lambda_remove * c_remove +
                self.lambda_edge * c_edge +
                self.lambda_sweep * c_sweep +
                c_reg
            )
        }


# ============================================================================
# 优化器（完全同步 RPP）
# ============================================================================

def _grid_search_optimize(
    pile_mask: np.ndarray,
    goal_mask: np.ndarray,
    cfg_m2: Dict[str, Any]
) -> Tuple[TransformParams, float]:
    """
    网格搜索优化器（同步 RPP 的 GridSearchOptimizer）
    """
    cost_fn = CostFunction(cfg_m2)
    cost_fn.set_pile_mask(pile_mask)

    h, w = pile_mask.shape
    res = cfg_m2.get("grid_resolution", 10)

    tx_max_cfg = cfg_m2.get("tx_abs_max")
    ty_max_cfg = cfg_m2.get("ty_abs_max")
    tx_max = w / 3 if tx_max_cfg is None else float(tx_max_cfg)
    ty_max = h / 3 if ty_max_cfg is None else float(ty_max_cfg)

    # 获取参数范围
    theta_range = cfg_m2.get("theta_range", [-np.pi/4, np.pi/4])
    scale_min = cfg_m2.get("scale_min", cfg_m2.get("bounds", {}).get("scale_low", 0.3))
    scale_max = cfg_m2.get("scale_max", cfg_m2.get("bounds", {}).get("scale_high", 1.5))

    # 定义搜索范围
    tx_range = np.linspace(-tx_max, tx_max, res)
    ty_range = np.linspace(-ty_max, ty_max, res)
    theta_vals = np.linspace(theta_range[0], theta_range[1], max(5, res // 4))
    scale_vals = np.linspace(scale_min, scale_max, max(5, res // 4))

    best_cost = float('inf')
    best_params = TransformParams()

    # 网格搜索
    for scale in scale_vals:
        for theta in theta_vals:
            for tx in tx_range:
                for ty in ty_range:
                    params = TransformParams(tx=tx, ty=ty, theta=theta, scale=scale)
                    transformed = _transform_mask_rpp(goal_mask, params, pile_mask.shape)
                    cost = cost_fn.compute_total_cost(transformed, params)

                    if cost < best_cost:
                        best_cost = cost
                        best_params = params

    return best_params, best_cost


def _scipy_optimize(
    pile_mask: np.ndarray,
    goal_mask: np.ndarray,
    cfg_m2: Dict[str, Any],
    x0: Optional[np.ndarray] = None
) -> Tuple[TransformParams, float]:
    """
    Scipy L-BFGS-B 优化器（同步 RPP 的 ScipyOptimizer）
    """
    cost_fn = CostFunction(cfg_m2)
    cost_fn.set_pile_mask(pile_mask)

    h, w = pile_mask.shape

    tx_max_cfg = cfg_m2.get("tx_abs_max")
    ty_max_cfg = cfg_m2.get("ty_abs_max")
    tx_max = w / 2 if tx_max_cfg is None else float(tx_max_cfg)
    ty_max = h / 2 if ty_max_cfg is None else float(ty_max_cfg)

    # 获取参数范围
    theta_range = cfg_m2.get("theta_range", [-np.pi/4, np.pi/4])
    scale_min = cfg_m2.get("scale_min", cfg_m2.get("bounds", {}).get("scale_low", 0.3))
    scale_max = cfg_m2.get("scale_max", cfg_m2.get("bounds", {}).get("scale_high", 1.5))

    def objective(x):
        params = TransformParams.from_array(x)
        transformed = _transform_mask_rpp(goal_mask, params, pile_mask.shape)
        return cost_fn.compute_total_cost(transformed, params)

    # 定义边界
    bounds = [
        (-tx_max, tx_max),      # tx
        (-ty_max, ty_max),      # ty
        tuple(theta_range),  # theta
        (scale_min, scale_max)  # scale
    ]

    # 初始猜测
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0, 1.0])

    maxiter = cfg_m2.get("local_maxiter", 2000)

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter}
    )

    best_params = TransformParams.from_array(result.x)
    return best_params, result.fun


def _differential_evolution_optimize(
    pile_mask: np.ndarray,
    goal_mask: np.ndarray,
    cfg_m2: Dict[str, Any]
) -> Tuple[TransformParams, float]:
    """
    差分进化优化器（同步 RPP 的 DifferentialEvolutionOptimizer）
    """
    cost_fn = CostFunction(cfg_m2)
    cost_fn.set_pile_mask(pile_mask)

    h, w = pile_mask.shape

    tx_max_cfg = cfg_m2.get("tx_abs_max")
    ty_max_cfg = cfg_m2.get("ty_abs_max")
    tx_max = w / 2 if tx_max_cfg is None else float(tx_max_cfg)
    ty_max = h / 2 if ty_max_cfg is None else float(ty_max_cfg)

    # 获取参数范围
    theta_range = cfg_m2.get("theta_range", [-np.pi/4, np.pi/4])
    scale_min = cfg_m2.get("scale_min", cfg_m2.get("bounds", {}).get("scale_low", 0.3))
    scale_max = cfg_m2.get("scale_max", cfg_m2.get("bounds", {}).get("scale_high", 1.5))

    def objective(x):
        params = TransformParams.from_array(x)
        transformed = _transform_mask_rpp(goal_mask, params, pile_mask.shape)
        return cost_fn.compute_total_cost(transformed, params)

    # 定义边界
    bounds = [
        (-tx_max, tx_max),      # tx
        (-ty_max, ty_max),      # ty
        tuple(theta_range),  # theta
        (scale_min, scale_max)  # scale
    ]

    maxiter = cfg_m2.get("maxiter", 1000)
    seed = cfg_m2.get("seed", 42)

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        seed=seed,
        workers=1,
        updating='deferred'
    )

    best_params = TransformParams.from_array(result.x)
    return best_params, result.fun


def _hybrid_optimize(
    pile_mask: np.ndarray,
    goal_mask: np.ndarray,
    cfg_m2: Dict[str, Any]
) -> Tuple[TransformParams, float]:
    """
    混合优化器：粗网格搜索 + 局部优化（同步 RPP 的 HybridOptimizer）
    """
    h, w = pile_mask.shape

    tx_max_cfg = cfg_m2.get("tx_abs_max")
    ty_max_cfg = cfg_m2.get("ty_abs_max")
    tx_max = w / 2 if tx_max_cfg is None else float(tx_max_cfg)
    ty_max = h / 2 if ty_max_cfg is None else float(ty_max_cfg)

    # 获取参数范围
    theta_range = cfg_m2.get("theta_range", [-np.pi/4, np.pi/4])
    scale_min = cfg_m2.get("scale_min", cfg_m2.get("bounds", {}).get("scale_low", 0.3))
    scale_max = cfg_m2.get("scale_max", cfg_m2.get("bounds", {}).get("scale_high", 1.5))

    # 第一步：粗网格搜索
    coarse_cfg = cfg_m2.copy()
    coarse_cfg["grid_resolution"] = cfg_m2.get("grid_resolution", 10)  # 粗网格
    coarse_cfg["tx_abs_max"] = tx_max
    coarse_cfg["ty_abs_max"] = ty_max

    coarse_params, _ = _grid_search_optimize(pile_mask, goal_mask, coarse_cfg)

    # 第二步：局部优化
    cost_fn = CostFunction(cfg_m2)
    cost_fn.set_pile_mask(pile_mask)

    def objective(x):
        params = TransformParams.from_array(x)
        transformed = _transform_mask_rpp(goal_mask, params, pile_mask.shape)
        return cost_fn.compute_total_cost(transformed, params)

    # 在粗解周围收窄边界
    tx_span = min(w / 10, tx_max)
    ty_span = min(h / 10, ty_max)
    bounds = [
        (max(-tx_max, coarse_params.tx - tx_span), min(tx_max, coarse_params.tx + tx_span)),
        (max(-ty_max, coarse_params.ty - ty_span), min(ty_max, coarse_params.ty + ty_span)),
        (coarse_params.theta - np.pi/8, coarse_params.theta + np.pi/8),
        (max(scale_min, coarse_params.scale * 0.8),
         min(scale_max, coarse_params.scale * 1.2))
    ]

    maxiter = cfg_m2.get("local_maxiter", 2000)

    result = minimize(
        objective,
        coarse_params.to_array(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter}
    )

    best_params = TransformParams.from_array(result.x)
    return best_params, result.fun


# ============================================================================
# M2 主入口函数
# ============================================================================

def m2_compute_cost(
    B_01: ImgBin01,
    G_01: ImgBin01,
    lambda1: float,
    lambda2: float
) -> Tuple[float, float, float]:
    """
    计算简化的代价函数（仅 fill 和 remove）

    Args:
        B_01: 积木堆 mask (224,224), dtype=uint8, 值为0/1
        G_01: 目标 mask (224,224), dtype=uint8, 值为0/1
        lambda1: fill 权重
        lambda2: remove 权重

    Returns:
        (J_total, C_fill, C_remove)
    """
    B_bool = B_01.astype(bool)
    G_bool = G_01.astype(bool)

    C_fill = float(np.sum(G_bool & (~B_bool)))
    C_remove = float(np.sum((~G_bool) & B_bool))
    J_total = lambda1 * C_fill + lambda2 * C_remove

    return J_total, C_fill, C_remove


def m2_optimize_T2_de(
    B_01: ImgBin01,
    goal_01: ImgBin01,
    cfg_m2: Dict[str, Any]
) -> Tuple[Mat23, ImgBin01, float, float, float]:
    """
    使用差分进化优化 T2

    Args:
        B_01: 积木堆 mask (224,224), dtype=uint8, 值为0/1
        goal_01: 目标 mask (已标准化) (224,224), dtype=uint8, 值为0/1
        cfg_m2: M2 配置字典

    Returns:
        (T2, G_01, J_best, C_fill, C_remove)
    """
    # 平滑积木堆 mask
    pile_smoothed = m2_prepare_pile_mask(B_01, cfg_m2)

    # 优化
    best_params, best_cost = _differential_evolution_optimize(
        pile_smoothed, goal_01, cfg_m2
    )

    # 变换目标（与 RPP 一致的几何逻辑）
    G_01 = m2_warp_goal(goal_01, best_params)

    # 计算最终代价（含正则）
    cost_fn = CostFunction(cfg_m2)
    cost_fn.set_pile_mask(pile_smoothed)
    breakdown = cost_fn.compute_cost_breakdown(G_01, best_params)
    J_best = breakdown["total"]
    C_fill = breakdown["fill"]
    C_remove = breakdown["remove"]
    best_cost = float(J_best)

    # 构建 T2 矩阵（向后兼容其他调用需要矩阵的场景）
    T2 = m2_params_to_T2(
        best_params.tx, best_params.ty,
        best_params.theta, best_params.scale
    )

    return T2, G_01, J_best, C_fill, C_remove


def m2_run(
    B_01: ImgBin01,
    goal_any: np.ndarray,
    cfg_m2: Dict[str, Any]
) -> Tuple[ImgBin01, ImgBin01, Mat23, float, Dict[str, Any]]:
    """
    M2 模块主入口：目标对齐优化

    Args:
        B_01: 积木堆 mask (224,224), dtype=uint8, 值为0/1
        goal_any: 目标图像，可能是 {0,1} 或 {0,255}
        cfg_m2: M2 配置字典

    Returns:
        (mask_table_goal_01, mask_table_residual_01, T2, J_best, stats)
        - mask_table_goal_01: 对齐后的目标 mask (224,224), 值为0/1
        - mask_table_residual_01: 需要扫除的区域 mask (224,224), 值为0/1
        - T2: 2x3 仿射变换矩阵
        - J_best: 最优代价值
        - stats: 优化统计信息
    """
    # 标准化目标图像
    goal_01 = m2_normalize_goal_to01(goal_any)

    # 平滑积木堆 mask（用于优化和 residual 计算）
    pile_smoothed = m2_prepare_pile_mask(B_01, cfg_m2)

    # 选择优化器
    optimizer_type = cfg_m2.get("optimizer", "differential_evolution")

    if optimizer_type == "grid":
        best_params, best_cost = _grid_search_optimize(pile_smoothed, goal_01, cfg_m2)
    elif optimizer_type == "scipy":
        best_params, best_cost = _scipy_optimize(pile_smoothed, goal_01, cfg_m2)
    elif optimizer_type == "differential_evolution":
        best_params, best_cost = _differential_evolution_optimize(pile_smoothed, goal_01, cfg_m2)
    elif optimizer_type == "hybrid":
        best_params, best_cost = _hybrid_optimize(pile_smoothed, goal_01, cfg_m2)
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")

    # 变换目标（与 RPP 同步的实现）
    G_01 = m2_warp_goal(goal_01, best_params)

    # 计算 residual（需要扫除的区域 = 积木堆中但不在目标中的部分）
    residual_01 = (pile_smoothed.astype(bool) & (~G_01.astype(bool))).astype(np.uint8)

    # 使用最终解重新计算代价分解（确保包含正则项）
    cost_fn = CostFunction(cfg_m2)
    cost_fn.set_pile_mask(pile_smoothed)
    breakdown = cost_fn.compute_cost_breakdown(G_01, best_params)
    best_cost = float(breakdown["total"])

    # 统计信息
    stats = {
        "optimizer": optimizer_type,
        "tx": float(best_params.tx),
        "ty": float(best_params.ty),
        "theta_rad": float(best_params.theta),
        "theta_deg": float(np.degrees(best_params.theta)),
        "scale": float(best_params.scale),
        "J_raw": best_cost,
        "C_fill": float(breakdown["fill"]),
        "C_remove": float(breakdown["remove"]),
        "C_edge": float(breakdown["edge"]),
        "C_sweep": float(breakdown["sweep"]),
        "C_reg": float(breakdown["reg"]),
    }

    # 构建 T2 矩阵（向后兼容对外接口需要矩阵的场景）
    T2 = m2_params_to_T2(
        best_params.tx, best_params.ty,
        best_params.theta, best_params.scale
    )

    return G_01, residual_01, T2, best_cost, stats
