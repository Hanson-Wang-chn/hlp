"""
视角变换标定工具

通过点击图像上的 4 个角点生成 table_corners 配置，用于将主相机图像变换到 table 俯视视角。

使用方法:
    python calibrate.py [--port 7860] [--share]

标定步骤:
    1. 上传一张主相机拍摄的图像（或使用默认的 data/E.png）
    2. 按顺序点击 table 区域的 4 个角点: 左上 → 右上 → 右下 → 左下
    3. 复制生成的 table_corners 配置到 config.yaml 中
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import gradio as gr
import numpy as np


class CalibrationState:
    """标定状态管理"""

    def __init__(self):
        self.points: List[Tuple[int, int]] = []
        self.image: Optional[np.ndarray] = None
        self.T1: Optional[np.ndarray] = None

    def reset(self):
        self.points = []
        self.T1 = None

    def add_point(self, x: int, y: int) -> int:
        """添加一个点，返回当前点数"""
        if len(self.points) < 4:
            self.points.append((x, y))
        return len(self.points)

    def compute_T1(self, output_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        根据 4 个角点计算 T1 单应变换矩阵。

        Args:
            output_size: 输出图像尺寸 (width, height)

        Returns:
            3x3 单应变换矩阵
        """
        if len(self.points) != 4:
            raise ValueError("需要 4 个点才能计算 T1 矩阵")

        src_points = np.array(self.points, dtype=np.float32)

        width, height = output_size
        dst_points = np.array([
            [0, 0],              # 左上
            [width - 1, 0],      # 右上
            [width - 1, height - 1],  # 右下
            [0, height - 1]      # 左下
        ], dtype=np.float32)

        self.T1 = cv2.getPerspectiveTransform(src_points, dst_points)
        return self.T1

    def format_corners_yaml(self) -> str:
        """将 4 个角点坐标格式化为 YAML 格式"""
        if len(self.points) != 4:
            return "# table_corners 尚未完成标定"

        labels = ["左上", "右上", "右下", "左下"]
        lines = ["table_corners:"]
        for i, (x, y) in enumerate(self.points):
            lines.append(f"  - [{x:.1f}, {y:.1f}]        # {labels[i]}")
        return "\n".join(lines)

    def format_T1_yaml(self) -> str:
        """将 T1 矩阵格式化为 YAML 格式（用于参考）"""
        if self.T1 is None:
            return "# T1 矩阵尚未生成"

        lines = ["# [参考] T1 透视变换矩阵（由 table_corners 自动计算）:", "T1:"]
        for row in self.T1:
            row_str = ", ".join(f"{v:.10f}" for v in row)
            lines.append(f"  - [{row_str}]")
        return "\n".join(lines)


state = CalibrationState()

# 点的颜色（BGR → RGB for display）
POINT_COLORS = [
    (255, 0, 0),    # 红色 - 左上
    (0, 255, 0),    # 绿色 - 右上
    (0, 0, 255),    # 蓝色 - 右下
    (255, 255, 0),  # 黄色 - 左下
]
POINT_LABELS = ["左上", "右上", "右下", "左下"]


def load_image(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """加载图像"""
    if image is None:
        return None, "请上传一张图像"

    state.image = image.copy()
    state.reset()

    return image, "图像已加载。请按顺序点击 4 个角点: 左上(红) → 右上(绿) → 右下(蓝) → 左下(黄)"


def handle_click(image: np.ndarray, evt: gr.SelectData) -> Tuple[np.ndarray, str, str, str, Optional[np.ndarray]]:
    """处理图像点击事件"""
    if state.image is None:
        return image, "请先上传图像", "", "", None

    x, y = evt.index[0], evt.index[1]
    num_points = state.add_point(x, y)

    # 在图像上绘制点
    display_image = state.image.copy()
    for i, (px, py) in enumerate(state.points):
        color = POINT_COLORS[i]
        cv2.circle(display_image, (px, py), 8, color, -1)
        cv2.circle(display_image, (px, py), 10, color, 2)
        cv2.putText(display_image, str(i + 1), (px + 12, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 绘制连线
    if len(state.points) > 1:
        for i in range(len(state.points)):
            p1 = state.points[i]
            p2 = state.points[(i + 1) % len(state.points)] if i < len(state.points) - 1 or len(state.points) == 4 else None
            if p2:
                cv2.line(display_image, p1, p2, (255, 255, 255), 2)

    if num_points < 4:
        next_label = POINT_LABELS[num_points]
        status = f"已选择 {num_points}/4 个点。下一个: {next_label}"
        return display_image, status, "", "", None

    # 4 个点已选完，计算 T1
    state.compute_T1()
    corners_yaml = state.format_corners_yaml()
    t1_yaml = state.format_T1_yaml()

    # 生成预览图
    preview = cv2.warpPerspective(state.image, state.T1, (224, 224))

    status = "标定完成！请复制下方的 table_corners 配置到 config.yaml 中"
    return display_image, status, corners_yaml, t1_yaml, preview


def reset_calibration(image: np.ndarray) -> Tuple[np.ndarray, str, str, str, None]:
    """重置标定"""
    state.reset()
    display_image = state.image.copy() if state.image is not None else image
    return display_image, "已重置。请重新点击 4 个角点", "", "", None


def load_default_image() -> Tuple[np.ndarray, str]:
    """加载默认图像"""
    default_path = Path(__file__).parent / "data" / "E.png"
    if default_path.exists():
        img = cv2.imread(str(default_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return load_image(img)
    return None, "默认图像不存在，请手动上传"


# 构建 Gradio 界面
with gr.Blocks(title="视角变换标定工具") as demo:
    gr.Markdown("# 视角变换标定工具")
    gr.Markdown("""
    通过点击图像上的 4 个角点生成 `table_corners` 配置。

    **步骤:**
    1. 上传主相机拍摄的图像
    2. 按顺序点击 table 区域的 4 个角点: **左上(红) → 右上(绿) → 右下(蓝) → 左下(黄)**
    3. 复制生成的 `table_corners` 配置到 `config.yaml`
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="主相机图像（点击选择角点）",
                type="numpy",
                interactive=True
            )
            with gr.Row():
                load_btn = gr.Button("加载默认图像 (data/E.png)")
                reset_btn = gr.Button("重置标定")

        with gr.Column():
            preview_output = gr.Image(
                label="T1 变换预览 (224×224)",
                type="numpy",
                interactive=False
            )

    status_text = gr.Textbox(label="状态", interactive=False)

    corners_output = gr.Textbox(
        label="table_corners (复制到 config.yaml)",
        lines=6,
        interactive=False
    )

    t1_output = gr.Textbox(
        label="T1 矩阵 (参考，通常无需复制)",
        lines=5,
        interactive=False
    )

    # 事件绑定
    image_input.upload(
        fn=load_image,
        inputs=[image_input],
        outputs=[image_input, status_text]
    )

    image_input.select(
        fn=handle_click,
        inputs=[image_input],
        outputs=[image_input, status_text, corners_output, t1_output, preview_output]
    )

    load_btn.click(
        fn=load_default_image,
        outputs=[image_input, status_text]
    )

    reset_btn.click(
        fn=reset_calibration,
        inputs=[image_input],
        outputs=[image_input, status_text, corners_output, t1_output, preview_output]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视角变换标定工具")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="启用公共链接")
    parser.add_argument("--server", type=str, default="127.0.0.1", help="服务器地址")

    args = parser.parse_args()

    demo.launch(server_name=args.server, server_port=args.port, share=args.share)
