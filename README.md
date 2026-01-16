# HLP - High Level Policy for Sweep to Shapes

**版本: 1.2.0**

## 项目简介

本项目实现了用于 "Sweep to Shapes" 任务的高层策略（High Level Policy, HLP）。该任务面向真机双臂毛刷协同场景：桌面（30cm×30cm 正方形框）内有大量最小号乐高积木（红色），目标是通过多次直线 sweep 将积木"扫出"指定的目标图案（如字母/数字/简单图案），使方框内最终仅保留目标形状区域的积木。

HLP 的职责是在每一步（每次 sweep 动作之前）根据**当前主相机图像**与**二值目标图像（table 视角）**，生成下一次最应该执行的 **sweep 区域 mask** 及其方向，并输出一个用于 LLP/VLA 的可视化叠加图（224×224）。

## 系统要求

- Python >= 3.10
- 仅 CPU 推理（不依赖 GPU）

## 安装

```bash
# 克隆项目
cd hlp_sweep_to_shapes

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```bash
# 使用默认配置运行
python scripts/demo_infer.py \
    --main_img data/snapshot.png \
    --goal_img data/E.png \
    --output outputs/demo

# 使用自定义配置
python scripts/demo_infer.py \
    --main_img data/snapshot.png \
    --goal_img data/E.png \
    --config config.yaml \
    --output outputs/demo

# 指定角点坐标和步骤ID（推荐）
python scripts/demo_infer.py \
    --main_img data/snapshot.png \
    --goal_img data/E.png \
    --corners assets/table_corners.npy \
    --step_id 0 \
    --output outputs/demo
```

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--main_img` | `-m` | 主相机图像路径（BGR格式） | 必需 |
| `--goal_img` | `-g` | 目标二值图像路径（1=保留区域） | 必需 |
| `--corners` | `-r` | **[推荐]** table_corners角点坐标文件（.npy, shape=(4,2)） | 从配置读取 |
| `--t1` | `-t` | [已弃用] T1单应矩阵文件路径（.npy） | 从配置计算 |
| `--config` | `-c` | 配置文件路径（YAML） | 默认配置 |
| `--output` | `-o` | 输出目录路径 | `outputs` |
| `--step_id` | `-s` | 步骤ID | `0` |
| `--index` | `-i` | 选择执行的mask索引 | `0` |

### Python API 使用

```python
import numpy as np
import cv2
from hlp import HighLevelPolicy, validate_cfg, compute_perspective_transforms

# 准备配置
cfg = validate_cfg({})  # 使用默认配置

# 定义4个角点坐标（左上、右上、右下、左下）
# 这4个点表示224x224主相机图像上的table区域
table_corners = np.array([
    [20.0, 30.0],     # 左上
    [200.0, 25.0],    # 右上
    [210.0, 195.0],   # 右下
    [15.0, 200.0],    # 左下
], dtype=np.float32)

# 创建HLP实例（推荐方式：使用table_corners）
hlp = HighLevelPolicy(cfg, table_corners=table_corners)

# 或者从配置文件读取table_corners（配置中已有table_corners字段）
# hlp = HighLevelPolicy(cfg)

# 加载图像
main_img = cv2.imread("data/snapshot.png")
main_img = cv2.resize(main_img, (224, 224))

goal_img = cv2.imread("data/E.png", cv2.IMREAD_GRAYSCALE)
goal_img = cv2.resize(goal_img, (224, 224))

# 执行推理
result = hlp.infer(
    img_rgb_main_cur_224=main_img,
    img_bin_table_goal_224=goal_img,
    index_mask_exec=0,
    step_id=0,
    save_dir="outputs"
)

# 获取结果
overlay_img = result["img_rgb_main_exec_224"]  # 叠加可视化图
exec_mask = result["mask_main_exec_224"]       # 执行区域mask
meta = result["meta"]                           # 元信息

print(f"状态: {meta['status']}")
if meta['status'] == 'ok':
    print(f"扫除方向: {meta['chosen']['direction']}")
    print(f"区域面积: {meta['chosen']['area']}")
```

## 视角变换标定

### 什么是 table_corners

`table_corners` 是 4 个角点坐标，表示 224×224 主相机图像上 table 区域的 4 个顶点位置。这 4 个点按照 **左上、右上、右下、左下** 的顺序排列，围成的四边形区域将被透视变换到 224×224 的 table 俯视视角图像。

这是 HLP 系统 v1.2 引入的新配置方式，比直接配置 T1 矩阵更加直观易用。系统会根据这 4 个角点自动计算透视变换矩阵 T1 及其逆矩阵 T1_inv。

### 使用 calibrate.py 标定工具

项目提供了基于 Gradio 的可视化标定工具 `calibrate.py`，可以通过点击图像上的 4 个点来标定 table 区域。

#### 功能说明

`calibrate.py` 是一个精简的 Gradio 标定页面，主要功能包括：

- **图像加载**：支持上传主相机图像或加载默认图像 (`data/E.png`)
- **交互式标定**：通过鼠标点击图像上的 4 个角点定义变换区域
- **实时预览**：完成标定后自动生成 224×224 的变换预览图
- **YAML 输出**：自动生成可直接复制到 `config.yaml` 的 `table_corners` 格式
- **重置功能**：支持重新标定

#### 运行方式

```bash
# 运行标定工具（默认端口 7860）
python calibrate.py

# 指定端口
python calibrate.py --port 8080

# 启用公共链接（用于远程访问）
python calibrate.py --share

# 指定服务器地址
python calibrate.py --server 0.0.0.0 --port 7860
```

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--port` | 端口号 | 7860 |
| `--server` | 服务器地址 | 127.0.0.1 |
| `--share` | 启用 Gradio 公共链接 | False |

### 标定步骤

1. **启动工具**：运行 `python calibrate.py`，在浏览器中打开显示的 URL（默认 http://localhost:7860）

2. **上传图像**：上传一张主相机拍摄的图像，或点击"加载默认图像"按钮加载 `data/E.png`

3. **点击 4 个角点**：按照以下顺序在图像上点击 table 区域的 4 个角点：
   - 点 1（红色）：左上角
   - 点 2（绿色）：右上角
   - 点 3（蓝色）：右下角
   - 点 4（黄色）：左下角

4. **预览结果**：完成 4 点标定后，右侧会显示透视变换后的 224×224 预览图

5. **复制角点坐标**：将下方生成的 YAML 格式 `table_corners` 复制到 `config.yaml` 中

6. **重新标定**：如果标定不满意，点击"重置标定"按钮重新开始

### 在 config.yaml 中配置 table_corners

```yaml
# table_corners: 4个角点坐标
# 表示224x224的img_rgb_main_cur上的4个点
# 按左上、右上、右下、左下顺序排列
table_corners:
  - [20.0, 30.0]      # 左上
  - [200.0, 25.0]     # 右上
  - [210.0, 195.0]    # 右下
  - [15.0, 200.0]     # 左下
```

### 读取优先级

运行 demo_infer.py 时，变换参数的读取优先级为：
1. 如果指定了 `--corners` 参数（.npy 文件，shape=(4,2)），使用该文件
2. 如果指定了 `--t1` 参数（.npy 文件，已弃用），使用该文件
3. 否则从 `config.yaml` 的 `table_corners` 字段读取并计算

```bash
# 使用 config.yaml 中的 table_corners（推荐）
python scripts/demo_infer.py \
    --main_img data/snapshot.png \
    --goal_img data/E.png \
    --config config.yaml

# 使用外部 .npy 文件指定角点坐标
python scripts/demo_infer.py \
    --main_img data/snapshot.png \
    --goal_img data/E.png \
    --config config.yaml \
    --corners assets/my_corners.npy
```

## 配置说明

配置文件 `config.yaml` 包含以下主要部分：

### table_corners: 视角变换角点

```yaml
# 4个角点坐标，定义主相机图像上的table区域
# 顺序：左上、右上、右下、左下
# 默认为整个图像（无变换）
table_corners:
  - [0.0, 0.0]        # 左上
  - [223.0, 0.0]      # 右上
  - [223.0, 223.0]    # 右下
  - [0.0, 223.0]      # 左下
```

### 调试配置 (debug)

```yaml
debug:
  save_intermediate: true    # 是否保存中间产物到磁盘
  return_intermediate: false # 是否返回中间数组（注意内存）
```

### M1: 红色积木分割 (m1)

```yaml
m1:
  # HSV红色双区间阈值
  hsv_red_low1: [0, 120, 60]
  hsv_red_high1: [10, 255, 255]
  hsv_red_low2: [170, 120, 60]
  hsv_red_high2: [180, 255, 255]

  # 形态学参数
  morph_kernel_size: 3
  morph_open_iter: 1
  morph_close_iter: 1
```

### M2: 目标对齐优化 (m2)

```yaml
m2:
  seed: 0                    # 随机种子
  maxiter: 30                # 最大迭代次数
  popsize: 10                # 种群大小
  lambda1: 0.6               # C_fill权重
  lambda2: 0.4               # C_remove权重
  bounds:
    tx: 18.0                 # 平移范围
    ty: 18.0
    theta_deg: 8.0           # 旋转范围（度）
    scale_low: 0.92          # 缩放范围
    scale_high: 1.08
```

### M3: Flow Field (m3)

```yaml
m3:
  bfs_connectivity: 4        # BFS连通性
  obstacle_fill_dist_margin: 5.0
  sobel_ksize: 3
```

### M4: 区域切分 (m4)

```yaml
m4:
  lambda_h: 1.2              # 水平偏置系数（>1）
  mag_min: 0.001             # 最小梯度阈值
  A_min: 60                  # 最小面积
  H_min: 18                  # 最小高度
  H_max: 42                  # 最大高度
```

### M5: 可视化 (m5)

```yaml
m5:
  overlay_alpha: 0.35        # 叠加透明度
  color_up: [255, 0, 0]      # 向上颜色（BGR）
  color_down: [0, 0, 255]    # 向下颜色
  color_left: [0, 255, 255]  # 向左颜色
  color_right: [0, 255, 0]   # 向右颜色
```

## 项目结构

```
hlp_sweep_to_shapes/
├── README.md                    # 本文档
├── requirements.txt             # 依赖列表
├── config.yaml                  # 默认配置文件
├── calibrate.py                 # 视角变换标定工具（Gradio界面）
│
├── hlp/                         # 核心模块
│   ├── __init__.py              # 包入口
│   ├── high_level_policy.py     # 顶层HLP类
│   ├── config.py                # 配置管理
│   ├── types.py                 # 类型定义
│   ├── logger.py                # 日志系统
│   ├── io_utils.py              # I/O工具
│   ├── cache.py                 # 缓存管理
│   ├── m1_rectify_segment.py    # M1: 图像矫正+分割
│   ├── m2_align_goal.py         # M2: 目标对齐优化
│   ├── m3_flow_field.py         # M3: Flow Field计算
│   ├── m4_chunker.py            # M4: 区域切分
│   └── m5_select_overlay.py     # M5: 选择+可视化
│
├── scripts/                     # 脚本
│   └── demo_infer.py            # 演示脚本
│
├── tests/                       # 测试
│   └── ...
│
├── assets/                      # 示例资源
│   └── ...
│
└── outputs/                     # 输出目录
    └── ...
```

## 处理流程

HLP 每次推理按以下流程执行：

1. **M1 (矫正+分割)**：主相机图像 → table平面矫正 → HSV红色积木分割 → `img_bin_table_cur`

2. **M2 (目标对齐)**：优化相似变换T2 → 得到 `mask_table_goal` 和 `mask_table_residual`

3. **M3 (Flow Field)**：以goal为障碍计算测地线距离场 → Sobel梯度 → `map_grad`

4. **M4 (区域切分)**：方向离散化 + 连通域提取 + 高度切分 → `mask_exec_list`

5. **M5 (选择+叠加)**：优先级排序 → 选择执行mask → 反变换回主相机 → 叠加可视化

## 输出说明

### 返回结果 (HLPResult)

```python
{
    "img_rgb_main_exec_224": np.ndarray,  # (224,224,3) 叠加可视化图
    "mask_main_exec_224": np.ndarray,     # (224,224) 执行区域mask
    "meta": {
        "status": "ok" | "empty" | "fail",
        "step_id": int,
        "cache": {"m2_hit": bool, "m3_hit": bool},
        "counts": {
            "cur_pixels": int,
            "goal_pixels": int,
            "residual_pixels": int,
            "chunks": int
        },
        "chosen": {
            "direction": "Up" | "Down" | "Left" | "Right",
            "G_min": float,
            "G_max": float,
            "area": int,
            "bbox": (x, y, w, h),
            "priority_score": float,
            "chosen_idx_sorted": int
        }
    },
    "debug": dict  # 调试信息
}
```

### 保存的中间产物

当 `save_dir` 不为空时，会保存以下文件：

```
{save_dir}/
├── m1/
│   ├── img_rgb_table_cur.png    # 矫正后的table视角图像
│   └── img_bin_table_cur.png    # 红色积木分割结果
├── m2/
│   ├── mask_table_goal.png      # 对齐后的目标mask
│   └── mask_table_residual.png  # 需要扫除的区域
├── m3/
│   ├── dist_norm.png            # 归一化距离场
│   └── mag.png                  # 梯度模长
├── m4_masks/
│   └── mask_*.png               # 各个可执行区域mask
├── m5_overlay/
│   └── overlay_*.png            # 叠加可视化图
├── result_overlay.png           # 最终结果叠加图
└── result_mask.png              # 最终执行mask
```

## 注意事项

1. **图像格式**：所有图像处理使用OpenCV，采用BGR格式
2. **分辨率**：内部固定处理分辨率为 224×224
3. **二值图像**：目标图像中 1（或255）表示需要保留积木的区域
4. **table_corners**：需要提供主相机图像上 table 区域的 4 个角点坐标，可通过 `calibrate.py` 标定工具生成
5. **缓存机制**：M2和M3的结果会被缓存，相同goal时可复用

## 版本历史

### v1.2.0
- **重大更新**：使用 `table_corners`（4个角点坐标）替代直接配置 T1 矩阵
- 新增 `get_table_corners_from_cfg()` 和 `compute_perspective_transforms()` 函数
- 新增 `load_cfg_and_transforms()` 便捷函数，一次性返回配置和变换矩阵
- `HighLevelPolicy` 类支持通过 `table_corners` 参数初始化
- demo_infer.py 新增 `--corners` 命令行参数
- 保留旧接口 `get_T1_from_cfg()` 和 `load_cfg_and_T1()` 以保持向后兼容（标记为已弃用）
- 更新 README 文档

### v1.1.1
- 新增 `calibrate.py` 标定工具，基于 Gradio 实现可视化标定
- 支持交互式 4 点标定、实时预览、YAML 格式输出
- 更新 README 文档

### v1.1
- T1矩阵现在从 `config.yaml` 读取，无需单独的 .npy 文件
- 新增 `get_T1_from_cfg()` 和 `load_cfg_and_T1()` 辅助函数
- 更新 README 文档

### v1.0
- 初始版本
- 完整的 M1-M5 模块实现
- 支持缓存机制和结构化日志

## License

MIT License
