# hlp_sweep_to_shapes 配置参数有效性检查（config.yaml / config_m2.yaml）

## 结论摘要

1) `hlp_sweep_to_shapes/config_m2.yaml` **在当前仓库的标准运行路径中不会生效**：`scripts/demo_infer.py` 只支持加载一个 `--config` 文件，而 `hlp/config.py::validate_cfg()` 期望 M2 配置位于 `cfg["m2"]`（嵌套结构）。`config_m2.yaml` 的键全部在顶层，因此即便作为 `--config` 传入，也只会被合并为“未知顶层键”，不会影响 `self.cfg["m2"]`，等价于 **所有 M2 参数仍使用默认值**。

2) `hlp_sweep_to_shapes/config.yaml` 中存在 **至少 2 个“看起来可配但实际不生效”的参数**：
   - `m1.morph_kernel_shape`：代码始终使用椭圆核（ellipse），不会读取该字段。
   - `m3.bfs_connectivity`：代码始终使用 4 邻域 BFS，不会读取该字段。

下文给出逐项映射与证据位置（文件 + 行号）。

---

## 配置加载与生效机制（决定“是否会从配置读取”）

### 1) 入口脚本只支持一个 `--config`
- `hlp_sweep_to_shapes/scripts/demo_infer.py#L175`：只要 `--config` 存在，就调用 `load_cfg_from_yaml(args.config)` 加载**单个 YAML**，没有第二份配置（例如 `config_m2.yaml`）的合并逻辑。

### 2) 配置结构期望：`DEFAULT_CFG` 是“模块嵌套字典”
- `hlp_sweep_to_shapes/hlp/config.py#L12`：默认配置结构为：
  - 顶层：`table_corners`, `debug`, `m1`, `m2`, `m3`, `m4`, `m5`, `io`, `log`
  - 其中 M2 必须位于 `m2:` 子树（例如 `m2.lambda_fill`）。

### 3) YAML 合并策略：未知键不会报错，但也不会被业务代码读取
- `hlp_sweep_to_shapes/hlp/config.py#L152`：`merge_dict()` 会把用户 YAML 中的键递归合并到默认配置上：
  - 若键不在默认配置里（比如 `optimizer` 顶层），也会直接挂到 `result` 顶层；
  - **但业务代码普遍只读取 `cfg["m1"] / cfg["m2"] / ...` 等模块子树**，因此这类“未知顶层键”会静默变成“无效配置项”。

### 4) M2 实际读取位置：`self.cfg["m2"]`
- `hlp_sweep_to_shapes/hlp/high_level_policy.py#L191`：M2 调用 `m2_run(B_01, goal_01, self.cfg["m2"])`，即 M2 只会读 `cfg["m2"]` 里的字段。

---

## config.yaml：逐项检查（哪些生效 / 哪些无效）

文件：`hlp_sweep_to_shapes/config.yaml`

### A. 生效项（能在代码中找到对应读取）

- `table_corners`（`config.yaml#L17`）：通过 `get_table_corners_from_cfg()` 读出并用于透视变换。
- `debug.save_intermediate`（`config.yaml#L25`）：影响是否创建/写入输出目录（`hlp/high_level_policy.py#L148`）。
- `debug.return_intermediate`（`config.yaml#L26`）：影响 `result["debug"]` 是否返回中间数组（`hlp/high_level_policy.py#L340` 附近）。
- `m1.*` 中除 `morph_kernel_shape` 外：HSV 阈值、核大小、开闭运算次数都会被读取（`hlp/m1_rectify_segment.py#L55`、`hlp/m1_rectify_segment.py#L66`、`hlp/m1_rectify_segment.py#L70` 等）。
- `m3.obstacle_fill_dist_margin`（`config.yaml#L45`）：用于距离场填充值（`hlp/m3_flow_field.py#L107`）。
- `m3.sobel_ksize`（`config.yaml#L46`）：用于 Sobel 核大小（`hlp/m3_flow_field.py#L150`）。
- `m4.*`：全部在 `hlp/m4_chunker.py` 中读取（方向离散、连通域过滤、切分等）。
- `m5.*`：叠加透明度与颜色在 `hlp/m5_select_overlay.py` 中读取（`overlay_alpha`、`color_*`）。
- `io.*`：保存开关在 `hlp/high_level_policy.py`、`hlp/m4_chunker.py`、`hlp/m5_select_overlay.py` 中读取。
- `log.*`：日志级别、是否结构化、是否输出到控制台在 `hlp/logger.py` / `hlp/high_level_policy.py` 中读取。

### B. 无效/不生效项（配置存在，但代码不读取或被硬编码）

1) `m1.morph_kernel_shape`（`hlp_sweep_to_shapes/config.yaml#L37`）
- 现状：代码始终使用 `cv2.MORPH_ELLIPSE` 创建核，未读取 `cfg_m1["morph_kernel_shape"]`。
- 证据：`hlp_sweep_to_shapes/hlp/m1_rectify_segment.py#L66` 直接 `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...)`。
- 影响：即便把 `config.yaml` 里该字段改成 `"rect"`/`"cross"` 等，运行结果也不会变化。

2) `m3.bfs_connectivity`（`hlp_sweep_to_shapes/config.yaml#L44`）
- 现状：BFS 固定使用 4 邻域（上/下/左/右），未读取 `cfg_m3["bfs_connectivity"]`。
- 证据：`hlp_sweep_to_shapes/hlp/m3_flow_field.py#L57` 开始的 BFS 循环只实现了 4 个方向的扩展，函数体内没有 `bfs_connectivity` 相关读取。
- 影响：该字段目前是“形式上的参数”，更像注释说明而非真实可配项。

---

## config_m2.yaml：逐项检查（哪些生效 / 哪些无效）

文件：`hlp_sweep_to_shapes/config_m2.yaml`

### A. 标准路径下的整体结论：全部不生效

在“使用 `HighLevelPolicy` + `load_cfg_from_yaml()`”的默认使用方式下（`scripts/demo_infer.py`）：
- `config_m2.yaml` 的所有键（如 `optimizer`, `lambda_fill`, `theta_range`, `grid_resolution` 等）都位于 YAML 顶层（`config_m2.yaml#L13` 起）。
- 但运行时只会读取 `cfg["m2"]` 子树（`hlp/high_level_policy.py#L191`）。
- `validate_cfg()` 会把这些顶层键静默合并进总配置（`hlp/config.py#L152`），**不会报错**，但也**没有任何代码路径会读取它们**。

因此：如果你希望“`config_m2.yaml` 中的参数在代码运行时使用”，当前实现并不能满足。

### B. 额外的“部分生效/被覆盖”风险（即便未来接入也需要注意）

以下项属于“即便把 M2 配置正确放到 `cfg["m2"]`，也可能与你预期不同”的点：

1) `grid_resolution` 在 `optimizer=hybrid` 时不会完全按配置走
- `config_m2.yaml#L90` 设为 `20`，但混合优化器的“粗网格阶段”强制设置为 `10`。
- 证据：`hlp_sweep_to_shapes/hlp/m2_align_goal.py#L579-L580`：`coarse_cfg["grid_resolution"] = 10`。
- 影响：`grid_resolution` 对 `hybrid` 的粗阶段无效（只影响非 hybrid 的 grid optimizer / 或 hybrid 的后续部分间接行为）。

2) 默认值与 config_m2.yaml 存在明显差异，若 config_m2 未生效会造成行为偏差
- 例如 `theta_range`：`config_m2.yaml#L51-L53` 设为 `[0, 0]`（禁用旋转），但默认 `m2.theta_range` 是 `[-π/4, π/4]`（`hlp/config.py#L76`）。
- 例如 `grid_resolution`：`config_m2.yaml#L90` 为 `20`，默认 `m2.grid_resolution` 为 `10`（`hlp/config.py#L68`）。
- 影响：如果你以为 `config_m2.yaml` 已经控制了这些参数，实际上当前运行会走默认值，导致 M2 优化搜索空间、收敛行为与预期不一致。

---

## 建议的下一步（仅建议，不修改代码）

如果你希望“两个 YAML 都作为运行时配置来源”并且 `config_m2.yaml` 的参数确实生效，至少需要解决一个结构/加载问题（任选其一）：

1) 让 `config_m2.yaml` 变成嵌套结构（顶层增加 `m2:`，把所有键放到 `m2` 下），并在主配置里合并；或
2) 在代码入口处支持 `--config` + `--config_m2` 两份文件并做合并（把 M2 部分写入 `cfg["m2"]`）；或
3) 保留 `config_m2.yaml` 作为“仅供直接调用 `m2_run(..., cfg_m2)`”的配置文件，但同时需要在 README/脚本中明确说明它不会影响 `HighLevelPolicy`。

