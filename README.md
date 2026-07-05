# Cascade 串联商用守护方案

`cascade/` 是一个可以单独拷走运行的串联方案，用于基于 PPG/ACC 的手表佩戴活体检测。它的核心原则是：**完全保留现有商用特征、商用模型参数和商用推理逻辑，只在商用模型后面增加一个很小的风险复核层**。

这个方案适合优先降低“误判佩戴”的商业风险。它不会把商用模型判为未佩戴的样本改成佩戴，只会在商用模型已经判为佩戴的样本中，额外识别是否存在高风险非佩戴样本。

## 1. 项目定位

串联方案的推理路径是：

```text
原始 PPG/ACC 数据
    -> 冻结商用模型 M_c
    -> 商用输出 commercial_pred / commercial_score
    -> 只对商用阳性候选做二次特征提取和风险判断
    -> 输出 cascade_pred / guard_action / veto_risk
```

设计目标：

- 保持商用模型 `s01_model.py` 不变。
- 保持商用 8 个特征不变。
- 保持商用 AdaBoost 树结构、阈值、后处理延迟参数不变。
- 新增模型只作为后置风险守护层。
- 默认 `shadow` 模式只记录风险，不改变最终输出。
- 重点分析 hard negative，也就是“真实非佩戴但商用模型判为佩戴”的样本。

## 2. 独立运行边界

这个文件夹是独立项目。把整个 `cascade/` 文件夹拷贝到其他位置后，只要有数据集和 Python 环境，就可以直接运行。

它不依赖父目录 `new_codex_1` 中的脚本，不从 `parallel/` 导入代码，也不要求两个项目同时存在。

运行时仍需要外部输入：

- H5 数据集目录，通过 `--dataset_dir` 指定。
- Python 包：`numpy`、`pandas`、`scikit-learn`、`xgboost`、`h5py`、`joblib`、`matplotlib`、`scipy`。
- 可选 Graphviz `dot`，用于把 XGBoost 树导出为 PNG 图片。如果没有 Graphviz，仍会保留 JSON/DOT/TXT 树结构文件。

默认输出在当前文件夹下：

```text
cascade/artifacts/
```

其中：

```text
artifacts/splits.json
artifacts/cascade/*
```

## 3. 商用模型冻结约束

商用模型位于：

```text
s01_model.py
```

它包含：

- 商用 8 个特征名：`feature_names`。
- 商用 AdaBoost 参数：`tree_num`、`tree_node`、`detect_tree_threshold`。
- 商用树数组：`TREE_INDEX`、`TREE_VALUE`。
- 商用逻辑判断阈值：`good_corr_threshold`、`good_ac_threshold` 等。
- 商用状态延迟：`live_flag_delay`、`un_live_flag_delay`。
- 商用窗口：`commercial_win_sec=5`、`commercial_stride_sec=1`。
- Stage1 门控参数。

运行 `s05_run_commercial.py` 时会生成：

```text
artifacts/cascade/commercial_model_manifest.json
```

这个 manifest 用于验收商用模型是否被冻结，关键字段包括：

```text
model_name = frozen_commercial_adaboost
feature_names
tree_num
tree_node
detect_tree_threshold
stage1_primitive_sec
stage1_decision_sec
stage1_fs
stage1_gate_k
tree_index_sha256
tree_value_sha256
frozen = true
```

验收时应确认：

- `frozen` 是 `true`。
- `tree_index_sha256` 和 `tree_value_sha256` 没有变化。
- `feature_names` 没有变化。
- 新增守护逻辑不修改 `s01_model.py` 的树数组和阈值。

## 4. 数据格式

数据目录中应包含 `.h5` 文件。

支持两类 H5 样本结构：

1. 普通样本 group：

```text
sample_xxx/
    ppg
    target
    acc          可选
```

2. grouped-window 样本：

```text
sample_xxx/
    xxx_w20_1/
        ppg
        acc      可选
    xxx_w20_2/
        ppg
        acc      可选
```

支持的 PPG shape：

```text
(40, T)
(N_windows, 40, T_window)
```

`target` 约定：

```text
0 = 非佩戴 / 攻击 / 负样本
1 = 正常佩戴 / 正样本
```

## 5. split 方法

split 逻辑在：

```text
s04_data.py
```

默认参数：

```text
valid_size = 0.15
test_size = 0.15
random_state = 42
```

切分方式：

- 扫描数据集中的所有 H5 文件。
- 按样本读取 `target`。
- 使用 stratified split，保持 train/valid/test 中正负样本比例尽量一致。
- 第一次运行时写入 `artifacts/splits.json`。
- 后续运行默认复用已有 `splits.json`。
- 如果需要重新切分，使用 `--force_split`。

命令：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --force_split
```

## 6. 代码结构

### `s01_model.py`

冻结商用模型模块。

主要内容：

- `FEATURE_NAMES`：商用 8 个特征。
- `TREE_INDEX`、`TREE_VALUE`：商用 AdaBoost 树。
- `OldLivenessModel`：商用模型推理类。
- `CommercialStage1Gate`：Stage1 门控。
- `extract_8_commercial_features()`：商用 8 特征提取。
- `commercial_model_manifest()`：冻结模型验收 manifest。

这份文件是商用基线，不应随新增守护方案修改。

### `s02_features.py`

PPG/ACC 特征提取模块。

主要内容：

- PPG 预处理。
- 绿光、环境光、IR、ACC 特征。
- Stage1 阈值配置。
- 部署友好特征白名单。
- 特征池生成工具。

当前 Stage1 默认阈值：

```text
DEFAULT_STAGE1_DC_THRESHOLD = 0.3e6
DEFAULT_STAGE1_AC_DC_THRESHOLD = 1.0
```

### `s03_selection.py`

高级特征分析和可视化工具。

主要能力：

- 特征稳定性分析。
- VIF / 相关性分析。
- PCA、t-SNE、UMAP 嵌入图。
- 特征分布图。
- 特征排序报告。

当前图片策略：只输出高清 PNG。

### `s04_data.py`

数据扫描和 split 工具。

主要能力：

- 扫描 H5 文件。
- 兼容普通样本和 grouped-window 样本。
- 生成 train/valid/test。
- 保存和读取 `splits.json`。

### `s05_run_commercial.py`

运行冻结商用模型。

输入：

```text
artifacts/splits.json
H5 数据文件
```

输出：

```text
artifacts/cascade/commercial_model_manifest.json
artifacts/cascade/commercial_results_train.csv
artifacts/cascade/commercial_results_valid.csv
artifacts/cascade/commercial_results_test.csv
```

每行结果包含：

- `sample_name`
- `target`
- `window_idx`
- `pred`
- `score`
- `stage2_enabled`
- `fallback`
- `is_error`

### `s06_extract_errors.py`

从商用阳性候选中提取守护模型训练样本。

串联方案不会对所有样本训练，而是聚焦于：

```text
commercial_pred == 1
stage2_enabled == true
fallback == false
```

训练标签：

```text
should_veto = 1 when target == 0 among commercial-positive candidates
```

也就是说，真实非佩戴但商用判为佩戴的样本，就是 hard negative。

输出：

```text
artifacts/cascade/error_features_train.csv
artifacts/cascade/error_features_valid.csv
artifacts/cascade/error_features_test.csv
artifacts/cascade/hard_negative_audit/*
```

### `s07_select_features.py`

为新增小模型选择特征。

特点：

- 只从数值特征中选。
- 自动排除标签、预测结果和泄漏字段。
- 输出 ranked feature 和人工选择模板。

输出：

```text
artifacts/cascade/selected_features.json
artifacts/cascade/feature_review/ranked_features.csv
artifacts/cascade/feature_review/ranked_features.json
artifacts/cascade/feature_review/ranked_features.md
artifacts/cascade/feature_review/manual_feature_selection_template.json
```

### `s08_train_corrector.py`

训练串联守护模型。

默认模型：

```text
XGBoost
n_estimators = 10
max_depth = 2
```

如果训练标签只有一个类别，会退化为 constant probability guard，避免训练崩溃。

输出：

```text
artifacts/cascade/corrector_model.json
artifacts/cascade/corrector_bundle.pkl
```

### `s09_evaluate.py`

评估商用基线和完整方案。

输出：

```text
artifacts/cascade/evaluation_report.json
artifacts/cascade/evaluation_samples.csv
artifacts/cascade/evaluation_comparison.csv
```

核心对比：

```text
commercial_pred  只依赖商用模型的输出
cascade_pred     当前完整串联方案输出
bypass_pred      回退模式输出，等于商用输出
```

### `s10_pipeline.py`

一键运行脚本。

它串联执行：

```text
自动生成或读取 splits.json
S05 运行商用模型
S06 提取商用阳性候选和 hard negative
S07 选择特征
S08 训练小 XGBoost / constant guard
S09 评估
S11 可解释性报告，可选
```

### `s11_explain.py`

解释性报告脚本。

输出内容：

- 商用模型 vs 完整方案指标对比。
- 样本流转 funnel。
- 错误样本分布图。
- XGBoost 树结构导出。
- 错误样本经过了哪些树节点。
- 商用过滤报告。

输出目录：

```text
artifacts/cascade/figures/*.png
artifacts/cascade/tree_export/*
artifacts/cascade/error_trace/*
artifacts/cascade/commercial_filter_report/*
```

## 7. 快速运行

进入项目目录：

```bash
cd D:\wearing_liveness\new\new_codex_1\cascade
```

只检查命令和路径，不实际运行：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --dry_run
```

完整运行 shadow 模式，并生成解释性报告：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --guard_mode shadow --explain
```

重新生成 split：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --force_split
```

指定输出目录，避免覆盖已有结果：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --splits_dir artifacts_run_001 --artifact_dir artifacts_run_001\cascade --force_split --guard_mode shadow --explain
```

## 8. Guard 模式

支持 4 种模式：

```text
bypass
shadow
soft_guard
hard_veto
```

### `bypass`

最终输出完全等于商用输出。

用途：

- 回退验证。
- 确认新增逻辑不会影响商用结果。

### `shadow`

默认模式。

最终输出仍等于商用输出，但记录守护模型风险：

```text
final_pred = commercial_pred
```

用途：

- 线上静默观察。
- 收集 disagreement。
- 分析 hard negative。

### `soft_guard`

最终分类仍不直接推翻商用输出，但当风险持续出现时，建议延长检测或进入更保守后处理。

用途：

- 比 hard veto 更温和。
- 适合先做体验风险较低的灰度。

### `hard_veto`

当风险满足持续条件时，可以把商用阳性改为阴性：

```text
commercial_pred = 1
risk_count >= min_veto_windows
risk_ratio >= min_veto_ratio
```

默认持续条件：

```text
min_veto_windows = 2
min_veto_ratio = 0.4
```

注意：`hard_veto` 不建议直接全量商用，应该只用于离线评估或严格灰度。

## 9. 手工特征选择

自动特征筛选后会生成模板：

```text
artifacts/cascade/feature_review/manual_feature_selection_template.json
```

人工确认后，另存为：

```text
artifacts/cascade/feature_review/manual_feature_selection.json
```

然后运行：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --manual_features artifacts/cascade/feature_review/manual_feature_selection.json
```

训练脚本会拒绝标签泄漏字段，例如：

```text
target
should_veto
commercial_pred
is_error
fallback
```

## 10. 主要输出解释

### `commercial_model_manifest.json`

商用模型冻结证据。用于确认商用模型参数、特征名和树结构哈希没有变化。

### `commercial_results_*.csv`

商用模型逐窗口输出。用于查看商用模型在哪些窗口启用 Stage2、哪些窗口判为佩戴、哪些窗口出错。

### `error_features_*.csv`

串联守护模型的候选训练数据。只包含商用阳性且进入 Stage2 的候选窗口。

### `hard_negative_audit/`

hard negative 审计报告。重点看：

- 有多少非佩戴样本被商用模型判为佩戴。
- 这些样本来自哪些 sample。
- hard negative 在候选集中的占比。

### `feature_review/`

特征排序和人工选择材料。用于解释为什么选择某些特征进入新增小模型。

### `corrector_bundle.pkl`

串联守护模型部署包，包含：

- 模型对象。
- 选择的特征。
- 缺失值填充值。
- 阈值。
- 特征来源。

### `evaluation_report.json`

评估摘要，包含商用基线和完整方案指标。

### `evaluation_samples.csv`

样本级评估明细。用于查看每个样本的：

- 商用输出。
- 串联方案输出。
- veto risk。
- guard action。
- 是否 fallback。

### `tree_export/`

模型树结构导出：

- `all_trees.txt`
- `tree_*.json`
- `tree_*.dot`
- `tree_*.png`，需要 Graphviz。
- `model_structure_summary.csv`

### `error_trace/`

错误样本路径追踪：

- `error_samples.csv`
- `error_tree_paths.csv`
- `error_escape_rules.csv`
- `error_escape_rules.md`
- `error_path_node_frequency.png`

用于回答：错误样本是在哪些树节点、哪些分支上逃出的。

## 11. 推荐使用路径

建议上线节奏：

1. 使用 `shadow` 模式跑线上或离线数据。
2. 查看 `hard_negative_audit/`，确认误判佩戴样本是否集中在特定场景。
3. 查看 `feature_review/`，人工确认特征是否可解释、可部署。
4. 查看 `tree_export/` 和 `error_trace/`，确认新增模型没有学到明显异常规则。
5. 如果希望降低误判佩戴但避免强硬拦截，优先尝试 `soft_guard`。
6. `hard_veto` 只用于离线评估、小流量灰度或强安全场景。

## 12. 当前验收结论

当前项目满足：

- 可单独拷贝运行。
- 不依赖 `parallel/` 或父目录脚本。
- 商用模型以 manifest 记录冻结证据。
- 默认 `shadow` 不改变最终商用输出。
- 支持商用基线和完整方案的准确率对比。
- 支持高分辨率 PNG 图片输出。
- 支持树结构可视化和错误样本路径追踪。
- 支持 hard negative 审计。
- 支持人工特征选择。

## 13. 注意事项

- 当前 `dc_threshold` 默认是 `0.3e6`。
- 当前 `AC/DC` 阈值默认是 `1.0`。
- 如果必须和线上旧阈值完全一致，需要显式传入线上阈值，并记录在产物中。
- 当前新增模型不是替代商用模型，而是后置守护层。
- 商用发布前应优先使用 `shadow` 数据做复核，而不是直接使用训练集结论做上线判断。
