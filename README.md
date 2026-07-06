# Cascade 串联商用守护方案

`cascade/` 是一个可以单独拷走运行的串联方案，用于基于 PPG/ACC 的手表佩戴活体检测。它的核心原则是：**完全保留现有商用特征、商用模型参数和商用推理逻辑，只在商用模型后面增加一个很小的风险复核层**。

这个方案适合优先降低“误判佩戴”的商业风险。它不会把商用模型判为未佩戴的样本改成佩戴，只会在商用模型已经判为佩戴的样本中，额外识别是否存在高风险非佩戴样本。

## 新手快速上手

如果你第一次接触这个项目，先按这一节跑通最小流程，再看后面的原理和产物说明。

### 1. 准备目录

把整个 `cascade/` 文件夹拷贝到任意位置都可以运行。推荐目录关系如下：

```text
your_workspace/
    cascade/
        s01_model.py
        s10_pipeline.py
        README.md
        ...
    dataset/
        sample_001.h5
        sample_002.h5
        ...
```

`dataset/` 不一定要放在 `cascade/` 旁边，也可以放在任意磁盘路径，运行时用 `--dataset_dir` 指向它即可。

### 2. 安装 Python 依赖

建议使用 Python 3.9+。在你的 Python 环境中安装：

```bash
pip install numpy pandas scipy scikit-learn xgboost h5py joblib matplotlib
```

可选安装 Graphviz，用于把树结构导出为 PNG。如果没有 Graphviz，项目仍然会输出 `tree_*.json`、`tree_*.dot` 和 `all_trees.txt`。

### 3. 先做 dry-run

进入 `cascade/` 目录：

```bash
cd path\to\cascade
```

先检查命令链路，不真正跑数据：

```bash
python s10_pipeline.py --dataset_dir path\to\dataset --dry_run
```

如果 dry-run 能打印 S05/S06/S07/S08/S09 的命令，说明入口脚本和参数基本正常。

### 4. 跑最小完整流程

推荐先用 `shadow`，因为它不会改变商用输出，只记录新增风险分析：

```bash
python s10_pipeline.py --dataset_dir path\to\dataset --guard_mode shadow
```

需要解释性图片和树结构时再加：

```bash
python s10_pipeline.py --dataset_dir path\to\dataset --guard_mode shadow --explain
```

运行结束后先看这几个文件：

```text
artifacts/cascade/commercial_model_manifest.json
artifacts/cascade/evaluation_report.json
artifacts/cascade/evaluation_comparison.csv
artifacts/cascade/feature_review/ranked_features.md
artifacts/cascade/hard_negative_audit/
```

需要交给工程化同事时，导出独立部署包：

```bash
python s12_export_deploy.py --artifact_dir artifacts/cascade
```

默认会生成：

```text
artifacts/cascade/deploy_export/
```

### 5. 最重要的理解

- `commercial_pred`：只依赖原商用模型的结果。
- `cascade_pred`：商用模型后面增加守护层后的完整方案结果。
- `shadow`：只记录风险，不改变最终输出。
- `hard negative`：真实非佩戴，但商用模型判成佩戴，是本方案最关注的错误。
- `commercial_model_manifest.json`：证明商用特征和模型参数没有被修改。

## 0. 先看这一节：如何理解整个项目

这一节用于快速建立全局认识。后面的章节会逐个解释代码文件、参数、产物和使用方法。

### 0.1 一句话理解 cascade

`cascade` 不是替换商用模型，而是在商用模型后面增加一个很小的“风险复核层”。

```text
商用模型仍然先做决定。
新增模型只看：商用模型已经判为佩戴的样本里，有没有明显像非佩戴的高风险样本。
默认 shadow 模式只记录风险，不改变商用最终输出。
```

### 0.2 端到端运行流程图

```mermaid
flowchart TD
    A["H5 数据集<br/>PPG / ACC / target"] --> B["S10 pipeline<br/>统一调度"]
    B --> C{"是否已有<br/>artifacts/splits.json"}
    C -- "有，且未指定 --force_split" --> D["复用固定 split"]
    C -- "没有，或指定 --force_split" --> E["S04 扫描 H5<br/>stratified train/valid/test split"]
    E --> D
    D --> F["S05 运行冻结商用模型<br/>输出 commercial_results_*.csv<br/>带进度显示"]
    F --> G["S06 提取商用阳性候选<br/>聚焦 hard negative"]
    G --> H["S07 特征排序<br/>生成 ranked_features 和人工模板"]
    H --> I["S08 训练后置小模型<br/>XGBoost 或 constant guard"]
    I --> J["S09 评估<br/>commercial_pred vs cascade_pred"]
    J --> K{"是否加 --explain"}
    K -- "否" --> L["结束：评估报告和部署包"]
    K -- "是" --> M["S11 解释性报告<br/>PNG 图、树结构、错误路径"]
    M --> L
```

关键点：

- split 在 `S05` 之前完成。`S10` 会先生成或复用 `splits.json`，然后才运行商用模型。
- 如果已经存在 `splits.json`，默认不会重新划分，所以日志中会显示复用 split。
- `S05` 时间较长，因为它要逐样本、逐窗口运行冻结商用模型；当前已经增加进度输出，并支持按样本并行。
- `S05` 的并行策略是保守的：默认小于 32 个样本的 split 仍串行，避免 Windows 多进程启动开销；样本数较大时自动启用最多 4 个 worker，也可以通过 `--n_workers` 显式指定。

### 0.3 商用模型冻结边界图

```mermaid
flowchart LR
    subgraph Frozen["冻结商用部分：不改特征、不改参数、不改树"]
        A["s01_model.py"]
        B["商用 8 特征"]
        C["OldLivenessModel"]
        D["TREE_INDEX / TREE_VALUE"]
        E["detect_tree_threshold"]
        A --> B
        A --> C
        A --> D
        A --> E
    end

    subgraph Added["新增部分：只做后置风险复核"]
        F["error_features_*.csv"]
        G["selected_features.json"]
        H["corrector_bundle.pkl"]
        I["guard_action / veto_risk"]
    end

    C --> F
    F --> G
    G --> H
    H --> I
```

验收时优先看：

```text
artifacts/cascade/commercial_model_manifest.json
```

其中 `frozen=true`，并且 `tree_index_sha256`、`tree_value_sha256` 不变，说明商用模型参数保持冻结。

### 0.4 hard negative 是怎么形成的

```mermaid
flowchart TD
    A["所有样本窗口"] --> B["冻结商用模型输出"]
    B --> C{"commercial_pred == 1<br/>且 stage2_enabled == true<br/>且 fallback == false"}
    C -- "否" --> D["不进入 cascade 小模型训练集"]
    C -- "是" --> E["商用阳性候选"]
    E --> F{"target == 0"}
    F -- "是" --> G["hard negative<br/>真实非佩戴，但商用判为佩戴"]
    F -- "否" --> H["正常商用阳性样本"]
    G --> I["should_veto = 1"]
    H --> J["should_veto = 0"]
```

这也是串联方案的核心：它不是在全量数据上重新训练一个替代模型，而是专门盯住最不能接受的错误类型，也就是“误判佩戴”。

### 0.5 guard 模式决策图

```mermaid
flowchart TD
    A["商用输出 commercial_pred"] --> B{"guard_mode"}
    B -- "bypass" --> C["最终输出 = commercial_pred<br/>完全回退"]
    B -- "shadow" --> D["最终输出 = commercial_pred<br/>只记录 veto_risk / guard_action"]
    B -- "soft_guard" --> E{"风险是否持续"}
    E -- "否" --> F["最终输出 = commercial_pred"]
    E -- "是" --> G["仍不直接推翻商用<br/>建议延长检测或进入保守后处理"]
    B -- "hard_veto" --> H{"commercial_pred=1<br/>且风险持续"}
    H -- "否" --> I["最终输出 = commercial_pred"]
    H -- "是" --> J["最终输出可改为 0<br/>仅建议离线或严格灰度"]
```

推荐顺序：

```text
先 shadow -> 再 soft_guard 灰度 -> 最后才考虑 hard_veto
```

### 0.6 人工特征选择闭环图

```mermaid
flowchart TD
    A["第一次运行 pipeline"] --> B["生成 feature_review/ranked_features.*"]
    B --> C["人工查看排序、稳定性、业务可解释性"]
    C --> D["编辑 manual_feature_selection.json"]
    D --> E["第二次运行 pipeline<br/>--manual_features manual_feature_selection.json"]
    E --> F["重新训练 corrector_bundle.pkl"]
    F --> G["重新评估 evaluation_report.json"]
    G --> H["查看 tree_export / error_trace / figures"]
    H --> C
```

这个闭环的目的不是追求训练集指标最高，而是让新增特征满足：

- 能解释。
- 能部署。
- 在 train/valid/test 上表现一致。
- 不依赖标签泄漏字段。
- 对误判佩戴风险有实际帮助。

### 0.7 产物关系和部署文件图

```mermaid
flowchart TD
    A["splits.json<br/>固定数据划分"] --> B["commercial_results_*.csv"]
    B --> C["error_features_*.csv"]
    C --> D["feature_review/ranked_features.*"]
    D --> E["selected_features.json<br/>或 manual_feature_selection.json"]
    E --> F["corrector_bundle.pkl<br/>部署核心文件"]
    F --> G["evaluation_report.json<br/>评估摘要"]
    F --> H["tree_export/<br/>树结构"]
    F --> I["error_trace/<br/>错误路径"]
    F --> J["figures/*.png<br/>可视化结果"]
```

最终部署或交付时，至少保留：

```text
commercial_model_manifest.json
splits.json
selected_features.json
feature_review/manual_feature_selection.json
corrector_model.json
corrector_bundle.pkl
evaluation_report.json
evaluation_comparison.csv
figures/*.png
tree_export/*
error_trace/*
hard_negative_audit/*
skipped_error_features_*.csv
```

注意：`tree_*.png` 依赖系统安装 Graphviz `dot`。如果没有 Graphviz，项目仍会输出 `tree_*.json`、`tree_*.dot` 和 `all_trees.txt`，可解释信息不会丢失。

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
artifacts/cascade/skipped_error_features_train.csv
artifacts/cascade/skipped_error_features_valid.csv
artifacts/cascade/skipped_error_features_test.csv
```

`skipped_error_features_*.csv` 会记录商用阳性候选窗口没有成功抽取特征的原因，例如样本未找到、信号读取失败、窗口越界或特征抽取失败。它用于解释 hard negative 候选数和最终 `error_features_*.csv` 行数不一致的情况。

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

### `s12_export_deploy.py`

部署交接包导出脚本。它把训练产物和部署参考脚本整理成一个可独立传递的目录：

```text
artifacts/cascade/deploy_export/
```

主要输出：

```text
model.json
method.json
selected_features.json
fill_values.json
commercial_model_manifest.json
feature_extractor.py
s02_features.py
commercial_model.py
deploy_inference.py
README_DEPLOY.md
deploy_manifest.json
```

`method.json` 是核心方法配置，包含特征顺序、缺失值填充值、阈值、guard 模式、串联策略和商用模型冻结信息。

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

指定 worker 数运行。`--n_workers` 会用于首次扫描 H5 数据，也会传给 `S05` 做样本级并行：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --guard_mode shadow --n_workers 4
```

如果数据量较小，建议保持默认或显式使用串行，避免多进程启动开销：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --guard_mode shadow --n_workers 1
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

### `skipped_error_features_*.csv`

串联候选特征提取跳过报告。每行包含：

- `split`
- `sample_name`
- `window_idx`
- `reason`
- `detail`

常见 `reason` 包括：

- `sample_not_found`
- `load_signal_failed`
- `window_index_out_of_bounds`
- `window_slice_out_of_range`
- `feature_extraction_failed`

这个文件用于定位数据质量、窗口索引或特征抽取异常，不参与模型训练。

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

### `deploy_export/`

端侧或工程化交接目录。重点文件：

- `model.json`：新增串联守护模型。
- `method.json`：完整部署方法配置。
- `feature_extractor.py`：部署参考特征提取脚本。
- `s02_features.py`：兼容文件名，保证 `commercial_model.py` 内部导入可用。
- `commercial_model.py`：冻结商用模型脚本。
- `commercial_model_manifest.json`：商用冻结证据。
- `deploy_inference.py`：最小 Python 推理参考。
- `deploy_manifest.json`：文件清单和 SHA256。

导出命令：

```bash
python s12_export_deploy.py --artifact_dir artifacts/cascade
```

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

## 附录 A：数据/特征完整报告与人工特征闭环

本项目已经具备数据分析、特征排序、人工特征确认、重新训练和可解释性复核的闭环能力。这里的“完整报告”不是单个文件，而是一组围绕样本、特征、模型和错误路径的产物。

### A.1 当前会自动生成哪些报告

运行完整 pipeline 并打开 `--explain` 后：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --guard_mode shadow --explain
```

会生成以下几类报告。

#### 1. 数据切分报告

位置：

```text
artifacts/splits.json
```

用途：

- 记录 train/valid/test 的样本列表。
- 固定后续所有实验的数据划分。
- 后续默认复用，避免每次运行切分变化。

如果要重新切分：

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --force_split
```

#### 2. 商用模型冻结报告

位置：

```text
artifacts/cascade/commercial_model_manifest.json
```

用途：

- 证明商用模型参数没有被新增方案修改。
- 记录商用特征名、树数量、树节点数、阈值和树数组哈希。
- 用于上线验收时对比 `tree_index_sha256` 和 `tree_value_sha256`。

#### 3. 商用模型逐窗口输出

位置：

```text
artifacts/cascade/commercial_results_train.csv
artifacts/cascade/commercial_results_valid.csv
artifacts/cascade/commercial_results_test.csv
```

用途：

- 查看每个窗口商用模型是否进入 Stage2。
- 查看商用模型的 `pred`、`score`、`fallback`、`is_error`。
- 分析商用模型在哪些样本上出现误判佩戴。

#### 4. hard negative 审计报告

位置：

```text
artifacts/cascade/hard_negative_audit/
```

包含：

```text
hard_negative_summary_train.json/csv/md
hard_negative_summary_valid.json/csv/md
hard_negative_summary_test.json/csv/md
hard_negative_candidates_train.csv
hard_negative_candidates_valid.csv
hard_negative_candidates_test.csv
```

用途：

- 聚焦真实非佩戴但商用模型判为佩戴的样本。
- 统计 hard negative 的窗口数、样本数和占比。
- 帮助判断误判佩戴是否集中在某类数据、某些窗口或某些样本。

#### 5. 特征排序和人工审核报告

位置：

```text
artifacts/cascade/feature_review/
```

包含：

```text
ranked_features.csv
ranked_features.json
ranked_features.md
manual_feature_selection_template.json
```

用途：

- `ranked_features.csv`：适合用 Excel 或脚本查看完整排序。
- `ranked_features.json`：适合程序读取。
- `ranked_features.md`：适合人工阅读。
- `manual_feature_selection_template.json`：供你手工指定最终训练特征。

排序报告会记录每个候选特征的稳定性、训练/验证 AUC、是否自动入选等信息。训练标签和泄漏字段不会进入候选池。

#### 6. 商用基线 vs 完整方案评估报告

位置：

```text
artifacts/cascade/evaluation_report.json
artifacts/cascade/evaluation_samples.csv
artifacts/cascade/evaluation_comparison.csv
```

用途：

- 对比只依赖商用模型的 `commercial_pred` 和完整方案的 `cascade_pred`。
- 查看准确率、precision、recall、F1、混淆矩阵。
- 在 `shadow` 模式下，`cascade_pred` 不改变商用输出，但仍记录风险。

#### 7. 可解释性图片

位置：

```text
artifacts/cascade/figures/*.png
```

当前图片策略：只输出高清 PNG，不输出 PDF、SVG、TIFF。

主要图片包括：

- 商用基线 vs 完整方案指标对比。
- 样本流转 funnel。
- 错误类型分布。
- guard risk 分布。

#### 8. 树结构可视化

位置：

```text
artifacts/cascade/tree_export/
```

包含：

```text
all_trees.txt
tree_*.json
tree_*.dot
tree_*.png
model_structure_summary.csv
```

用途：

- 查看最终小 XGBoost 每棵树的完整结构。
- 检查树深度、分裂特征和阈值是否可解释。
- 如果训练数据只有单一类别，会输出 constant guard 的结构说明，而不是强行训练一棵无意义的树。

#### 9. 错误样本路径追踪

位置：

```text
artifacts/cascade/error_trace/
```

包含：

```text
error_samples.csv
error_tree_paths.csv
error_escape_rules.csv
error_escape_rules.md
error_path_node_frequency.png
```

用途：

- 找出最终仍然错误的样本。
- 记录这些错误样本经过了哪些树、哪些节点、哪些分支。
- 总结高频错误路径，辅助判断模型是否学到了不合理规则。

### A.2 推荐的人工特征选择流程

当前 pipeline 不会在特征排序后自动暂停。因此推荐采用“两次运行”的方式。

#### 第一步：先生成排序报告和人工模板

```bash
cd D:\wearing_liveness\new\new_codex_1\cascade
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --guard_mode shadow --explain
```

这一步会自动完成特征排序、自动选择、训练和评估。第一次训练结果可以作为参考，但不是最终结果。

重点查看：

```text
artifacts/cascade/feature_review/ranked_features.csv
artifacts/cascade/feature_review/ranked_features.md
artifacts/cascade/feature_review/manual_feature_selection_template.json
```

#### 第二步：人工指定最终特征

复制模板：

```text
manual_feature_selection_template.json
```

另存为：

```text
manual_feature_selection.json
```

编辑其中的：

```json
{
  "selected_features": [
    "commercial_score",
    "GREEN_SEG_ACDC_CV",
    "ACC_MAG_MAD"
  ]
}
```

实际特征名必须来自 `ranked_features.csv` 或 `ranked_features.md`。

#### 第三步：使用人工指定特征重新训练和评估

```bash
python s10_pipeline.py --dataset_dir D:\wearing_liveness\dataset --manual_features artifacts/cascade/feature_review/manual_feature_selection.json --guard_mode shadow --explain
```

这次训练会优先使用你指定的 `selected_features`，而不是自动选择结果。

### A.3 人工特征选择的保护机制

训练脚本会拒绝明显的数据泄漏字段。如果手工文件中包含以下字段，会直接报错：

```text
target
should_veto
commercial_pred
is_error
fallback
```

这些字段不能用于模型训练，因为它们直接或间接包含标签、商用预测结果或错误状态。

### A.4 建议人工审核哪些信息

人工选择特征时，建议至少看以下几类信息：

- `ranked_features.md`：排序靠前的特征是否符合业务直觉。
- `ranked_features.csv`：训练集和验证集表现是否一致。
- `hard_negative_audit/`：误判佩戴样本是否足够多，是否集中。
- `evaluation_comparison.csv`：完整方案有没有修复商用错误，同时有没有引入新错误。
- `tree_export/`：树结构是否过度依赖单一特征或异常阈值。
- `error_trace/`：错误样本是否集中在某些分支节点。

### A.5 推荐保留的交付材料

一次完整实验建议至少保存：

```text
commercial_model_manifest.json
splits.json
feature_review/ranked_features.csv
feature_review/ranked_features.md
feature_review/manual_feature_selection.json
selected_features.json
corrector_model.json
corrector_bundle.pkl
evaluation_report.json
evaluation_comparison.csv
figures/*.png
tree_export/*
error_trace/*
hard_negative_audit/*
skipped_error_features_*.csv
```

这样可以完整复现：数据怎么切、特征怎么排、人工选了哪些特征、模型怎么训、最终效果如何、错误样本为什么错。

## 13. 注意事项

- 当前 `dc_threshold` 默认是 `0.3e6`。
- 当前 `AC/DC` 阈值默认是 `1.0`。
- 如果必须和线上旧阈值完全一致，需要显式传入线上阈值，并记录在产物中。
- 当前新增模型不是替代商用模型，而是后置守护层。
- 商用发布前应优先使用 `shadow` 数据做复核，而不是直接使用训练集结论做上线判断。
