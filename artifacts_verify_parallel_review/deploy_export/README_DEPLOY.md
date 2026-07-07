# Cascade 部署交接包

这个目录是从训练/分析工程中导出的独立部署交接包，可以直接交给工程化同事。

## 文件说明

- `model.json`：新增串联守护模型。可能是 XGBoost JSON，也可能是 constant guard JSON。
- `method.json`：部署方法配置，包含特征顺序、填充值、阈值、guard 模式和串联策略。
- `selected_features.json`：训练时确认的最终特征列表。
- `fill_values.json`：每个特征的缺失值/异常值填充值。
- `feature_extractor.py`：部署侧参考特征提取脚本，来源于项目 `s02_features.py`。
- `s02_features.py`：兼容文件名，供 `commercial_model.py` 内部导入使用。
- `commercial_model.py`：冻结商用模型脚本，来源于项目 `s01_model.py`。
- `commercial_model_manifest.json`：商用模型冻结证据，用于核对树参数和特征是否变化。
- `deploy_inference.py`：最小 Python 推理参考，用于说明模型加载、特征顺序和阈值逻辑。
- `deploy_manifest.json`：导出清单和文件 SHA256。

## 工程化重点

1. 商用模型仍然是主决策，新增模型只在商用阳性候选后做风险复核。
2. 默认 `shadow` 不改变最终输出，只记录风险。
3. 真正上线前应由端侧工程按 `method.json` 重写为目标语言实现，并用本目录文件做一致性核对。
