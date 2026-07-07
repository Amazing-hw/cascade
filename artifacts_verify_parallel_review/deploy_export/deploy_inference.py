# -*- coding: utf-8 -*-
"""
Minimal deployment reference for the exported cascade/parallel guard package.

This file documents the feature order, fill-value behavior, and JSON model load
path for engineering integration. It is intentionally small and independent of
the training pipeline.
"""

import json
from pathlib import Path

import numpy as np


def load_method(package_dir="."):
    package_dir = Path(package_dir)
    with open(package_dir / "method.json", encoding="utf-8") as f:
        return json.load(f)


def build_feature_vector(feature_dict, method):
    values = []
    fills = method.get("fill_values", {})
    for name in method["selected_features"]:
        raw = feature_dict.get(name, fills.get(name, 0.0))
        try:
            value = float(raw)
        except Exception:
            value = float(fills.get(name, 0.0))
        if not np.isfinite(value):
            value = float(fills.get(name, 0.0))
        values.append(value)
    return np.asarray(values, dtype=float).reshape(1, -1)


def predict_guard_probability(feature_dict, package_dir="."):
    package_dir = Path(package_dir)
    method = load_method(package_dir)
    constant_probability = method["model"].get("constant_probability")
    if constant_probability is not None:
        return float(constant_probability)

    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(package_dir / "model.json"))
    x = build_feature_vector(feature_dict, method)
    return float(booster.predict(xgb.DMatrix(x))[0])


def apply_guard(commercial_pred, feature_dict, package_dir=".", guard_mode=None):
    method = load_method(package_dir)
    guard_mode = guard_mode or method["guard"]["default_mode"]
    p_guard = predict_guard_probability(feature_dict, package_dir)
    threshold = float(method["model"]["threshold"])
    commercial_pred = int(commercial_pred)

    if method["project_type"] == "cascade":
        if commercial_pred == 0 or guard_mode in ("bypass", "shadow", "soft_guard"):
            return {"final_pred": commercial_pred, "guard_probability": p_guard, "guard_action": "record"}
        if guard_mode == "hard_veto" and p_guard >= threshold:
            return {"final_pred": 0, "guard_probability": p_guard, "guard_action": "hard_veto"}
        return {"final_pred": commercial_pred, "guard_probability": p_guard, "guard_action": "pass"}

    raise ValueError(f"unsupported project_type for this package: {method['project_type']}")
