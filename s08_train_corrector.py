# -*- coding: utf-8 -*-
"""
S08: Train tiny XGBoost veto guard on commercial-positive candidates.

Output: {artifact_dir}/corrector_model.json, corrector_bundle.pkl
"""

import argparse, json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

LEAKAGE_FEATURES = {
    "target",
    "should_veto",
    "commercial_pred",
    "is_error",
    "fallback",
}


class ConstantProbabilityModel:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


def resolve_feature_list(auto_path, manual_path=None):
    source = {"source": "auto", "path": auto_path}
    path = auto_path
    if manual_path and os.path.exists(manual_path):
        path = manual_path
        source = {"source": "manual", "path": manual_path}
    elif manual_path:
        print(f"[WARN] manual feature file not found, using auto: {manual_path}")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    features = [str(x) for x in payload.get("selected_features", [])]
    if not features:
        raise ValueError(f"no selected_features in {path}")
    leaked = [f for f in features if f in LEAKAGE_FEATURES]
    if leaked:
        raise ValueError(f"label leakage features are not allowed in selected_features: {leaked}")
    source["payload"] = payload
    return features, source


def prepare(df, features):
    X = df[features].values.astype(float)
    label_col = "should_veto" if "should_veto" in df.columns else "target"
    y = df[label_col].values.astype(int); fills = {}
    for i, c in enumerate(features):
        ok = np.isfinite(X[:, i]); fills[c] = float(np.median(X[:, i][ok])) if ok.sum() > 0 else 0.0
        X[~ok, i] = fills[c]
    return X, y, fills


def evaluate(model, X, y, thr=0.5):
    p = model.predict_proba(X)[:, 1]; preds = (p >= thr).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {"n": len(y), "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def select_veto_threshold(y_true, prob, min_precision=0.95):
    best = {"threshold": 0.95, "score": -np.inf, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    fallback = None
    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        item = {"threshold": float(t), "score": recall, "precision": precision, "recall": recall, "f1": f1}
        if fallback is None or precision > fallback["precision"] or (
            precision == fallback["precision"] and recall > fallback["recall"]
        ):
            fallback = item
        if precision >= min_precision and recall > best["score"]:
            best = item
    if best["score"] == -np.inf:
        best = fallback if fallback is not None else best
    best["min_precision"] = float(min_precision)
    return best


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/cascade")
    p.add_argument("--n_estimators", type=int, default=10); p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--manual_features", default=None)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    feats, feature_source = resolve_feature_list(
        os.path.join(args.artifact_dir, "selected_features.json"), args.manual_features
    )
    tp = os.path.join(args.artifact_dir, "error_features_train.csv")
    vp = os.path.join(args.artifact_dir, "error_features_valid.csv")
    if not os.path.exists(tp): print("ERROR: train not found"); sys.exit(1)
    dt = pd.read_csv(tp); dv = pd.read_csv(vp) if os.path.exists(vp) else dt.copy()
    label_col = "should_veto" if "should_veto" in dt.columns else "target"
    np_, nn_ = int(dt[label_col].sum()), len(dt) - int(dt[label_col].sum())
    sw = max(0.5, nn_ / max(1, np_)) if np_ > 0 else 1.0
    Xt, yt, fills = prepare(dt, feats); Xv, yv, _ = prepare(dv, feats); t0 = time.time()
    if len(np.unique(yt)) < 2:
        constant_probability = float(yt[0]) if len(yt) else 0.0
        model = ConstantProbabilityModel(constant_probability)
        thr = 0.5
        tm = evaluate(model, Xt, yt, thr)
        vm = evaluate(model, Xv, yv, thr)
        cfg = {"model_type": "constant_veto_guard", "label_col": label_col,
               "reason": "single_class_training_labels",
               "constant_probability": constant_probability,
               "n_estimators": 0, "max_depth": 0, "n_nodes": 0,
               "threshold_objective": "constant_single_class_fallback",
               "feature_source": feature_source,
               "selected_features": feats, "threshold": float(thr),
               "fill_values": {k: float(v) for k, v in fills.items()},
               "train_metrics": tm, "valid_metrics": vm}
        with open(os.path.join(args.artifact_dir, "corrector_model.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        joblib.dump({"model": None, "selected_features": feats, "threshold": thr,
                     "fill_values": fills, "constant_probability": constant_probability,
                     "config": cfg},
                    os.path.join(args.artifact_dir, "corrector_bundle.pkl"))
        print(f"Constant guard fallback: probability={constant_probability:.3f}, reason=single_class_training_labels")
        print(f"Done ({time.time()-t0:.1f}s)")
        return
    model = xgb.XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                              learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=20, reg_lambda=10, reg_alpha=1,
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=42, scale_pos_weight=sw, n_jobs=1)
    model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    pv = model.predict_proba(Xv)[:, 1]; best = select_veto_threshold(yv, pv, min_precision=0.95)
    thr = best["threshold"]; tm = evaluate(model, Xt, yt, thr); vm = evaluate(model, Xv, yv, thr)
    nn = sum(1 for l in model.get_booster().get_dump() if "leaf" in l or "yes=" in l)
    print(f"Trees={args.n_estimators} Depth={args.max_depth} Nodes={nn} Thr={thr:.3f}")
    print(f"Threshold objective: veto precision>=0.95, precision={best['precision']:.4f}, recall={best['recall']:.4f}")
    print(f"Train AUC={tm['auc']:.4f} F1={tm['f1']:.4f}  Valid AUC={vm['auc']:.4f} F1={vm['f1']:.4f}")
    model.get_booster().save_model(os.path.join(args.artifact_dir, "corrector_model.json"))
    cfg = {"model_type": "xgboost_veto_guard", "label_col": label_col,
           "n_estimators": args.n_estimators, "max_depth": args.max_depth, "n_nodes": nn,
           "threshold_objective": "veto_precision_constrained", "min_veto_precision": 0.95,
           "threshold_selection": best,
           "feature_source": feature_source,
           "selected_features": feats, "threshold": float(thr),
           "fill_values": {k: float(v) for k, v in fills.items()}, "train_metrics": tm, "valid_metrics": vm}
    joblib.dump({"model": model, "selected_features": feats, "threshold": thr, "fill_values": fills, "config": cfg},
                os.path.join(args.artifact_dir, "corrector_bundle.pkl"))
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
