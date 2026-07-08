# -*- coding: utf-8 -*-
"""
S08: Train tiny XGBoost veto guard on commercial-positive candidates.

Output: {artifact_dir}/corrector_model.json, corrector_bundle.pkl
"""

import argparse, hashlib, json, os, platform, sys, time
from itertools import product
import numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

LEAKAGE_FEATURES = {
    "target",
    "should_veto",
    "commercial_pred",
    "is_error",
    "fallback",
}


def sha256_head(path, head_bytes=4 * 1024 * 1024):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


def build_training_fingerprint(artifact_dir, feature_pool_train_path=None, splits_path=None):
    fingerprint = {
        "train_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xgboost": xgb.__version__,
        "splits_sha256_head": sha256_head(splits_path or os.path.join(os.path.dirname(artifact_dir), "splits.json")),
        "feature_pool_train_sha256_head": sha256_head(
            feature_pool_train_path or os.path.join(artifact_dir, "feature_pool_train.csv")
        ),
        "selection_policy": {
            "selection_data": "train_only",
            "valid_used_for_selection": False,
            "test_used_for_selection": False,
            "test_role": "final_closed_evaluation_only",
        },
    }
    return fingerprint


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


def parse_model_search_values(raw, cast, name):
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast(part))
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _freeze_params(params):
    return tuple(sorted(params.items()))


def build_xgb_params(scale_pos_weight, n_estimators=10, max_depth=2, learning_rate=0.05,
                     min_child_weight=20, reg_lambda=10, reg_alpha=1, n_jobs=1):
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": int(min_child_weight),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "scale_pos_weight": float(scale_pos_weight),
        "n_jobs": max(1, int(n_jobs)),
        "verbosity": 0,
    }


def build_model_search_candidates(args, scale_pos_weight=1.0):
    axes = {
        "n_estimators": parse_model_search_values(args.model_search_n_estimators, int, "model_search_n_estimators"),
        "max_depth": parse_model_search_values(args.model_search_max_depth, int, "model_search_max_depth"),
        "learning_rate": parse_model_search_values(args.model_search_learning_rate, float, "model_search_learning_rate"),
        "min_child_weight": parse_model_search_values(args.model_search_min_child_weight, int, "model_search_min_child_weight"),
        "reg_lambda": parse_model_search_values(args.model_search_reg_lambda, float, "model_search_reg_lambda"),
        "reg_alpha": parse_model_search_values(args.model_search_reg_alpha, float, "model_search_reg_alpha"),
    }
    keys = list(axes.keys())
    default_params = build_xgb_params(
        scale_pos_weight,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_jobs=getattr(args, "n_jobs", 1),
    )
    grid = []
    for values in product(*(axes[k] for k in keys)):
        params = build_xgb_params(scale_pos_weight, n_jobs=getattr(args, "n_jobs", 1))
        params.update(dict(zip(keys, values)))
        grid.append(params)
    max_candidates = max(1, int(args.model_search_max_candidates))
    if len(grid) > max_candidates:
        rng = np.random.default_rng(int(args.model_search_random_state))
        keep = sorted(rng.choice(len(grid), size=max_candidates, replace=False).tolist())
        grid = [grid[i] for i in keep]
    grid.append(default_params)
    seen = set()
    candidates = []
    for params in grid:
        frozen = _freeze_params(params)
        if frozen in seen:
            continue
        seen.add(frozen)
        candidates.append({
            "rank_input_order": len(candidates) + 1,
            "params": params,
            "is_default_params": frozen == _freeze_params(default_params),
        })
    return candidates


def count_xgb_nodes(model):
    return sum(1 for line in model.get_booster().get_dump() for _ in line.splitlines() if "leaf" in _ or "yes=" in _)


def train_xgb_with_params(params, X, y, Xv=None, yv=None, n_jobs=1):
    fit_params = dict(params)
    fit_params["n_jobs"] = max(1, int(n_jobs))
    model = xgb.XGBClassifier(**fit_params)
    eval_set = [(Xv, yv)] if Xv is not None and yv is not None else None
    model.fit(X, y, eval_set=eval_set, verbose=False)
    return model


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


def evaluate_model_search_candidate(candidate, X_train, y_train, X_valid, y_valid, size_cost=0.002):
    params = candidate["params"]
    model = train_xgb_with_params(params, X_train, y_train, X_valid, y_valid, n_jobs=params.get("n_jobs", 1))
    prob = model.predict_proba(X_valid)[:, 1]
    threshold = select_veto_threshold(y_valid, prob, min_precision=0.95)
    metrics = evaluate(model, X_valid, y_valid, threshold["threshold"])
    nodes = count_xgb_nodes(model)
    score = float(threshold.get("recall", 0.0)) + 0.1 * float(metrics.get("f1", 0.0)) + 0.01 * float(metrics.get("auc", 0.5))
    score -= float(size_cost) * nodes
    record = {
        "rank_input_order": candidate["rank_input_order"],
        "is_default_params": candidate["is_default_params"],
        "score": float(score),
        "threshold": float(threshold["threshold"]),
        "threshold_selection": threshold,
        "valid_metrics": metrics,
        "n_nodes": int(nodes),
        "params": params,
    }
    return model, record


def select_best_model_search_record(records):
    return sorted(
        records,
        key=lambda r: (
            -float(r["score"]),
            int(r["n_nodes"]),
            not bool(r["is_default_params"]),
            int(r["rank_input_order"]),
        ),
    )[0]


def write_model_search_outputs(artifact_dir, records, best_record):
    rows = []
    safe_records = []
    for record in records:
        safe = dict(record)
        safe_records.append(safe)
        row = {
            "rank_input_order": record["rank_input_order"],
            "is_default_params": record["is_default_params"],
            "chosen": record is best_record,
            "score": record["score"],
            "threshold": record["threshold"],
            "n_nodes": record["n_nodes"],
            "valid_accuracy": record["valid_metrics"]["accuracy"],
            "valid_precision": record["valid_metrics"]["precision"],
            "valid_recall": record["valid_metrics"]["recall"],
            "valid_f1": record["valid_metrics"]["f1"],
            "valid_auc": record["valid_metrics"]["auc"],
        }
        row.update({f"param_{k}": v for k, v in record["params"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(artifact_dir, "model_search_results.csv"), index=False)
    summary = {
        "enabled": True,
        "selection_objective": "precision_constrained_veto_recall_with_size_penalty",
        "candidate_count": len(records),
        "best": best_record,
        "top_candidates": sorted(safe_records, key=lambda r: r["score"], reverse=True)[:10],
    }
    with open(os.path.join(artifact_dir, "model_search_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def search_model(args, X_train, y_train, X_valid, y_valid, scale_pos_weight):
    candidates = build_model_search_candidates(args, scale_pos_weight=scale_pos_weight)
    models = []
    records = []
    print(f"Model search: {len(candidates)} tiny candidates")
    for i, candidate in enumerate(candidates, start=1):
        model, record = evaluate_model_search_candidate(
            candidate, X_train, y_train, X_valid, y_valid,
            size_cost=args.model_search_size_cost,
        )
        models.append(model)
        records.append(record)
        print(
            f"  [{i}/{len(candidates)}] trees={record['params']['n_estimators']} "
            f"depth={record['params']['max_depth']} score={record['score']:.4f} "
            f"nodes={record['n_nodes']} f1={record['valid_metrics']['f1']:.4f}"
        )
    best_record = select_best_model_search_record(records)
    best_idx = records.index(best_record)
    summary = write_model_search_outputs(args.artifact_dir, records, best_record)
    return models[best_idx], best_record, summary


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/cascade")
    p.add_argument("--n_estimators", type=int, default=10); p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--model_search_n_estimators", default="6,8,10,12,16,20")
    p.add_argument("--model_search_max_depth", default="1,2,3")
    p.add_argument("--model_search_learning_rate", default="0.03,0.05")
    p.add_argument("--model_search_min_child_weight", default="10,20")
    p.add_argument("--model_search_reg_lambda", default="5,10")
    p.add_argument("--model_search_reg_alpha", default="0,1")
    p.add_argument("--model_search_max_candidates", type=int, default=32)
    p.add_argument("--model_search_random_state", type=int, default=42)
    p.add_argument("--model_search_size_cost", type=float, default=0.002)
    p.add_argument("--manual_features", default=None)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    feats, feature_source = resolve_feature_list(
        os.path.join(args.artifact_dir, "selected_features.json"), args.manual_features
    )
    tp = os.path.join(args.artifact_dir, "error_features_train.csv")
    vp = os.path.join(args.artifact_dir, "error_features_valid.csv")
    if not os.path.exists(tp): print("ERROR: train not found"); sys.exit(1)
    fingerprint = build_training_fingerprint(
        args.artifact_dir,
        os.path.join(args.artifact_dir, "feature_pool_train.csv") if os.path.exists(os.path.join(args.artifact_dir, "feature_pool_train.csv")) else tp,
    )
    with open(os.path.join(args.artifact_dir, "model_fingerprint.json"), "w", encoding="utf-8") as f:
        json.dump(fingerprint, f, indent=2, ensure_ascii=False)
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
               "n_estimators": 0, "max_depth": 0, "n_jobs": max(1, int(args.n_jobs)), "n_nodes": 0,
               "threshold_objective": "constant_single_class_fallback",
               "feature_source": feature_source,
               "fingerprint": fingerprint,
               "selected_features": feats, "threshold": float(thr),
               "fill_values": {k: float(v) for k, v in fills.items()},
               "train_metrics": tm, "valid_metrics": vm}
        with open(os.path.join(args.artifact_dir, "corrector_model.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        joblib.dump({"model": None, "selected_features": feats, "threshold": thr,
                     "fill_values": fills, "constant_probability": constant_probability,
                     "fingerprint": fingerprint, "config": cfg},
                    os.path.join(args.artifact_dir, "corrector_bundle.pkl"))
        print(f"Constant guard fallback: probability={constant_probability:.3f}, reason=single_class_training_labels")
        print(f"Done ({time.time()-t0:.1f}s)")
        return
    model, best_record, model_search_summary = search_model(args, Xt, yt, Xv, yv, sw)
    pv = model.predict_proba(Xv)[:, 1]; best = select_veto_threshold(yv, pv, min_precision=0.95)
    thr = best["threshold"]; tm = evaluate(model, Xt, yt, thr); vm = evaluate(model, Xv, yv, thr)
    nn = count_xgb_nodes(model)
    best_params = best_record["params"]
    print(f"Trees={best_params['n_estimators']} Depth={best_params['max_depth']} Nodes={nn} Thr={thr:.3f}")
    print(f"Threshold objective: veto precision>=0.95, precision={best['precision']:.4f}, recall={best['recall']:.4f}")
    print(f"Train AUC={tm['auc']:.4f} F1={tm['f1']:.4f}  Valid AUC={vm['auc']:.4f} F1={vm['f1']:.4f}")
    model.get_booster().save_model(os.path.join(args.artifact_dir, "corrector_model.json"))
    cfg = {"model_type": "xgboost_veto_guard", "label_col": label_col,
           "n_estimators": int(best_params["n_estimators"]), "max_depth": int(best_params["max_depth"]),
           "learning_rate": float(best_params["learning_rate"]), "n_jobs": max(1, int(args.n_jobs)), "n_nodes": nn,
           "model_search": model_search_summary,
           "threshold_objective": "veto_precision_constrained", "min_veto_precision": 0.95,
           "threshold_selection": best,
           "feature_source": feature_source,
           "fingerprint": fingerprint,
           "selected_features": feats, "threshold": float(thr),
           "fill_values": {k: float(v) for k, v in fills.items()}, "train_metrics": tm, "valid_metrics": vm}
    joblib.dump({"model": model, "selected_features": feats, "threshold": thr, "fill_values": fills,
                 "fingerprint": fingerprint, "config": cfg},
                os.path.join(args.artifact_dir, "corrector_bundle.pkl"))
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
