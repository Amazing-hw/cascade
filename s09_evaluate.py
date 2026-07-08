# -*- coding: utf-8 -*-
"""
S09: End-to-end evaluation for the cascade soft guard.

Output: {artifact_dir}/evaluation_report.json, evaluation_comparison.csv
"""

import argparse, json, os, time
import numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from s01_model import OldLivenessModel, extract_8_commercial_features, FEATURE_FS, COMMERCIAL_WIN_SEC, COMMERCIAL_STRIDE_SEC
from s02_features import load_ppg, load_acc, get_channels_from_window, detect_green_mode
from s02_features import is_prewindowed_signal, _downsample_ppg, _is_25hz_sample, extract_feature_pool_from_window, validate_h5_file
from s04_data import load_splits


def _to_25hz(s, ppg, acc):
    if _is_25hz_sample(s): return (np.asarray(ppg, dtype=np.float64),
        np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None, 25)
    ppg25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS); acc25 = None
    if acc is not None and len(acc) > 0:
        from scipy.signal import resample_poly
        acc25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg25, acc25, 100


def _prewindow_to_25hz(s, w, ws):
    n = int(w.shape[0])
    if (_is_25hz_sample(s) or n == int(round(float(ws) * FEATURE_FS)) or (n <= 200 and n > 0 and n % FEATURE_FS == 0)):
        return np.asarray(w, dtype=np.float64), 25
    return _downsample_ppg(np.asarray(w, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100


def apply_corrector(score, new_feats, bundle):
    if bundle.get("constant_probability") is not None:
        return float(bundle["constant_probability"])
    feats, fills = bundle["selected_features"], bundle["fill_values"]
    X = np.array([[float(score) if score is not None else -2000.0 if f == "commercial_score"
                    else new_feats.get(f, 0.0) for f in feats]], dtype=float)
    for i, c in enumerate(feats):
        if not np.isfinite(X[0, i]): X[0, i] = fills.get(c, 0.0)
    return float(bundle["model"].predict_proba(X)[0, 1])


def build_feature_matrix(df, features, fills):
    if not features:
        return np.empty((len(df), 0), dtype=float)
    matrix = df.reindex(columns=features)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    X = matrix.to_numpy(dtype=float)
    for i, feature in enumerate(features):
        invalid = ~np.isfinite(X[:, i])
        X[invalid, i] = fills.get(feature, 0.0)
    return X


def predict_corrector_many(df, bundle):
    if bundle.get("constant_probability") is not None:
        return np.full(len(df), float(bundle["constant_probability"]), dtype=float)
    feats, fills = bundle["selected_features"], bundle["fill_values"]
    X = build_feature_matrix(df, feats, fills)
    return np.asarray(bundle["model"].predict_proba(X)[:, 1], dtype=float)


def metric(yt, yp):
    cm = confusion_matrix(yt, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    return {"n": len(yt), "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def confusion_matrix_rows(model_name, metrics):
    c = metrics["confusion"]
    return [
        {"model": model_name, "true_label": 0, "pred_0": int(c["TN"]), "pred_1": int(c["FP"])},
        {"model": model_name, "true_label": 1, "pred_0": int(c["FN"]), "pred_1": int(c["TP"])},
    ]


def print_confusion_matrix(title, metrics):
    c = metrics["confusion"]
    print(f"{title} confusion matrix (rows=true label, cols=pred label)")
    print("              pred_0  pred_1")
    print(f"  true_0      {int(c['TN']):6d}  {int(c['FP']):6d}")
    print(f"  true_1      {int(c['FN']):6d}  {int(c['TP']):6d}")


GUARD_MODES = ("bypass", "shadow", "soft_guard", "hard_veto")


def _row_pred_for_guard_mode(row, mode, min_veto_windows=2, min_veto_ratio=0.4):
    commercial_pred = int(row.get("commercial_pred", 0))
    if mode in {"bypass", "shadow", "soft_guard"} or commercial_pred == 0:
        return commercial_pred
    risk_count = int(row.get("risk_count", 0) or 0)
    risk_ratio = float(row.get("risk_ratio", 0.0) or 0.0)
    if risk_count >= int(min_veto_windows) and risk_ratio >= float(min_veto_ratio):
        return 0
    return commercial_pred


def evaluate_all_guard_modes_from_rows(rows, pred_key="cascade_pred", min_veto_windows=2, min_veto_ratio=0.4):
    results = {}
    y_true = [int(r.get("target", 0)) for r in rows]
    commercial_pred = [int(r.get("commercial_pred", 0)) for r in rows]
    for mode in GUARD_MODES:
        y_pred = [_row_pred_for_guard_mode(r, mode, min_veto_windows, min_veto_ratio) for r in rows]
        metrics = metric(y_true, y_pred) if rows else metric([], [])
        disagreements = [i for i, (c, p) in enumerate(zip(commercial_pred, y_pred)) if c != p]
        fixed = sum(1 for i in disagreements if y_pred[i] == y_true[i] and commercial_pred[i] != y_true[i])
        broken = sum(1 for i in disagreements if commercial_pred[i] == y_true[i] and y_pred[i] != y_true[i])
        results[mode] = {
            "metrics": metrics,
            "n_disagreements": int(len(disagreements)),
            "fixed": int(fixed),
            "broken": int(broken),
        }
    return results


def write_guard_mode_comparison(artifact_dir, rows, pred_key="cascade_pred", min_veto_windows=2, min_veto_ratio=0.4):
    comparison = evaluate_all_guard_modes_from_rows(
        rows,
        pred_key=pred_key,
        min_veto_windows=min_veto_windows,
        min_veto_ratio=min_veto_ratio,
    )
    out_rows = []
    for mode, payload in comparison.items():
        metrics = payload["metrics"]
        out_rows.append({
            "guard_mode": mode,
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "TN": metrics["confusion"]["TN"],
            "FP": metrics["confusion"]["FP"],
            "FN": metrics["confusion"]["FN"],
            "TP": metrics["confusion"]["TP"],
            "n_disagreements": payload["n_disagreements"],
            "fixed": payload["fixed"],
            "broken": payload["broken"],
        })
    pd.DataFrame(out_rows).to_csv(os.path.join(artifact_dir, "evaluation_guard_modes.csv"), index=False)
    with open(os.path.join(artifact_dir, "evaluation_guard_modes.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def _summarize_risks(veto_risks, veto_threshold):
    arr = np.asarray(veto_risks, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.asarray([0.0], dtype=float)
    high = arr >= float(veto_threshold)
    return {
        "veto_risk": float(np.max(arr)),
        "risk_count": int(np.sum(high)),
        "window_count": int(arr.size),
        "risk_ratio": float(np.mean(high)),
    }


def make_guard_decision(
    commercial_pred,
    veto_risks,
    guard_mode="shadow",
    veto_threshold=0.5,
    min_veto_windows=2,
    min_veto_ratio=0.4,
):
    commercial_pred = int(commercial_pred)
    if guard_mode not in GUARD_MODES:
        raise ValueError(f"unknown guard_mode: {guard_mode}")

    summary = _summarize_risks(veto_risks, veto_threshold)
    hard_candidate = (
        commercial_pred == 1
        and summary["risk_count"] >= int(min_veto_windows)
        and summary["risk_ratio"] >= float(min_veto_ratio)
    )
    decision = {
        "final_pred": commercial_pred,
        "guard_action": "pass",
        "decision_source": "commercial",
        **summary,
    }
    if commercial_pred == 0:
        return decision
    if guard_mode == "bypass":
        return decision
    if summary["risk_count"] > 0:
        decision["guard_action"] = "extend_detection"
    if guard_mode == "soft_guard":
        decision["decision_source"] = "soft_guard" if decision["guard_action"] != "pass" else "commercial"
        return decision
    if guard_mode == "shadow":
        return decision
    if hard_candidate:
        decision["final_pred"] = 0
        decision["guard_action"] = "hard_veto"
        decision["decision_source"] = "hard_veto"
    return decision


def apply_guard_decision(commercial_pred, veto_risks, guard_mode="shadow", veto_threshold=0.5):
    return make_guard_decision(
        commercial_pred, veto_risks, guard_mode=guard_mode, veto_threshold=veto_threshold
    )["final_pred"]


def _normal_window_mask(df):
    if "fallback" not in df.columns:
        return pd.Series(True, index=df.index)
    raw = df["fallback"].fillna(False)
    if raw.dtype == bool:
        return ~raw
    text = raw.astype(str).str.strip().str.lower()
    return ~text.isin(["1", "true", "yes", "y"])


def _commercial_probabilities(df):
    if "score" in df.columns:
        com = OldLivenessModel()
        scores = pd.to_numeric(df["score"], errors="coerce")
        probs = scores.apply(lambda x: com.score_to_probability(float(x)) if np.isfinite(x) else np.nan)
        if probs.notna().any():
            return probs.fillna(0.0).to_numpy(dtype=float)
    if "pred" in df.columns:
        return pd.to_numeric(df["pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "commercial_pred" in df.columns:
        return pd.to_numeric(df["commercial_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.zeros(len(df), dtype=float)


def evaluate_cached_feature_rows(
    commercial_df,
    feature_df,
    bundle,
    guard_mode="shadow",
    min_veto_windows=2,
    min_veto_ratio=0.4,
):
    if len(commercial_df) == 0:
        return []
    work = commercial_df.copy()
    feats = feature_df.copy() if feature_df is not None else pd.DataFrame()
    thr = float(bundle["threshold"])
    results = []
    for sn, group in work.groupby("sample_name"):
        target = int(pd.to_numeric(group["target"], errors="coerce").dropna().iloc[0])
        normal = group[_normal_window_mask(group)]
        if len(normal) == 0:
            results.append({"sample_name": sn, "target": target, "commercial_pred": 0,
                            "cascade_pred": 0, "bypass_pred": 0, "fallback": True})
            continue
        cp = int(np.mean(_commercial_probabilities(normal)) >= 0.5)
        feature_group = feats[feats["sample_name"] == sn] if "sample_name" in feats.columns else pd.DataFrame()
        if len(feature_group) > 0:
            feature_group = feature_group[_normal_window_mask(feature_group)]
        Pcas = predict_corrector_many(feature_group, bundle) if len(feature_group) > 0 else np.asarray([0.0])
        decision = make_guard_decision(
            cp, Pcas, guard_mode=guard_mode, veto_threshold=thr,
            min_veto_windows=min_veto_windows, min_veto_ratio=min_veto_ratio
        )
        results.append({"sample_name": sn, "target": target, "commercial_pred": cp,
                        "cascade_pred": decision["final_pred"], "bypass_pred": cp,
                        "veto_risk": decision["veto_risk"], "risk_count": decision["risk_count"],
                        "window_count": decision["window_count"], "risk_ratio": decision["risk_ratio"],
                        "guard_action": decision["guard_action"],
                        "decision_source": decision["decision_source"],
                        "guard_mode": guard_mode, "fallback": False})
    return results


def write_evaluation_outputs(artifact_dir, split, guard_mode, min_veto_windows, min_veto_ratio, results):
    cm = metric([r["target"] for r in results], [r["commercial_pred"] for r in results])
    casm = metric([r["target"] for r in results], [r["cascade_pred"] for r in results])
    disc = [r for r in results if r["commercial_pred"] != r["cascade_pred"]]
    fixed = sum(1 for d in disc if d["cascade_pred"] == d["target"] and d["commercial_pred"] != d["target"])
    broken = sum(1 for d in disc if d["commercial_pred"] == d["target"] and d["cascade_pred"] != d["target"])
    print(f"Commercial: acc={cm['accuracy']:.4f} prec={cm['precision']:.4f} rec={cm['recall']:.4f} f1={cm['f1']:.4f}")
    print_confusion_matrix("Commercial baseline", cm)
    print(f"Cascade:    acc={casm['accuracy']:.4f} prec={casm['precision']:.4f} rec={casm['recall']:.4f} f1={casm['f1']:.4f}")
    print_confusion_matrix("Cascade final", casm)
    print(f"Disagreements: {len(disc)}/{len(results)} (fixed={fixed}, broken={broken})")
    guard_mode_comparison = write_guard_mode_comparison(
        artifact_dir, results, pred_key="cascade_pred",
        min_veto_windows=min_veto_windows, min_veto_ratio=min_veto_ratio,
    )
    report = {"split": split, "n": len(results), "guard_mode": guard_mode,
              "min_veto_windows": min_veto_windows, "min_veto_ratio": min_veto_ratio,
              "commercial": cm, "cascade": casm, "bypass": cm,
              "guard_mode_comparison": guard_mode_comparison,
              "n_disagreements": len(disc), "fixed": fixed, "broken": broken}
    with open(os.path.join(artifact_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(results).to_csv(os.path.join(artifact_dir, "evaluation_samples.csv"), index=False)
    pd.DataFrame([{"metric": m, "commercial": cm[m], "cascade": casm[m], "delta": casm[m] - cm[m]}
                  for m in ["accuracy", "precision", "recall", "f1"]])\
      .to_csv(os.path.join(artifact_dir, "evaluation_comparison.csv"), index=False)
    pd.DataFrame(
        confusion_matrix_rows("commercial", cm) + confusion_matrix_rows("cascade", casm)
    ).to_csv(os.path.join(artifact_dir, "evaluation_confusion_matrices.csv"), index=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/cascade")
    p.add_argument("--splits_dir", default="artifacts"); p.add_argument("--split", default="test")
    p.add_argument("--guard_mode", default="shadow", choices=GUARD_MODES)
    p.add_argument("--min_veto_windows", type=int, default=2)
    p.add_argument("--min_veto_ratio", type=float, default=0.4)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    splits = load_splits(args.splits_dir); samples = splits[args.split]
    bundle = joblib.load(os.path.join(args.artifact_dir, "corrector_bundle.pkl"))
    t0 = time.time()
    commercial_path = os.path.join(args.artifact_dir, f"commercial_results_{args.split}.csv")
    feature_path = os.path.join(args.artifact_dir, f"error_features_{args.split}.csv")
    if os.path.exists(commercial_path):
        print(f"Using cached commercial results: {commercial_path}")
        if os.path.exists(feature_path):
            print(f"Using cached error features: {feature_path}")
            feature_df = pd.read_csv(feature_path)
        else:
            print(f"Cached error features not found: {feature_path}; evaluating commercial-only risk=0")
            feature_df = pd.DataFrame()
        results = evaluate_cached_feature_rows(
            pd.read_csv(commercial_path), feature_df, bundle, guard_mode=args.guard_mode,
            min_veto_windows=args.min_veto_windows, min_veto_ratio=args.min_veto_ratio
        )
        print(f"Inference ({time.time()-t0:.1f}s)")
        write_evaluation_outputs(
            args.artifact_dir, args.split, args.guard_mode,
            args.min_veto_windows, args.min_veto_ratio, results
        )
        print("Done")
        return
    com = OldLivenessModel(); thr = bundle["threshold"]; t0 = time.time(); results = []
    for sample in samples:
        sn, target = sample.get("sample_name", "unknown"), int(sample.get("target", 0))
        try:
            ppg, acc = load_ppg(sample), load_acc(sample)
            ok, err = validate_h5_file(sample["h5_file"], sn)
            if not ok: raise ValueError(err)
        except Exception:
            results.append({"sample_name": sn, "target": target, "commercial_pred": 0, "cascade_pred": 0,
                            "bypass_pred": 0, "fallback": True}); continue
        Pc, Pcas = [], []
        if is_prewindowed_signal(ppg):
            mode = detect_green_mode(ppg)
            for idx in range(3, ppg.shape[0]):
                win25, _ = _prewindow_to_25hz(sample, ppg[idx], COMMERCIAL_WIN_SEC)
                try:
                    ir, amb, g1, g2, g3 = get_channels_from_window(win25, mode)
                    acc_seg = None
                    if acc is not None and is_prewindowed_signal(acc) and idx < acc.shape[0]:
                        acc_seg, _ = _prewindow_to_25hz(sample, acc[idx], COMMERCIAL_WIN_SEC)
                    _, score, _, _ = com.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, acc_seg))
                    pc = com.score_to_probability(score) if score is not None else 1.0
                    nf = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
                    Pc.append(pc); Pcas.append(apply_corrector(score, nf, bundle))
                except Exception: Pc.append(0.0); Pcas.append(0.0)
        else:
            ppg25, acc25, _ = _to_25hz(sample, ppg, acc); mode = detect_green_mode(ppg)
            sw, ss = int(round(COMMERCIAL_WIN_SEC * FEATURE_FS)), int(round(COMMERCIAL_STRIDE_SEC * FEATURE_FS))
            for step in range(3, max(0, (len(ppg25) - sw) // ss + 1)):
                win = ppg25[step * ss:step * ss + sw, :]
                try:
                    ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
                    _, score, _, _ = com.predict_raw(extract_8_commercial_features(ir, amb, g1, g2, g3, None))
                    pc = com.score_to_probability(score) if score is not None else 1.0
                    nf = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
                    Pc.append(pc); Pcas.append(apply_corrector(score, nf, bundle))
                except Exception: Pc.append(0.0); Pcas.append(0.0)
        cp = int(np.mean(Pc) >= 0.5) if Pc else 0
        decision = make_guard_decision(
            cp, Pcas if Pcas else [0.0], guard_mode=args.guard_mode, veto_threshold=thr,
            min_veto_windows=args.min_veto_windows, min_veto_ratio=args.min_veto_ratio
        )
        casp = decision["final_pred"]
        results.append({"sample_name": sn, "target": target, "commercial_pred": cp, "cascade_pred": casp,
                        "bypass_pred": cp, "veto_risk": decision["veto_risk"],
                        "risk_count": decision["risk_count"], "window_count": decision["window_count"],
                        "risk_ratio": decision["risk_ratio"], "guard_action": decision["guard_action"],
                        "decision_source": decision["decision_source"],
                        "guard_mode": args.guard_mode, "fallback": False})
    print(f"Inference ({time.time()-t0:.1f}s)")
    write_evaluation_outputs(
        args.artifact_dir, args.split, args.guard_mode,
        args.min_veto_windows, args.min_veto_ratio, results
    )
    print("Done")

if __name__ == "__main__": main()
