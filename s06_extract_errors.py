# -*- coding: utf-8 -*-
"""
S06: Extract new feature pool on commercial-positive guard candidates.

Output: {artifact_dir}/error_features_{train,valid,test}.csv
"""

import argparse, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import pandas as pd

from s02_features import load_ppg, load_acc, get_channels_from_window, detect_green_mode
from s02_features import is_prewindowed_signal, _downsample_ppg, _is_25hz_sample
from s02_features import extract_feature_pool_from_window, validate_h5_file
from s04_data import load_splits, multiprocessing_context_from_env, resolve_n_workers

FEATURE_FS = 25
MIN_AUTO_PARALLEL_GROUPS = 32
MIN_CHUNKED_MAP_GROUPS = 200
_THREADPOOL_LIMITER = None


def _format_duration(seconds):
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _progress_interval(total):
    if total <= 20:
        return 1
    return max(1, total // 20)


def _print_progress(split_name, done, total, start_time, ok, skip):
    elapsed = max(1e-9, time.time() - start_time)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = 100.0 * done / total if total else 100.0
    print(
        f"[{split_name}] candidates {done}/{total} ({pct:5.1f}%) "
        f"speed={rate:.2f}/s eta={_format_duration(eta)} ok={ok} skipped={skip}",
        flush=True,
    )


def _resolve_s06_workers(n_workers, total):
    if n_workers is None:
        if total < MIN_AUTO_PARALLEL_GROUPS:
            return 1
        return resolve_n_workers(None, n_items=total)
    return resolve_n_workers(n_workers, n_items=total)


def _init_feature_worker():
    """Limit nested BLAS/NumExpr threads inside each process-pool worker."""
    global _THREADPOOL_LIMITER
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except Exception:
        _THREADPOOL_LIMITER = None


def _parallel_chunksize(total, n_workers):
    return max(1, int(total) // max(1, int(n_workers) * 8))


def _use_chunked_map(total, n_workers):
    return int(n_workers) > 1 and int(total) >= MIN_CHUNKED_MAP_GROUPS


def _normal_bool_series(series, default=False):
    if series is None:
        return pd.Series(default, dtype=bool)
    if series.dtype == bool:
        return series.fillna(default)
    text = series.fillna(default).astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y"])


def build_cascade_training_candidates(comm, include_positive_keep=True):
    if comm is None or len(comm) == 0:
        return pd.DataFrame()
    df = comm.copy()
    pred = pd.to_numeric(df.get("pred", 0), errors="coerce").fillna(0).astype(int)
    target = pd.to_numeric(df.get("target", 0), errors="coerce").fillna(0).astype(int)
    stage2 = _normal_bool_series(df.get("stage2_enabled", pd.Series(True, index=df.index)), default=True)
    fallback = _normal_bool_series(df.get("fallback", pd.Series(False, index=df.index)), default=False)
    base_mask = (~fallback) & stage2
    veto_mask = base_mask & (pred == 1)
    if include_positive_keep:
        keep_mask = base_mask & (target == 1)
        mask = veto_mask | keep_mask
    else:
        mask = veto_mask
    out = df.loc[mask].copy()
    if len(out) == 0:
        return out
    out["target"] = pd.to_numeric(out["target"], errors="coerce").fillna(0).astype(int)
    out["should_veto"] = (out["target"] == 0).astype(int)
    out["candidate_role"] = np.where(out["should_veto"] == 1, "veto_negative", "keep_positive")
    return out


def build_candidate_health_summary(df, split):
    if df is None or len(df) == 0:
        return {
            "split": split,
            "total_candidates": 0,
            "unique_samples": 0,
            "should_veto_1": 0,
            "should_veto_0": 0,
            "commercial_positive_candidates": 0,
            "positive_keep_candidates": 0,
            "is_single_class": True,
        }
    target = pd.to_numeric(df.get("target", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    should_veto = pd.to_numeric(df.get("should_veto", (target == 0).astype(int)), errors="coerce").fillna(0).astype(int)
    pred = pd.to_numeric(df.get("pred", df.get("commercial_pred", pd.Series(0, index=df.index))), errors="coerce").fillna(0).astype(int)
    return {
        "split": split,
        "total_candidates": int(len(df)),
        "unique_samples": int(df["sample_name"].nunique()) if "sample_name" in df.columns else 0,
        "should_veto_1": int((should_veto == 1).sum()),
        "should_veto_0": int((should_veto == 0).sum()),
        "commercial_positive_candidates": int((pred == 1).sum()),
        "positive_keep_candidates": int((target == 1).sum()),
        "is_single_class": bool(should_veto.nunique() < 2),
    }


def write_candidate_health_report(artifact_dir, split, df):
    out_dir = os.path.join(artifact_dir, "candidate_health")
    os.makedirs(out_dir, exist_ok=True)
    summary = build_candidate_health_summary(df, split)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"candidate_health_{split}.csv"), index=False)
    with open(os.path.join(out_dir, f"candidate_health_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if summary["is_single_class"]:
        print(f"[{split}] WARNING: cascade candidates are single-class: {summary}", flush=True)
    else:
        print(f"[{split}] candidate health: {summary}", flush=True)
    return summary


def build_hard_negative_summary(df, split):
    if df is None or len(df) == 0:
        return {
            "split": split,
            "candidate_source": "cascade_training_candidates_commercial_positive_plus_positive_keep",
            "total_candidates": 0,
            "hard_negative_candidates": 0,
            "worn_positive_candidates": 0,
            "hard_negative_rate": 0.0,
            "unique_samples": 0,
            "hard_negative_samples": 0,
        }
    should_veto = pd.to_numeric(df.get("should_veto", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    target = pd.to_numeric(df.get("target", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
    total = int(len(df))
    hard = int(should_veto.sum())
    sample_col = df.get("sample_name", pd.Series(dtype=str))
    hard_samples = int(df.loc[should_veto == 1, "sample_name"].nunique()) if "sample_name" in df else 0
    return {
        "split": split,
        "candidate_source": "cascade_training_candidates_commercial_positive_plus_positive_keep",
        "total_candidates": total,
        "hard_negative_candidates": hard,
        "worn_positive_candidates": int((target == 1).sum()),
        "hard_negative_rate": float(hard / total) if total else 0.0,
        "unique_samples": int(sample_col.nunique()) if len(sample_col) else 0,
        "hard_negative_samples": hard_samples,
    }


def write_hard_negative_audit(artifact_dir, split, df):
    out_dir = os.path.join(artifact_dir, "hard_negative_audit")
    os.makedirs(out_dir, exist_ok=True)
    summary = build_hard_negative_summary(df, split)
    summary_path = os.path.join(out_dir, f"hard_negative_summary_{split}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"hard_negative_summary_{split}.csv"), index=False)
    candidate_path = os.path.join(out_dir, f"hard_negative_candidates_{split}.csv")
    if df is None or len(df) == 0:
        pd.DataFrame().to_csv(candidate_path, index=False)
    else:
        should_veto = pd.to_numeric(df.get("should_veto", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)
        hard = df[should_veto == 1].copy()
        hard.to_csv(candidate_path, index=False)
    with open(os.path.join(out_dir, f"hard_negative_summary_{split}.md"), "w", encoding="utf-8") as f:
        f.write(f"# Hard Negative Audit - {split}\n\n")
        f.write("Source: commercial-positive, Stage2-enabled candidates.\n\n")
        f.write(f"- total_candidates: {summary['total_candidates']}\n")
        f.write(f"- hard_negative_candidates: {summary['hard_negative_candidates']}\n")
        f.write(f"- hard_negative_rate: {summary['hard_negative_rate']:.6f}\n")
        f.write(f"- hard_negative_samples: {summary['hard_negative_samples']}\n")
    return summary


def write_skip_report(artifact_dir, split, rows):
    cols = ["split", "sample_name", "window_idx", "reason", "detail"]
    path = os.path.join(artifact_dir, f"skipped_error_features_{split}.csv")
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def feature_pool_path(artifact_dir, split):
    return os.path.join(artifact_dir, f"feature_pool_{split}.csv")


def build_error_features_from_feature_pool(comm_candidates, feature_pool):
    """Build cascade candidate rows from a cached feature_pool_*.csv when keys match."""
    required = {"sample_name", "window_idx"}
    if comm_candidates is None or feature_pool is None:
        return None
    if not required.issubset(comm_candidates.columns) or not required.issubset(feature_pool.columns):
        return None

    comm = comm_candidates.copy()
    pool = feature_pool.copy()
    comm["window_idx"] = pd.to_numeric(comm["window_idx"], errors="coerce").astype("Int64")
    pool["window_idx"] = pd.to_numeric(pool["window_idx"], errors="coerce").astype("Int64")
    comm = comm.dropna(subset=["window_idx"])
    pool = pool.dropna(subset=["window_idx"])
    if comm.empty or pool.empty:
        return None
    comm["window_idx"] = comm["window_idx"].astype(int)
    pool["window_idx"] = pool["window_idx"].astype(int)

    pool_value_cols = [c for c in pool.columns if c not in {"sample_name", "window_idx"}]
    if not pool_value_cols:
        return None
    pool["_feature_pool_hit"] = 1
    meta_cols = ["sample_name", "window_idx", "target", "pred", "score", "is_error"]
    meta = comm[[c for c in meta_cols if c in comm.columns]].copy()
    merged = meta.merge(pool, on=["sample_name", "window_idx"], how="left", suffixes=("_comm", ""))
    if len(merged) != len(comm) or "_feature_pool_hit" not in merged or merged["_feature_pool_hit"].isna().any():
        return None
    merged = merged.drop(columns=["_feature_pool_hit"])

    if "target_comm" in merged.columns:
        merged["target"] = merged["target_comm"]
        merged = merged.drop(columns=["target_comm"])
    if "target" not in merged.columns and "target_comm" in merged.columns:
        merged["target"] = merged["target_comm"]
    if "target" not in merged.columns:
        return None
    if merged["target"].isna().any():
        return None

    merged["target"] = pd.to_numeric(merged["target"], errors="coerce").fillna(0).astype(int)
    merged["should_veto"] = (merged["target"] == 0).astype(int)
    merged["commercial_pred"] = pd.to_numeric(merged.get("pred", 1), errors="coerce").fillna(1).astype(int)
    merged["commercial_score"] = pd.to_numeric(merged.get("score", -2000.0), errors="coerce").fillna(-2000.0)
    merged["is_error"] = pd.to_numeric(merged.get("is_error", merged["should_veto"]), errors="coerce").fillna(merged["should_veto"]).astype(int)
    merged = merged.drop(columns=[c for c in ["pred", "score"] if c in merged.columns])
    return merged


def _prewindow_to_25hz(sample, window, window_sec):
    n = int(window.shape[0])
    if (_is_25hz_sample(sample) or n == int(round(float(window_sec) * FEATURE_FS))
            or (n <= 200 and n > 0 and n % FEATURE_FS == 0)):
        return np.asarray(window, dtype=np.float64), 25
    return _downsample_ppg(np.asarray(window, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100


def _to_25hz(sample, ppg, acc):
    if _is_25hz_sample(sample):
        return (np.asarray(ppg, dtype=np.float64),
                np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None, 25)
    ppg25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS); acc25 = None
    if acc is not None and len(acc) > 0:
        from scipy.signal import resample_poly
        acc25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg25, acc25, 100


def _process_candidate_group(split, sn, records, sample):
    rows, skip_rows, ok, skip = [], [], 0, 0
    if sample is None:
        for row in records:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": int(row["window_idx"]),
                              "reason": "sample_not_found", "detail": ""})
        return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}
    try:
        ppg, acc = load_ppg(sample), load_acc(sample)
    except Exception as exc:
        for row in records:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": int(row["window_idx"]),
                              "reason": "load_signal_failed", "detail": str(exc)})
        return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}
    try:
        mode = detect_green_mode(ppg)
        prewindowed = is_prewindowed_signal(ppg)
        if not prewindowed:
            ppg25, _acc25, _ = _to_25hz(sample, ppg, acc)
            sw, ss = int(round(5.0 * FEATURE_FS)), int(round(1.0 * FEATURE_FS))
    except Exception as exc:
        for row in records:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": int(row["window_idx"]),
                              "reason": "sample_preprocess_failed", "detail": str(exc)})
        return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}

    for row in records:
        widx = int(row["window_idx"])
        try:
            score = float(row["score"]) if pd.notna(row["score"]) else -2000.0
            if prewindowed:
                if widx >= ppg.shape[0]:
                    skip += 1
                    skip_rows.append({"split": split, "sample_name": sn, "window_idx": widx,
                                      "reason": "window_index_out_of_bounds", "detail": f"n_windows={ppg.shape[0]}"})
                    continue
                win25, _ = _prewindow_to_25hz(sample, ppg[widx], 5.0)
                ir, amb, g1, g2, g3 = get_channels_from_window(win25, mode)
            else:
                if widx * ss + sw > len(ppg25):
                    skip += 1
                    skip_rows.append({"split": split, "sample_name": sn, "window_idx": widx,
                                      "reason": "window_slice_out_of_range", "detail": f"signal_len={len(ppg25)}"})
                    continue
                win = ppg25[widx * ss:widx * ss + sw, :]
                ir, amb, g1, g2, g3 = get_channels_from_window(win, mode)
            feats = extract_feature_pool_from_window(ir, amb, g1, g2, g3, fs=FEATURE_FS)
            target = int(row["target"])
            out = {"sample_name": sn, "target": target, "should_veto": int(target == 0),
                   "commercial_pred": int(row["pred"]), "window_idx": widx,
                   "commercial_score": score, "is_error": int(row["is_error"])}
            out.update(feats)
            rows.append(out)
            ok += 1
        except Exception as exc:
            skip += 1
            skip_rows.append({"split": split, "sample_name": sn, "window_idx": widx,
                              "reason": "feature_extraction_failed", "detail": str(exc)})
    return {"rows": rows, "skip_rows": skip_rows, "ok": ok, "skip": skip, "processed": len(records)}


def _s06_group_worker(args):
    return _process_candidate_group(*args)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifact_dir", default="artifacts/cascade")
    p.add_argument("--splits_dir", default="artifacts")
    p.add_argument("--n_workers", type=int, default=None)
    args = p.parse_args()
    os.makedirs(args.artifact_dir, exist_ok=True)
    splits = load_splits(args.splits_dir)
    sn_map = {}; [sn_map.update({s["sample_name"]: s}) for part in ["train","valid","test"] for s in splits[part]]
    t0 = time.time()
    for name in ["train", "valid", "test"]:
        cp = os.path.join(args.artifact_dir, f"commercial_results_{name}.csv")
        if not os.path.exists(cp): print(f"[{name}] Skipped"); continue
        comm = pd.read_csv(cp)
        errors = build_cascade_training_candidates(comm, include_positive_keep=True)
        write_candidate_health_report(args.artifact_dir, name, errors)
        if len(errors) == 0:
            print(f"[{name}] No cascade training candidates")
            write_skip_report(args.artifact_dir, name, [])
            write_hard_negative_audit(args.artifact_dir, name, pd.DataFrame())
            continue
        fp = feature_pool_path(args.artifact_dir, name)
        if os.path.exists(fp):
            cached_df = build_error_features_from_feature_pool(errors, pd.read_csv(fp))
            if cached_df is not None:
                cached_df.to_csv(os.path.join(args.artifact_dir, f"error_features_{name}.csv"), index=False)
                write_skip_report(args.artifact_dir, name, [])
                write_hard_negative_audit(args.artifact_dir, name, cached_df)
                print(f"[{name}] Reused cached feature pool: {fp} rows={len(cached_df)}")
                continue
            print(f"[{name}] Cached feature pool not usable, falling back to H5 extraction: {fp}")
        rows, skip_rows, ok, skip = [], [], 0, 0
        total_candidates = len(errors)
        interval = _progress_interval(total_candidates)
        split_t0 = time.time()
        processed = 0
        groups = [
            (sn, group.to_dict("records"), sn_map.get(sn))
            for sn, group in errors.groupby("sample_name", sort=False)
        ]
        n_workers = _resolve_s06_workers(args.n_workers, len(groups))
        print(f"[{name}] candidates={total_candidates}, samples={len(groups)}, workers={n_workers}", flush=True)
        if n_workers > 1 and len(groups) > 1:
            pool_kwargs = {"max_workers": n_workers, "initializer": _init_feature_worker}
            mp_ctx = multiprocessing_context_from_env()
            if mp_ctx is not None:
                pool_kwargs["mp_context"] = mp_ctx
            use_chunked = _use_chunked_map(len(groups), n_workers)
            chunksize = _parallel_chunksize(len(groups), n_workers)
            print(f"[{name}] process pool mp_start={mp_ctx.get_start_method()}", flush=True)
            if use_chunked:
                print(f"[{name}] group map chunksize={chunksize}", flush=True)
            with ProcessPoolExecutor(**pool_kwargs) as executor:
                if use_chunked:
                    args_iter = ((name, sn, records, sample) for sn, records, sample in groups)
                    result_iter = executor.map(_s06_group_worker, args_iter, chunksize=chunksize)
                    for result in result_iter:
                        rows.extend(result["rows"])
                        skip_rows.extend(result["skip_rows"])
                        ok += int(result["ok"])
                        skip += int(result["skip"])
                        processed += int(result["processed"])
                        if processed == total_candidates or processed % interval == 0 or processed - int(result["processed"]) == 0:
                            _print_progress(name, processed, total_candidates, split_t0, ok, skip)
                else:
                    futures = [
                        executor.submit(_s06_group_worker, (name, sn, records, sample))
                        for sn, records, sample in groups
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        rows.extend(result["rows"])
                        skip_rows.extend(result["skip_rows"])
                        ok += int(result["ok"])
                        skip += int(result["skip"])
                        processed += int(result["processed"])
                        if processed == total_candidates or processed % interval == 0 or processed - int(result["processed"]) == 0:
                            _print_progress(name, processed, total_candidates, split_t0, ok, skip)
        else:
            for sn, records, sample in groups:
                result = _process_candidate_group(name, sn, records, sample)
                rows.extend(result["rows"])
                skip_rows.extend(result["skip_rows"])
                ok += int(result["ok"])
                skip += int(result["skip"])
                processed += int(result["processed"])
                if processed == total_candidates or processed % interval == 0 or processed - int(result["processed"]) == 0:
                    _print_progress(name, processed, total_candidates, split_t0, ok, skip)
        print(f"[{name}] Extracted={ok} skipped={skip}")
        df = pd.DataFrame(rows)
        if rows:
            df.to_csv(os.path.join(args.artifact_dir, f"error_features_{name}.csv"), index=False)
            df.to_csv(fp, index=False)
        write_skip_report(args.artifact_dir, name, skip_rows)
        write_hard_negative_audit(args.artifact_dir, name, df)
    print(f"Done ({time.time()-t0:.1f}s)")

if __name__ == "__main__": main()
